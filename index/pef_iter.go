// Copyright 2026 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package index

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"

	"github.com/sourcegraph/zoekt"
)

var _ hitIterator = (*pefPostingIterator)(nil)

// pefPostingIterator implements hitIterator directly over an mmap-backed
// PISA PEF flat-blob with zero heap allocations during iteration and skipping.
type pefPostingIterator struct {
	blob         []byte
	n            uint32
	numChunks    int
	firstKey     uint32
	payloadStart uint64
	dirNKindOff  uint64
	dirOffOff    uint64

	curVal      uint32
	chunkIdx    int
	chunkBase   uint32
	chunkLast   uint32
	chunkCount  uint32
	chunkKind   pefChunkKind
	chunkLoaded bool

	// Active chunk payload sub-slices (zero-copy views into blob).
	bmPayload     []byte
	efSamples     []byte
	efHighBits    []byte
	efLowBits     []byte
	efHighBitsLen uint64
	efLowMask     uint64
	efEll         uint8
	efSeeked      bool
	keyIdxInChunk uint32
	curHigh       uint32
	curHighBitPos uint64

	indexBytesLoaded int
	ngramLookups     int
	what             ngram
}

func newPEFPostingIterator(blob []byte, w ngram) *pefPostingIterator {
	if len(blob) < pefHeaderSize {
		panic("corrupt pef posting: truncated header")
	}
	n := binary.LittleEndian.Uint32(blob[0:4])
	numChunks := binary.LittleEndian.Uint32(blob[4:8])
	firstKey := binary.LittleEndian.Uint32(blob[8:12])
	if n == 0 {
		panic("corrupt pef posting: zero postings")
	}
	if numChunks == 0 {
		panic("corrupt pef posting: zero chunks")
	}
	if numChunks > n {
		panic("corrupt pef posting: numChunks exceeds n")
	}
	payloadStart := uint64(pefHeaderSize) + 12*uint64(numChunks)
	if payloadStart > uint64(len(blob)) {
		panic("corrupt pef posting: truncated directory")
	}

	c := int(numChunks)
	it := &pefPostingIterator{
		blob:             blob,
		n:                n,
		numChunks:        c,
		firstKey:         firstKey,
		payloadStart:     payloadStart,
		dirNKindOff:      uint64(pefHeaderSize) + 4*uint64(c),
		dirOffOff:        uint64(pefHeaderSize) + 8*uint64(c),
		curVal:           firstKey,
		indexBytesLoaded: pefHeaderSize,
		what:             w,
	}

	// Validate chunk 0 directory entry on construction while keeping chunk 0
	// payload unloaded until actually entered.
	last0 := it.readChunkLastRaw(0)
	nKind0 := it.readChunkNKindRaw(0)
	off0 := it.readChunkOffRaw(0)
	it.indexBytesLoaded += 12

	count0 := nKind0 >> 2
	kind0 := pefChunkKind(nKind0 & 3)
	if kind0 > pefKindBitmap || count0 == 0 || count0 > n || last0 < firstKey || off0 != 0 {
		panic("corrupt pef posting: invalid chunk 0 directory entry")
	}
	lastRel0 := last0 - firstKey
	if lastRel0 < count0-1 {
		panic("corrupt pef posting: chunk 0 range smaller than count")
	}
	if kind0 == pefKindAllOnes && lastRel0 != count0-1 {
		panic("corrupt pef posting: chunk 0 all-ones range mismatch")
	}
	want0 := pefChunkPayloadSize(kind0, lastRel0, count0)
	payloadLen := uint64(len(blob)) - payloadStart
	if c == 1 {
		if count0 != n {
			panic("corrupt pef posting: single chunk count mismatch")
		}
		if want0 != payloadLen {
			panic("corrupt pef posting: chunk 0 payload size mismatch")
		}
	} else if want0 > payloadLen {
		panic("corrupt pef posting: chunk 0 payload exceeds blob")
	}
	it.chunkLast = last0
	return it
}

func (i *pefPostingIterator) setNgramLookups(n int) {
	i.ngramLookups = n
}

func (i *pefPostingIterator) String() string {
	return fmt.Sprintf("pef(%s, %d, [%d bytes, %d chunks])", i.what, i.curVal, len(i.blob), i.numChunks)
}

func (i *pefPostingIterator) first() uint32 {
	return i.curVal
}

func (i *pefPostingIterator) updateStats(s *zoekt.Stats) {
	s.IndexBytesLoaded += int64(i.indexBytesLoaded)
	s.NgramLookups += i.ngramLookups
	i.indexBytesLoaded = 0
	i.ngramLookups = 0
}

func (i *pefPostingIterator) next(limit uint32) {
	if limit == math.MaxUint32 {
		i.exhaust()
		return
	}
	if limit < i.curVal {
		return
	}
	target := limit + 1
	if target > i.chunkLast {
		if !i.advanceToChunk(target) {
			i.exhaust()
			return
		}
	} else if !i.chunkLoaded {
		i.loadChunk(0)
	}
	i.seekInChunk(target)
}

func (i *pefPostingIterator) exhaust() {
	i.curVal = math.MaxUint32
	i.chunkIdx = i.numChunks
	i.chunkLoaded = false
	i.efSeeked = false
}

func (i *pefPostingIterator) readChunkLastRaw(c int) uint32 {
	off := pefHeaderSize + 4*c
	return binary.LittleEndian.Uint32(i.blob[off : off+4])
}

func (i *pefPostingIterator) readChunkLast(c int) uint32 {
	i.indexBytesLoaded += 4
	return i.readChunkLastRaw(c)
}

func (i *pefPostingIterator) readChunkNKindRaw(c int) uint32 {
	off := int(i.dirNKindOff) + 4*c
	return binary.LittleEndian.Uint32(i.blob[off : off+4])
}

func (i *pefPostingIterator) readChunkOffRaw(c int) uint32 {
	off := int(i.dirOffOff) + 4*c
	return binary.LittleEndian.Uint32(i.blob[off : off+4])
}

func pefChunkPayloadSize(kind pefChunkKind, lastRel, count uint32) uint64 {
	switch kind {
	case pefKindAllOnes:
		return 0
	case pefKindBitmap:
		return (uint64(lastRel) + 8) / 8
	case pefKindEF:
		var ell uint8
		if lastRel >= count {
			ell = uint8(bits.Len32(lastRel/count) - 1)
		}
		numZeros := uint64(lastRel >> ell)
		numSamples := numZeros / pefSelect0SampleRate
		sampleBytes := numSamples * 4
		highBitsLen := uint64(count) + numZeros
		highBytes := (highBitsLen + 7) / 8
		lowBytes := (uint64(count)*uint64(ell) + 7) / 8
		return sampleBytes + highBytes + lowBytes
	default:
		panic("corrupt pef posting: invalid chunk kind")
	}
}

func (i *pefPostingIterator) advanceToChunk(target uint32) bool {
	lo := i.chunkIdx + 1
	if lo >= i.numChunks {
		return false
	}

	// Sequential chunk transition fast-path: check the immediate next chunk first.
	v := i.readChunkLast(lo)
	if v <= i.chunkLast {
		panic("corrupt pef posting: non-monotonic chunkLast")
	}
	if target <= v {
		i.loadChunk(lo)
		return true
	}

	// Galloping forward search across chunkLast[lo+1:].
	prevIdx := lo
	prevVal := v
	step := 2
	var hi int
	for {
		if step >= i.numChunks-prevIdx {
			lastIdx := i.numChunks - 1
			if lastIdx == prevIdx {
				return false
			}
			lastVal := i.readChunkLast(lastIdx)
			if lastVal <= prevVal {
				panic("corrupt pef posting: non-monotonic chunkLast")
			}
			if target > lastVal {
				return false
			}
			lo = prevIdx + 1
			hi = lastIdx
			break
		}
		probe := prevIdx + step
		probeVal := i.readChunkLast(probe)
		if probeVal <= prevVal {
			panic("corrupt pef posting: non-monotonic chunkLast")
		}
		if target <= probeVal {
			lo = prevIdx + 1
			hi = probe
			break
		}
		prevIdx = probe
		prevVal = probeVal
		step <<= 1
	}

	// Binary search [lo, hi] for the first chunk with chunkLast >= target.
	for lo < hi {
		mid := lo + (hi-lo)/2
		if i.readChunkLast(mid) < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	i.loadChunk(lo)
	return true
}

func (i *pefPostingIterator) loadChunk(c int) {
	if c != 0 {
		i.indexBytesLoaded += 8
	}

	last := i.readChunkLastRaw(c)
	nKind := i.readChunkNKindRaw(c)
	off := i.readChunkOffRaw(c)

	count := nKind >> 2
	kind := pefChunkKind(nKind & 3)
	if kind > pefKindBitmap {
		panic("corrupt pef posting: invalid chunk kind")
	}
	if count == 0 || count > i.n {
		panic("corrupt pef posting: invalid chunk count")
	}

	var base uint32
	if c == 0 {
		base = i.firstKey
		if last < base {
			panic("corrupt pef posting: chunkLast below firstKey")
		}
		if off != 0 {
			panic("corrupt pef posting: non-zero first chunk offset")
		}
	} else {
		prevLast := i.readChunkLastRaw(c - 1)
		if c-1 != i.chunkIdx {
			i.indexBytesLoaded += 4
		}
		if prevLast >= last {
			panic("corrupt pef posting: non-monotonic chunkLast")
		}
		if c > i.chunkIdx+1 && prevLast <= i.chunkLast {
			panic("corrupt pef posting: non-monotonic chunkLast")
		}
		base = prevLast + 1
		if base <= i.firstKey || base <= i.curVal {
			panic("corrupt pef posting: non-monotonic chunk base")
		}
	}

	lastRel := last - base
	if lastRel < count-1 {
		panic("corrupt pef posting: chunk range smaller than count")
	}

	payloadLen := uint64(len(i.blob)) - i.payloadStart
	if uint64(off) > payloadLen {
		panic("corrupt pef posting: chunk offset out of bounds")
	}
	var nextOff uint64
	if c+1 < i.numChunks {
		nextOff = uint64(i.readChunkOffRaw(c + 1))
		if nextOff < uint64(off) || nextOff > payloadLen {
			panic("corrupt pef posting: next chunk offset out of bounds")
		}
	} else {
		nextOff = payloadLen
	}
	avail := nextOff - uint64(off)
	absOff := i.payloadStart + uint64(off)

	switch kind {
	case pefKindAllOnes:
		if lastRel != count-1 {
			panic("corrupt pef posting: all-ones range mismatch")
		}
		if avail != 0 {
			panic("corrupt pef posting: all-ones non-empty payload")
		}
	case pefKindBitmap:
		wantSize := (uint64(lastRel) + 8) / 8
		if avail != wantSize {
			panic("corrupt pef posting: bitmap payload size mismatch")
		}
		bm := i.blob[absOff : absOff+wantSize]
		if (bm[lastRel>>3]>>(lastRel&7))&1 == 0 {
			panic("corrupt pef posting: bitmap missing last key")
		}
		i.bmPayload = bm
	case pefKindEF:
		var ell uint8
		if lastRel >= count {
			ell = uint8(bits.Len32(lastRel/count) - 1)
		}
		numZeros := uint64(lastRel >> ell)
		numSamples := numZeros / pefSelect0SampleRate
		sampleBytes := numSamples * 4
		highBitsLen := uint64(count) + numZeros
		highBytes := (highBitsLen + 7) / 8
		lowBytes := (uint64(count)*uint64(ell) + 7) / 8
		wantSize := sampleBytes + highBytes + lowBytes
		if avail != wantSize {
			panic("corrupt pef posting: EF payload size mismatch")
		}
		payload := i.blob[absOff : absOff+wantSize]
		i.efSamples = payload[:sampleBytes]
		i.efHighBits = payload[sampleBytes : sampleBytes+highBytes]
		i.efLowBits = payload[sampleBytes+highBytes:]
		i.efHighBitsLen = highBitsLen
		i.efEll = ell
		if ell > 0 {
			i.efLowMask = (uint64(1) << ell) - 1
		} else {
			i.efLowMask = 0
		}
		if c == 0 {
			if (i.efHighBits[0] & 1) == 0 {
				panic("corrupt pef posting: EF chunk 0 missing first key")
			}
			i.efSeeked = true
			i.keyIdxInChunk = 0
			i.curHigh = 0
			i.curHighBitPos = 0
		} else {
			i.efSeeked = false
		}
	}

	i.indexBytesLoaded += int(avail)
	i.chunkIdx = c
	i.chunkBase = base
	i.chunkLast = last
	i.chunkKind = kind
	i.chunkCount = count
	i.chunkLoaded = true
}

func (i *pefPostingIterator) seekInChunk(target uint32) {
	if target < i.chunkBase || target > i.chunkLast {
		panic("corrupt pef posting: target outside chunk range")
	}
	switch i.chunkKind {
	case pefKindAllOnes:
		i.curVal = target
	case pefKindBitmap:
		i.seekBitmap(target)
	case pefKindEF:
		i.seekEF(target)
	}
}

func (i *pefPostingIterator) seekBitmap(target uint32) {
	lastRel := i.chunkLast - i.chunkBase
	rel := target - i.chunkBase
	bm := i.bmPayload

	wordIdx := int(rel >> 6)
	bitShift := uint(rel & 63)
	byteOff := wordIdx * 8
	if byteOff >= len(bm) {
		panic("corrupt pef posting: bitmap seek out of bounds")
	}

	w := readPEFWord64(bm, byteOff) & (^uint64(0) << bitShift)
	for w == 0 {
		wordIdx++
		byteOff += 8
		if byteOff >= len(bm) {
			panic("corrupt pef posting: bitmap missing target key")
		}
		w = readPEFWord64(bm, byteOff)
	}

	bitPos := uint64(wordIdx)*64 + uint64(bits.TrailingZeros64(w))
	if bitPos > uint64(lastRel) {
		panic("corrupt pef posting: bitmap key past lastRel")
	}
	i.curVal = i.chunkBase + uint32(bitPos)
}

func (i *pefPostingIterator) seekEF(target uint32) {
	// Fast path 1: sequential advance next(curVal).
	if i.efSeeked && target == i.curVal+1 {
		i.advanceToEFKey(i.keyIdxInChunk+1, i.curHighBitPos+1, target)
		return
	}

	rel := target - i.chunkBase
	targetHigh := rel >> i.efEll
	lowTarget := uint64(rel) & i.efLowMask

	// Fast path 2: already positioned in this bucket.
	if i.efSeeked && targetHigh == i.curHigh {
		bucketEndPos := nextZeroPos(i.efHighBits, i.efHighBitsLen, i.curHighBitPos+1)
		afterCount64 := bucketEndPos - (i.curHighBitPos + 1)
		startIdx := i.keyIdxInChunk + 1
		if uint64(startIdx)+afterCount64 > uint64(i.chunkCount) {
			panic("corrupt pef posting: EF too many 1-bits in bucket")
		}
		endIdx := startIdx + uint32(afterCount64)
		if idx, low, found := i.searchEFBucket(startIdx, endIdx, lowTarget); found {
			val64 := uint64(i.chunkBase) + (uint64(targetHigh) << i.efEll) + low
			if val64 < uint64(target) || val64 > uint64(i.chunkLast) {
				panic("corrupt pef posting: EF key out of bounds")
			}
			i.curVal = uint32(val64)
			i.curHighBitPos = i.curHighBitPos + 1 + uint64(idx-startIdx)
			i.keyIdxInChunk = idx
			return
		}
		i.advanceToEFKey(endIdx, bucketEndPos+1, target)
		return
	}

	// Jump path: use curHighBitPos within the same 256-zero sample block, or select0Samples.
	var pos uint64
	var zerosToSkip uint64
	if i.efSeeked && (targetHigh>>8) == (i.curHigh>>8) && targetHigh > i.curHigh {
		pos = i.curHighBitPos + 1
		zerosToSkip = uint64(targetHigh - i.curHigh)
	} else {
		sampleNum := targetHigh / pefSelect0SampleRate
		if sampleNum > 0 {
			sampleOff := int(sampleNum-1) * 4
			if sampleOff+4 > len(i.efSamples) {
				panic("corrupt pef posting: EF select0 sample out of bounds")
			}
			samplePos := uint64(binary.LittleEndian.Uint32(i.efSamples[sampleOff : sampleOff+4]))
			pos = samplePos + 1
			if pos < uint64(sampleNum)*pefSelect0SampleRate || pos >= i.efHighBitsLen {
				panic("corrupt pef posting: EF corrupt select0 sample")
			}
		}
		zerosToSkip = uint64(targetHigh % pefSelect0SampleRate)
	}

	bucketStartPos := skipZeros(i.efHighBits, i.efHighBitsLen, pos, zerosToSkip)
	if bucketStartPos >= i.efHighBitsLen || bucketStartPos < uint64(targetHigh) {
		panic("corrupt pef posting: EF bucket start out of bounds")
	}
	bucketIdx64 := bucketStartPos - uint64(targetHigh)
	if bucketIdx64 >= uint64(i.chunkCount) {
		panic("corrupt pef posting: EF bucket index exceeds count")
	}
	bucketIdx := uint32(bucketIdx64)

	bucketEndPos := nextZeroPos(i.efHighBits, i.efHighBitsLen, bucketStartPos)
	bucketLen64 := bucketEndPos - bucketStartPos
	if bucketIdx64+bucketLen64 > uint64(i.chunkCount) {
		panic("corrupt pef posting: EF too many 1-bits")
	}
	endIdx := bucketIdx + uint32(bucketLen64)

	if idx, low, found := i.searchEFBucket(bucketIdx, endIdx, lowTarget); found {
		val64 := uint64(i.chunkBase) + (uint64(targetHigh) << i.efEll) + low
		if val64 < uint64(target) || val64 > uint64(i.chunkLast) {
			panic("corrupt pef posting: EF bucket key out of bounds")
		}
		i.curVal = uint32(val64)
		i.keyIdxInChunk = idx
		i.curHigh = targetHigh
		i.curHighBitPos = bucketStartPos + uint64(idx-bucketIdx)
		i.efSeeked = true
		return
	}

	i.advanceToEFKey(endIdx, bucketEndPos+1, target)
}

func (i *pefPostingIterator) advanceToEFKey(idx uint32, fromBitPos uint64, target uint32) {
	if idx >= i.chunkCount {
		panic("corrupt pef posting: EF missing key")
	}
	nextPos := nextOnePos(i.efHighBits, i.efHighBitsLen, fromBitPos)
	nextHigh := nextPos - uint64(idx)
	low := i.readEFLow(idx)
	val64 := uint64(i.chunkBase) + (nextHigh << i.efEll) + low
	if val64 < uint64(target) || val64 > uint64(i.chunkLast) {
		panic("corrupt pef posting: EF key out of bounds")
	}
	i.curVal = uint32(val64)
	i.keyIdxInChunk = idx
	i.curHigh = uint32(nextHigh)
	i.curHighBitPos = nextPos
	i.efSeeked = true
}

func (i *pefPostingIterator) searchEFBucket(startIdx, endIdx uint32, lowTarget uint64) (uint32, uint64, bool) {
	if startIdx >= endIdx {
		return 0, 0, false
	}
	if i.efEll == 0 {
		if lowTarget == 0 {
			return startIdx, 0, true
		}
		return 0, 0, false
	}
	if endIdx-startIdx <= 64 {
		for idx := startIdx; idx < endIdx; idx++ {
			low := i.readEFLow(idx)
			if low >= lowTarget {
				return idx, low, true
			}
		}
		return 0, 0, false
	}
	lo, hi := startIdx, endIdx
	for lo < hi {
		mid := lo + (hi-lo)/2
		if i.readEFLow(mid) < lowTarget {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < endIdx {
		return lo, i.readEFLow(lo), true
	}
	return 0, 0, false
}

func (i *pefPostingIterator) readEFLow(idx uint32) uint64 {
	if i.efEll == 0 {
		return 0
	}
	bitPos := uint64(idx) * uint64(i.efEll)
	byteOff := int(bitPos >> 3)
	bitShift := uint(bitPos & 7)
	return (readPEFWord64(i.efLowBits, byteOff) >> bitShift) & i.efLowMask
}

func readPEFWord64(buf []byte, byteOff int) uint64 {
	if byteOff+8 <= len(buf) {
		return binary.LittleEndian.Uint64(buf[byteOff : byteOff+8])
	}
	var w uint64
	for j, b := range buf[byteOff:] {
		w |= uint64(b) << (8 * uint(j))
	}
	return w
}

func skipZeros(highBits []byte, highBitsLen, pos, k uint64) uint64 {
	if k == 0 {
		return pos
	}
	if pos >= highBitsLen {
		panic("corrupt pef posting: EF select0 start out of bounds")
	}
	wordIdx := int(pos >> 6)
	bitShift := uint(pos & 63)
	byteOff := wordIdx * 8
	if byteOff >= len(highBits) {
		panic("corrupt pef posting: EF highBits out of bounds")
	}
	zeros := (^readPEFWord64(highBits, byteOff)) & (^uint64(0) << bitShift)
	for {
		cnt := uint64(bits.OnesCount64(zeros))
		if cnt >= k {
			for k > 1 {
				zeros &= zeros - 1
				k--
			}
			bitPos := uint64(wordIdx)*64 + uint64(bits.TrailingZeros64(zeros))
			if bitPos+1 >= highBitsLen {
				panic("corrupt pef posting: EF select0 past highBitsLen")
			}
			return bitPos + 1
		}
		k -= cnt
		wordIdx++
		byteOff += 8
		if byteOff >= len(highBits) {
			panic("corrupt pef posting: EF select0 missing zeros")
		}
		zeros = ^readPEFWord64(highBits, byteOff)
	}
}

func nextOnePos(highBits []byte, highBitsLen, pos uint64) uint64 {
	if pos >= highBitsLen {
		panic("corrupt pef posting: EF nextOnePos out of bounds")
	}
	wordIdx := int(pos >> 6)
	bitShift := uint(pos & 63)
	byteOff := wordIdx * 8
	if byteOff >= len(highBits) {
		panic("corrupt pef posting: EF highBits out of bounds")
	}
	w := readPEFWord64(highBits, byteOff) & (^uint64(0) << bitShift)
	for w == 0 {
		wordIdx++
		byteOff += 8
		if byteOff >= len(highBits) {
			panic("corrupt pef posting: EF missing 1-bit")
		}
		w = readPEFWord64(highBits, byteOff)
	}
	res := uint64(wordIdx)*64 + uint64(bits.TrailingZeros64(w))
	if res >= highBitsLen {
		panic("corrupt pef posting: EF 1-bit past highBitsLen")
	}
	return res
}

func nextZeroPos(highBits []byte, highBitsLen, pos uint64) uint64 {
	if pos >= highBitsLen {
		return highBitsLen
	}
	wordIdx := int(pos >> 6)
	bitShift := uint(pos & 63)
	byteOff := wordIdx * 8
	if byteOff >= len(highBits) {
		return highBitsLen
	}
	z := (^readPEFWord64(highBits, byteOff)) & (^uint64(0) << bitShift)
	for z == 0 {
		wordIdx++
		byteOff += 8
		if uint64(wordIdx)*64 >= highBitsLen || byteOff >= len(highBits) {
			return highBitsLen
		}
		z = ^readPEFWord64(highBits, byteOff)
	}
	res := uint64(wordIdx)*64 + uint64(bits.TrailingZeros64(z))
	if res > highBitsLen {
		return highBitsLen
	}
	return res
}
