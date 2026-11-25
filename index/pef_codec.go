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
	"math/bits"
)

const pefSelect0SampleRate = 256

const pefHeaderSize = 12
const pefUniformHeaderSize = 8

func validPEFUniformBlockShift(shift uint8) bool {
	return shift == 7 || shift == 8
}

func pefUniformChunkCount(n uint32, shift uint8) uint32 {
	return 1 + ((n - 1) >> shift)
}

type pefUniformEncoder struct {
	posBuf     []uint32
	outBuf     []byte
	blockShift uint8
}

func newPEFUniformEncoder(blockShift uint8) *pefUniformEncoder {
	if !validPEFUniformBlockShift(blockShift) {
		panic("pefUniformEncoder: invalid block size")
	}
	return &pefUniformEncoder{blockShift: blockShift}
}

func (e *pefUniformEncoder) encodePositions(keys []uint32) []byte {
	if len(keys) == 0 {
		panic("pefUniformEncoder: empty keys")
	}
	blockSize := 1 << e.blockShift
	numChunks := int(pefUniformChunkCount(uint32(len(keys)), e.blockShift))
	dirLast := pefUniformHeaderSize
	dirOff := dirLast + 4*numChunks
	dirKind := dirOff + 4*numChunks
	payloadStart := dirKind + (numChunks+3)/4
	if cap(e.outBuf) < payloadStart {
		e.outBuf = make([]byte, payloadStart)
	} else {
		e.outBuf = e.outBuf[:payloadStart]
		clear(e.outBuf[dirKind:payloadStart])
	}
	binary.LittleEndian.PutUint32(e.outBuf[0:4], uint32(len(keys)))
	binary.LittleEndian.PutUint32(e.outBuf[4:8], keys[0])

	base := uint64(keys[0])
	for c, start := 0, 0; start < len(keys); c, start = c+1, start+blockSize {
		chunkKeys := keys[start:min(start+blockSize, len(keys))]
		last := chunkKeys[len(chunkKeys)-1]
		lastRel := uint32(uint64(last) - base)
		kind := pefSelectCodec(lastRel, uint32(len(chunkKeys)))
		binary.LittleEndian.PutUint32(e.outBuf[dirLast+4*c:], last)
		binary.LittleEndian.PutUint32(e.outBuf[dirOff+4*c:], uint32(len(e.outBuf)-payloadStart))
		e.outBuf[dirKind+c/4] |= byte(kind) << (2 * uint(c%4))
		e.outBuf = appendPEFChunk(e.outBuf, chunkKeys, base, lastRel, kind)
		base = uint64(last) + 1
	}
	return e.outBuf
}

func (e *pefUniformEncoder) encodeFromDeltas(deltaVarints []byte, minPostings int) (blob []byte, count int, ok bool) {
	e.posBuf = decodePEFPostingDeltas(e.posBuf, deltaVarints)
	count = len(e.posBuf)
	if count == 0 || count < minPostings {
		return nil, count, false
	}
	return e.encodePositions(e.posBuf), count, true
}

// pefEncoder encodes strictly increasing uint32 posting positions into a
// self-contained, little-endian PISA PEF flat-blob. Instances are not safe for
// concurrent use and reuse internal scratch buffers across calls.
type pefEncoder struct {
	posBuf  []uint32
	partOut []uint32
	scratch pefPartitionScratch
	outBuf  []byte
	cfg     pefPartitionConfig
}

func newPEFEncoder(cfg pefPartitionConfig) *pefEncoder {
	if err := cfg.validate(); err != nil {
		panic(err)
	}
	return &pefEncoder{cfg: cfg}
}

// encodePositions encodes strictly increasing keys into e.outBuf following the
// PISA PEF flat-blob layout and returns a slice valid until the next call.
func (e *pefEncoder) encodePositions(keys []uint32) []byte {
	if len(keys) == 0 {
		panic("pefEncoder: empty keys")
	}

	e.partOut = e.scratch.partition(keys, e.cfg, e.partOut)
	numChunks := len(e.partOut)
	payloadStart := pefHeaderSize + 12*numChunks

	if cap(e.outBuf) < payloadStart {
		e.outBuf = make([]byte, payloadStart)
	} else {
		e.outBuf = e.outBuf[:payloadStart]
	}

	binary.LittleEndian.PutUint32(e.outBuf[0:4], uint32(len(keys)))
	binary.LittleEndian.PutUint32(e.outBuf[4:8], uint32(numChunks))
	binary.LittleEndian.PutUint32(e.outBuf[8:12], keys[0])

	dirLast := pefHeaderSize
	dirNKind := pefHeaderSize + 4*numChunks
	dirOff := pefHeaderSize + 8*numChunks

	start := 0
	base := uint64(keys[0])
	for c := 0; c < numChunks; c++ {
		end := int(e.partOut[c])
		chunkKeys := keys[start:end]
		count := uint32(end - start)
		last := chunkKeys[count-1]
		lastRel := uint32(uint64(last) - base)
		kind := pefSelectCodec(lastRel, count)

		off := uint32(len(e.outBuf) - payloadStart)
		binary.LittleEndian.PutUint32(e.outBuf[dirLast+4*c:], last)
		binary.LittleEndian.PutUint32(e.outBuf[dirNKind+4*c:], (count<<2)|uint32(kind))
		binary.LittleEndian.PutUint32(e.outBuf[dirOff+4*c:], off)

		e.outBuf = appendPEFChunk(e.outBuf, chunkKeys, base, lastRel, kind)

		start = end
		base = uint64(last) + 1
	}

	return e.outBuf
}

// encodeFromDeltas decodes Zoekt delta-varint postings into e.posBuf. If fewer
// than minPostings (or zero) postings are present, it returns (nil, count, false).
// Otherwise it encodes them with encodePositions and returns (blob, count, true),
// where blob is valid until the next encoder call.
func (e *pefEncoder) encodeFromDeltas(deltaVarints []byte, minPostings int) (blob []byte, count int, ok bool) {
	e.posBuf = decodePEFPostingDeltas(e.posBuf, deltaVarints)
	count = len(e.posBuf)
	if count == 0 || count < minPostings {
		return nil, count, false
	}
	return e.encodePositions(e.posBuf), count, true
}

func decodePEFPostingDeltas(dst []uint32, deltaVarints []byte) []uint32 {
	dst = dst[:0]
	var cur uint32
	i := 0
	n := len(deltaVarints)
	for i < n {
		b := deltaVarints[i]
		if b < 0x80 {
			cur += uint32(b)
			dst = append(dst, cur)
			i++
			continue
		}
		d, sz := binary.Uvarint(deltaVarints[i:])
		if sz <= 0 {
			panic(fmt.Sprintf("pefEncoder: malformed varint at offset %d (sz=%d)", i, sz))
		}
		cur += uint32(d)
		dst = append(dst, cur)
		i += sz
	}
	return dst
}

func appendPEFChunk(out []byte, keys []uint32, base uint64, lastRel uint32, kind pefChunkKind) []byte {
	switch kind {
	case pefKindAllOnes:
		return out
	case pefKindBitmap:
		return appendPEFChunkBitmap(out, keys, base, lastRel)
	case pefKindEF:
		return appendPEFChunkEF(out, keys, base, lastRel)
	default:
		panic(fmt.Sprintf("pefEncoder: unexpected chunk kind %d", kind))
	}
}

func growZeroed(buf []byte, n int) ([]byte, []byte) {
	if n == 0 {
		return buf, nil
	}
	oldLen := len(buf)
	newLen := oldLen + n
	if cap(buf) < newLen {
		nextCap := newLen
		if doubled := cap(buf) * 2; doubled > nextCap {
			nextCap = doubled
		}
		next := make([]byte, newLen, nextCap)
		copy(next, buf)
		buf = next
	} else {
		buf = buf[:newLen]
	}
	tail := buf[oldLen:newLen]
	clear(tail)
	return buf, tail
}

func appendPEFChunkBitmap(out []byte, keys []uint32, base uint64, lastRel uint32) []byte {
	byteLen := int((uint64(lastRel) >> 3) + 1)
	out, bm := growZeroed(out, byteLen)
	for _, k := range keys {
		rel := uint32(uint64(k) - base)
		bm[rel>>3] |= 1 << (rel & 7)
	}
	return out
}

func appendPEFChunkEF(out []byte, keys []uint32, base uint64, lastRel uint32) []byte {
	count := uint32(len(keys))
	var ell uint8
	if lastRel >= count {
		ell = uint8(bits.Len32(lastRel/count) - 1)
	}

	numZeros := uint64(lastRel >> ell)
	numSamples := int(numZeros / pefSelect0SampleRate)
	sampleBytes := numSamples * 4
	highBitsLen := uint64(count) + numZeros
	highBytes := int((highBitsLen + 7) / 8)
	lowBytes := int((uint64(count)*uint64(ell) + 7) / 8)

	out, payload := growZeroed(out, sampleBytes+highBytes+lowBytes)
	samplesBuf := payload[:sampleBytes]
	highBuf := payload[sampleBytes : sampleBytes+highBytes]
	lowBuf := payload[sampleBytes+highBytes:]

	nextSampleZero := uint64(pefSelect0SampleRate - 1)
	sampleIdx := 0
	for i, k := range keys {
		rel := uint32(uint64(k) - base)
		h := uint64(rel >> ell)
		for nextSampleZero < h {
			binary.LittleEndian.PutUint32(samplesBuf[sampleIdx*4:], uint32(uint64(i)+nextSampleZero))
			sampleIdx++
			nextSampleZero += pefSelect0SampleRate
		}
		pos := uint64(i) + h
		highBuf[pos>>3] |= 1 << (pos & 7)
	}

	if ell > 0 {
		writeLowBits(lowBuf, keys, base, ell)
	}
	return out
}

func writeLowBits(dst []byte, keys []uint32, base uint64, ell uint8) {
	lowMask := (uint32(1) << ell) - 1
	var acc uint64
	var accBits uint
	outIdx := 0
	for _, k := range keys {
		low := uint64(uint32(uint64(k)-base) & lowMask)
		acc |= low << accBits
		accBits += uint(ell)
		for accBits >= 8 {
			dst[outIdx] = byte(acc)
			outIdx++
			acc >>= 8
			accBits -= 8
		}
	}
	if accBits > 0 {
		dst[outIdx] = byte(acc)
	}
}
