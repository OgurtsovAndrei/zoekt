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
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
	"testing"
)

func readBitLSB(buf []byte, pos uint64) bool {
	return (buf[pos>>3]>>(pos&7))&1 != 0
}

func readLowBitsLSB(buf []byte, idx uint64, ell uint8) uint32 {
	if ell == 0 {
		return 0
	}
	bitPos := idx * uint64(ell)
	byteOff := bitPos >> 3
	bitShift := bitPos & 7
	var word uint64
	for j := 0; j < 8 && int(byteOff)+j < len(buf); j++ {
		word |= uint64(buf[int(byteOff)+j]) << (8 * j)
	}
	mask := (uint64(1) << ell) - 1
	return uint32((word >> bitShift) & mask)
}

func scanZeroBitPositions(highBits []byte, totalBits uint64) []uint32 {
	var zeros []uint32
	for pos := uint64(0); pos < totalBits; pos++ {
		if !readBitLSB(highBits, pos) {
			zeros = append(zeros, uint32(pos))
		}
	}
	return zeros
}

func TestPEFEncodeHeaderAndDirectory(t *testing.T) {
	// Build a multi-chunk list containing:
	// - contiguous run of 1000 keys starting at 100 (AllOnes)
	// - dense cluster of 100 keys with gap 2 (Bitmap)
	// - sparse tail near MaxUint32 (EF)
	keys := make([]uint32, 0, 1200)
	for i := uint32(0); i < 1000; i++ {
		keys = append(keys, 100+i)
	}
	base := uint32(11000)
	for i := uint32(0); i < 100; i++ {
		keys = append(keys, base+2*i)
	}
	sparseStart := uint32(math.MaxUint32 - 500000)
	for i := uint32(0); i < 50; i++ {
		keys = append(keys, sparseStart+i*10000)
	}

	enc := newPEFEncoder(defaultPEFPartitionConfig())
	blob := enc.encodePositions(keys)

	if len(blob) < 12 {
		t.Fatalf("blob too short: %d bytes", len(blob))
	}
	gotN := binary.LittleEndian.Uint32(blob[0:4])
	gotC := binary.LittleEndian.Uint32(blob[4:8])
	gotFirst := binary.LittleEndian.Uint32(blob[8:12])

	if gotN != uint32(len(keys)) {
		t.Fatalf("header n = %d, want %d", gotN, len(keys))
	}
	if gotC < 2 {
		t.Fatalf("expected >= 2 chunks for heterogeneous list, got %d", gotC)
	}
	if gotFirst != keys[0] {
		t.Fatalf("header firstKey = %d, want %d", gotFirst, keys[0])
	}

	c := int(gotC)
	payloadStart := 12 + 12*c
	if len(blob) < payloadStart {
		t.Fatalf("blob length %d < payloadStart %d", len(blob), payloadStart)
	}

	var totalCount uint32
	var prevLast uint32
	var prevOff uint32
	seenKinds := map[pefChunkKind]bool{}

	for i := 0; i < c; i++ {
		last := binary.LittleEndian.Uint32(blob[12+4*i : 16+4*i])
		nKind := binary.LittleEndian.Uint32(blob[12+4*c+4*i : 16+4*c+4*i])
		off := binary.LittleEndian.Uint32(blob[12+8*c+4*i : 16+8*c+4*i])

		count := nKind >> 2
		kind := pefChunkKind(nKind & 3)
		seenKinds[kind] = true

		if count == 0 {
			t.Fatalf("chunk %d has count 0", i)
		}
		if i > 0 && last <= prevLast {
			t.Fatalf("chunkLast not strictly increasing at %d: %d <= %d", i, last, prevLast)
		}
		if i == 0 && off != 0 {
			t.Fatalf("chunkOff[0] = %d, want 0", off)
		}
		if i > 0 && off < prevOff {
			t.Fatalf("chunkOff not monotonic at %d: %d < %d", i, off, prevOff)
		}

		var chunkBase uint64
		if i == 0 {
			chunkBase = uint64(gotFirst)
		} else {
			chunkBase = uint64(prevLast) + 1
		}
		lastRel := uint32(uint64(last) - chunkBase)
		wantKind := pefSelectCodec(lastRel, count)
		if kind != wantKind {
			t.Fatalf("chunk %d kind = %d, want %d (lastRel=%d, count=%d)", i, kind, wantKind, lastRel, count)
		}

		// Verify exact payload byte size against next offset or end of blob.
		var nextOff uint32
		if i+1 < c {
			nextOff = binary.LittleEndian.Uint32(blob[12+8*c+4*(i+1) : 16+8*c+4*(i+1)])
		} else {
			nextOff = uint32(len(blob) - payloadStart)
		}
		gotSize := nextOff - off
		var wantSize uint32
		switch kind {
		case pefKindAllOnes:
			wantSize = 0
		case pefKindBitmap:
			wantSize = uint32((uint64(lastRel) + 8) / 8)
		case pefKindEF:
			var ell uint8
			if lastRel >= count {
				ell = uint8(bits.Len32(lastRel/count) - 1)
			}
			numZeros := uint64(lastRel >> ell)
			var samples uint64
			if numZeros >= pefSelect0SampleRate {
				samples = (numZeros / pefSelect0SampleRate) * 4
			}
			highBytes := (uint64(count) + numZeros + 7) / 8
			lowBytes := (uint64(count)*uint64(ell) + 7) / 8
			wantSize = uint32(samples + highBytes + lowBytes)
		default:
			t.Fatalf("unexpected chunk kind %d", kind)
		}
		if gotSize != wantSize {
			t.Fatalf("chunk %d (kind=%d) payload size = %d, want %d", i, kind, gotSize, wantSize)
		}

		totalCount += count
		prevLast = last
		prevOff = off
	}

	if totalCount != gotN {
		t.Fatalf("sum of chunk counts = %d, want %d", totalCount, gotN)
	}
	if prevLast != keys[len(keys)-1] {
		t.Fatalf("last chunkLast = %d, want %d", prevLast, keys[len(keys)-1])
	}
	for _, k := range []pefChunkKind{pefKindAllOnes, pefKindBitmap, pefKindEF} {
		if !seenKinds[k] {
			t.Fatalf("expected chunk kind %d in heterogeneous test list", k)
		}
	}
}

func TestPEFEncodeAllThreeChunkKinds(t *testing.T) {
	t.Run("AllOnes", func(t *testing.T) {
		keys := make([]uint32, 64)
		for i := range keys {
			keys[i] = 500 + uint32(i)
		}
		enc := newPEFEncoder(defaultPEFPartitionConfig())
		blob := enc.encodePositions(keys)
		// 12-byte header + 12-byte directory (1 chunk) + 0-byte payload
		if len(blob) != 24 {
			t.Fatalf("AllOnes blob len = %d, want 24", len(blob))
		}
		nKind := binary.LittleEndian.Uint32(blob[16:20])
		if pefChunkKind(nKind&3) != pefKindAllOnes {
			t.Fatalf("kind = %d, want AllOnes", nKind&3)
		}
		if nKind>>2 != 64 {
			t.Fatalf("count = %d, want 64", nKind>>2)
		}
	})

	t.Run("Bitmap", func(t *testing.T) {
		// 100 keys at even offsets -> lastRel = 198 -> ceil(199/8) = 25 bytes.
		keys := make([]uint32, 100)
		const base = uint32(1000)
		for i := range keys {
			keys[i] = base + uint32(2*i)
		}
		enc := newPEFEncoder(defaultPEFPartitionConfig())
		blob := enc.encodePositions(keys)
		if gotC := binary.LittleEndian.Uint32(blob[4:8]); gotC != 1 {
			t.Fatalf("numChunks = %d, want 1", gotC)
		}
		nKind := binary.LittleEndian.Uint32(blob[16:20])
		if pefChunkKind(nKind&3) != pefKindBitmap {
			t.Fatalf("kind = %d, want Bitmap", nKind&3)
		}
		payload := blob[24:]
		const lastRel = 198
		wantLen := (lastRel + 8) / 8
		if len(payload) != wantLen {
			t.Fatalf("Bitmap payload len = %d, want %d", len(payload), wantLen)
		}
		for u := 0; u <= lastRel; u++ {
			gotBit := readBitLSB(payload, uint64(u))
			wantBit := u%2 == 0
			if gotBit != wantBit {
				t.Fatalf("bitmap bit %d = %v, want %v", u, gotBit, wantBit)
			}
		}
		// Unused padding bits in the final byte must be zero.
		for u := lastRel + 1; u < wantLen*8; u++ {
			if readBitLSB(payload, uint64(u)) {
				t.Fatalf("padding bit %d is non-zero", u)
			}
		}
	})

	t.Run("EF_ell0", func(t *testing.T) {
		// Direct check of encodeEFChunk when ell == 0 (lastRel < 2*count).
		keys := []uint32{10, 12, 13, 15, 18} // base=10, lastRel=8, count=5 -> ell=0, numZeros=8
		const base = uint64(10)
		payload := appendPEFChunkEF(nil, keys, base, 8)
		// numZeros = 8 < 256 => 0 select0Samples
		// highBitsLen = 5 + 8 = 13 => ceil(13/8) = 2 bytes
		// lowBits = 0 bytes (ell == 0)
		if len(payload) != 2 {
			t.Fatalf("EF ell=0 payload len = %d, want 2", len(payload))
		}
		// Expected 1-bits at i + (key - base):
		// i=0: 0+0=0; i=1: 1+2=3; i=2: 2+3=5; i=3: 3+5=8; i=4: 4+8=12.
		wantOnes := map[uint64]bool{0: true, 3: true, 5: true, 8: true, 12: true}
		for pos := uint64(0); pos < 16; pos++ {
			if readBitLSB(payload, pos) != wantOnes[pos] {
				t.Fatalf("highBits[%d] = %v, want %v", pos, readBitLSB(payload, pos), wantOnes[pos])
			}
		}
	})

	t.Run("EF_ellPositive_and_select0Samples", func(t *testing.T) {
		for _, numZerosTarget := range []uint32{255, 256, 257, 512} {
			t.Run(fmt.Sprintf("numZeros=%d", numZerosTarget), func(t *testing.T) {
				// Choose count = lastRel / 5 so lastRel/count == 5 => ell == 2 and EF beats Bitmap.
				const ell = uint8(2)
				const base = uint64(1000)
				lastRel := (numZerosTarget << ell) | 3
				count := int(lastRel / 5) // lastRel/count == 5 => ell=2 and EF beats Bitmap
				if gotEll := uint8(bits.Len32(lastRel/uint32(count)) - 1); gotEll != ell {
					t.Fatalf("setup error: got ell=%d, want %d (numZeros=%d)", gotEll, ell, numZerosTarget)
				}

				keys := make([]uint32, count)
				// Spread keys roughly evenly, ending exactly at base + lastRel.
				for i := 0; i < count-1; i++ {
					rel := uint32((uint64(i) * uint64(lastRel)) / uint64(count-1))
					if i > 0 && base+uint64(rel) <= uint64(keys[i-1]) {
						rel = uint32(uint64(keys[i-1]) - base + 1)
					}
					keys[i] = uint32(base) + rel
				}
				keys[count-1] = uint32(base) + lastRel

				// Verify via encodePositions (which must pick EF for single chunk when FixCost is high or forced).
				cfg := defaultPEFPartitionConfig()
				cfg.FixCost = 10000 // force single chunk so exact count & lastRel are preserved
				enc := newPEFEncoder(cfg)
				blob := enc.encodePositions(keys)

				if gotC := binary.LittleEndian.Uint32(blob[4:8]); gotC != 1 {
					t.Fatalf("numChunks = %d, want 1", gotC)
				}
				nKind := binary.LittleEndian.Uint32(blob[16:20])
				if pefChunkKind(nKind&3) != pefKindEF {
					t.Fatalf("kind = %d, want EF", nKind&3)
				}

				payload := blob[24:]
				numZeros := uint64(lastRel >> ell)
				if numZeros != uint64(numZerosTarget) {
					t.Fatalf("numZeros = %d, want %d", numZeros, numZerosTarget)
				}
				numSamples := numZeros / pefSelect0SampleRate
				sampleBytes := int(numSamples * 4)
				highBitsLen := uint64(count) + numZeros
				highBytes := int((highBitsLen + 7) / 8)
				lowBytes := int((uint64(count)*uint64(ell) + 7) / 8)

				if len(payload) != sampleBytes+highBytes+lowBytes {
					t.Fatalf("EF payload len = %d, want %d (samples=%d, high=%d, low=%d)",
						len(payload), sampleBytes+highBytes+lowBytes, sampleBytes, highBytes, lowBytes)
				}

				highBits := payload[sampleBytes : sampleBytes+highBytes]
				lowBits := payload[sampleBytes+highBytes:]

				// Reconstruct every key from highBits + lowBits.
				var reconstructed []uint32
				for pos := uint64(0); pos < highBitsLen; pos++ {
					if readBitLSB(highBits, pos) {
						idx := uint64(len(reconstructed))
						h := pos - idx
						lo := readLowBitsLSB(lowBits, idx, ell)
						val := uint32(base + (h << ell) + uint64(lo))
						reconstructed = append(reconstructed, val)
					}
				}
				if len(reconstructed) != len(keys) {
					t.Fatalf("reconstructed len = %d, want %d", len(reconstructed), len(keys))
				}
				for i := range keys {
					if reconstructed[i] != keys[i] {
						t.Fatalf("key[%d] = %d, want %d", i, reconstructed[i], keys[i])
					}
				}

				// Verify select0Samples match scanned zero positions.
				zeros := scanZeroBitPositions(highBits, highBitsLen)
				if uint64(len(zeros)) != numZeros {
					t.Fatalf("scanned zero count = %d, want %d", len(zeros), numZeros)
				}
				for s := uint64(0); s < numSamples; s++ {
					gotSample := binary.LittleEndian.Uint32(payload[s*4 : (s+1)*4])
					wantZeroIdx := (s+1)*pefSelect0SampleRate - 1
					if gotSample != zeros[wantZeroIdx] {
						t.Fatalf("numZeros=%d sample[%d] = %d, want %d",
							numZeros, s, gotSample, zeros[wantZeroIdx])
					}
				}
			})
		}
	})

	t.Run("BufferReuseNoDirtyBits", func(t *testing.T) {
		enc := newPEFEncoder(defaultPEFPartitionConfig())
		// First encode a noisy large list with all 1s in bytes so outBuf is dirty.
		noisy := make([]uint32, 300)
		for i := range noisy {
			noisy[i] = uint32(i) * 3
		}
		_ = enc.encodePositions(noisy)

		// Now encode a sparse Bitmap list and verify zero bits are truly zero.
		sparseBM := []uint32{0, 2, 10, 15}
		blob := append([]byte(nil), enc.encodePositions(sparseBM)...)
		if pefChunkKind(binary.LittleEndian.Uint32(blob[16:20])&3) != pefKindBitmap {
			t.Fatalf("expected Bitmap")
		}
		payload := blob[24:]
		if len(payload) != 2 {
			t.Fatalf("payload len = %d, want 2", len(payload))
		}
		// Bits 0, 2, 10, 15 => byte 0 = 0b0000_0101 (0x05), byte 1 = 0b1000_0100 (0x84).
		if payload[0] != 0x05 || payload[1] != 0x84 {
			t.Fatalf("Bitmap bytes = %#x, want [0x05, 0x84]", payload)
		}
	})
}

func TestPEFEncodeFromDeltasThreshold(t *testing.T) {
	encodeVarints := func(keys []uint32) []byte {
		var out []byte
		var buf [binary.MaxVarintLen64]byte
		var prev uint32
		for _, k := range keys {
			delta := uint64(k - prev)
			m := binary.PutUvarint(buf[:], delta)
			out = append(out, buf[:m]...)
			prev = k
		}
		return out
	}

	keys := make([]uint32, 32)
	cur := uint32(10) // 1-byte first varint
	for i := range keys {
		keys[i] = cur
		if i%3 == 0 {
			cur += 500 // multi-byte varint >= 0x80
		} else {
			cur += 3 // 1-byte varint < 0x80
		}
	}
	deltas := encodeVarints(keys)

	enc := newPEFEncoder(defaultPEFPartitionConfig())

	// Below threshold -> ok == false, blob == nil, exact count.
	blob, count, ok := enc.encodeFromDeltas(deltas, 33)
	if ok || blob != nil || count != 32 {
		t.Fatalf("below threshold: got (blob=%v, count=%d, ok=%v), want (nil, 32, false)", blob != nil, count, ok)
	}

	// Empty input -> ok == false, count == 0.
	blob, count, ok = enc.encodeFromDeltas(nil, 0)
	if ok || blob != nil || count != 0 {
		t.Fatalf("empty input: got (blob=%v, count=%d, ok=%v), want (nil, 0, false)", blob != nil, count, ok)
	}

	// At threshold -> ok == true, exact count, byte-identical to encodePositions.
	blob, count, ok = enc.encodeFromDeltas(deltas, 32)
	if !ok || count != 32 {
		t.Fatalf("at threshold: got (count=%d, ok=%v), want (32, true)", count, ok)
	}
	gotBlob := append([]byte(nil), blob...)
	wantBlob := enc.encodePositions(keys)
	if !bytes.Equal(gotBlob, wantBlob) {
		t.Fatalf("encodeFromDeltas output differs from encodePositions (len %d vs %d)", len(gotBlob), len(wantBlob))
	}

	// Malformed varint -> panics.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("expected panic on malformed varint")
			}
		}()
		_, _, _ = enc.encodeFromDeltas([]byte{0x80}, 1)
	}()
}
