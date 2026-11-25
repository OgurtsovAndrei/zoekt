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
	"math/rand"
	"reflect"
	"slices"
	"strings"
	"testing"
	"testing/quick"

	"github.com/google/go-cmp/cmp"

	"github.com/sourcegraph/zoekt"
)

func encodeTestPEF(keys []uint32, cfg pefPartitionConfig) []byte {
	enc := newPEFEncoder(cfg)
	return append([]byte(nil), enc.encodePositions(keys)...)
}

func drainPEFIterator(blob []byte) []uint32 {
	it := newPEFPostingIterator(blob, stringToNGram("abc"))
	var out []uint32
	for it.first() != math.MaxUint32 {
		cur := it.first()
		out = append(out, cur)
		it.next(cur)
	}
	return out
}

func assertPEFMatchesInMemory(t *testing.T, keys, limits []uint32, cfg pefPartitionConfig) {
	t.Helper()
	blob := encodeTestPEF(keys, cfg)

	wantSeq := slices.Clone(keys)
	gotSeq := drainPEFIterator(blob)
	if !reflect.DeepEqual(gotSeq, wantSeq) {
		t.Fatalf("sequential mismatch:\n%s", cmp.Diff(wantSeq, gotSeq))
	}

	memKeys := slices.Clone(keys)
	want := doHitIterator(&inMemoryIterator{postings: memKeys}, limits)
	got := doHitIterator(newPEFPostingIterator(blob, stringToNGram("abc")), limits)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("skip mismatch:\n%s", cmp.Diff(want, got))
	}
}

func makeSelect0SampleKeys(numZerosTarget uint32) ([]uint32, pefPartitionConfig) {
	// Choose ell = 2 and count = lastRel / 5 so lastRel/count == 5 => ell == 2
	// and EF beats Bitmap cleanly.
	const ell = uint8(2)
	const base = uint64(1000)
	lastRel := (numZerosTarget << ell) | 3
	count := int(lastRel / 5)
	keys := make([]uint32, count)
	for i := 0; i < count-1; i++ {
		rel := uint32((uint64(i) * uint64(lastRel)) / uint64(count-1))
		if i > 0 && base+uint64(rel) <= uint64(keys[i-1]) {
			rel = uint32(uint64(keys[i-1]) - base + 1)
		}
		keys[i] = uint32(base) + rel
	}
	keys[count-1] = uint32(base) + lastRel

	cfg := defaultPEFPartitionConfig()
	cfg.FixCost = 10000 // force single EF chunk preserving exact numZerosTarget
	return keys, cfg
}

func TestPEFPostingIteratorSequentialAndSkip(t *testing.T) {
	ng := stringToNGram("abc")

	t.Run("SingleKey", func(t *testing.T) {
		for _, k := range []uint32{0, 42} {
			blob := encodeTestPEF([]uint32{k}, defaultPEFPartitionConfig())
			it := newPEFPostingIterator(blob, ng)
			if got := it.first(); got != k {
				t.Fatalf("first() = %d, want %d", got, k)
			}
			if k > 0 {
				it.next(k - 1)
				if got := it.first(); got != k {
					t.Fatalf("after next(%d), first() = %d, want %d", k-1, got, k)
				}
			}
			it.next(k)
			if got := it.first(); got != math.MaxUint32 {
				t.Fatalf("after next(%d), first() = %d, want MaxUint32", k, got)
			}
			it.next(math.MaxUint32)
			if got := it.first(); got != math.MaxUint32 {
				t.Fatalf("after next(MaxUint32), first() = %d, want MaxUint32", got)
			}
		}
	})

	t.Run("AllOnes", func(t *testing.T) {
		keys := make([]uint32, 128)
		for i := range keys {
			keys[i] = 500 + uint32(i)
		}
		blob := encodeTestPEF(keys, defaultPEFPartitionConfig())
		if kind := pefChunkKind(binary.LittleEndian.Uint32(blob[16:20]) & 3); kind != pefKindAllOnes {
			t.Fatalf("kind = %d, want AllOnes", kind)
		}
		limits := []uint32{0, 499, 500, 505, 560, 626, 627, 1000}
		assertPEFMatchesInMemory(t, keys, limits, defaultPEFPartitionConfig())
	})

	t.Run("Bitmap", func(t *testing.T) {
		keys := make([]uint32, 100)
		for i := range keys {
			keys[i] = 1000 + uint32(2*i)
		}
		blob := encodeTestPEF(keys, defaultPEFPartitionConfig())
		if kind := pefChunkKind(binary.LittleEndian.Uint32(blob[16:20]) & 3); kind != pefKindBitmap {
			t.Fatalf("kind = %d, want Bitmap", kind)
		}
		limits := []uint32{999, 1000, 1001, 1063, 1064, 1127, 1128, 1197, 1198, 1199}
		assertPEFMatchesInMemory(t, keys, limits, defaultPEFPartitionConfig())
	})

	t.Run("EF_ell0", func(t *testing.T) {
		// Force a single EF chunk with ell == 0 (lastRel < 2*count).
		// 100 keys with gaps of 1 and 2 => lastRel = 149, count = 100 => ell = 0, numZeros = 149.
		keys := make([]uint32, 100)
		cur := uint32(200)
		for i := range keys {
			keys[i] = cur
			if i%2 == 0 {
				cur += 1
			} else {
				cur += 2
			}
		}
		// Build single-chunk EF directly to guarantee kindEF with ell == 0.
		lastRel := keys[len(keys)-1] - keys[0]
		if gotEll := uint8(bits.Len32(lastRel/uint32(len(keys))) - 1); gotEll != 0 {
			t.Fatalf("expected ell=0, got %d", gotEll)
		}
		payload := appendPEFChunkEF(nil, keys, uint64(keys[0]), lastRel)
		blob := make([]byte, 24+len(payload))
		binary.LittleEndian.PutUint32(blob[0:4], uint32(len(keys)))
		binary.LittleEndian.PutUint32(blob[4:8], 1)
		binary.LittleEndian.PutUint32(blob[8:12], keys[0])
		binary.LittleEndian.PutUint32(blob[12:16], keys[len(keys)-1])
		binary.LittleEndian.PutUint32(blob[16:20], (uint32(len(keys))<<2)|uint32(pefKindEF))
		binary.LittleEndian.PutUint32(blob[20:24], 0)
		copy(blob[24:], payload)

		if got := drainPEFIterator(blob); !reflect.DeepEqual(got, keys) {
			t.Fatalf("EF ell=0 sequential mismatch:\n%s", cmp.Diff(keys, got))
		}
		limits := []uint32{199, 200, 201, 220, 221, 290, keys[len(keys)-1] - 1, keys[len(keys)-1], keys[len(keys)-1] + 5}
		want := doHitIterator(&inMemoryIterator{postings: slices.Clone(keys)}, limits)
		got := doHitIterator(newPEFPostingIterator(blob, ng), limits)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("EF ell=0 skip mismatch:\n%s", cmp.Diff(want, got))
		}
	})

	t.Run("EF_ellPositive_select0Samples", func(t *testing.T) {
		for _, nz := range []uint32{255, 256, 257, 512, 700} {
			t.Run(fmt.Sprintf("numZeros=%d", nz), func(t *testing.T) {
				keys, cfg := makeSelect0SampleKeys(nz)
				blob := encodeTestPEF(keys, cfg)
				if kind := pefChunkKind(binary.LittleEndian.Uint32(blob[16:20]) & 3); kind != pefKindEF {
					t.Fatalf("kind = %d, want EF", kind)
				}

				// Sequential scan + stepped jumps around 256/512 zero boundaries.
				limits := []uint32{
					keys[0] - 1,
					keys[0],
					keys[0] + (254 << 2),
					keys[0] + (255 << 2),
					keys[0] + (256 << 2),
					keys[0] + (257 << 2),
					keys[len(keys)-1] - 1,
					keys[len(keys)-1],
					keys[len(keys)-1] + 100,
				}
				// Ensure sorted unique monotonic limits.
				slices.Sort(limits)
				assertPEFMatchesInMemory(t, keys, limits, cfg)

				// Single large jump spanning > 512 zeros directly from first().
				if nz >= 512 {
					it := newPEFPostingIterator(blob, ng)
					jumpTarget := keys[len(keys)-2] - 1
					it.next(jumpTarget)
					want := keys[len(keys)-2]
					if it.first() < jumpTarget+1 || it.first() > keys[len(keys)-1] {
						t.Fatalf("large jump got %d, want in [%d, %d]", it.first(), jumpTarget+1, keys[len(keys)-1])
					}
					_ = want
					// Compare exact single jump against inMemoryIterator.
					mem := &inMemoryIterator{postings: slices.Clone(keys)}
					mem.next(jumpTarget)
					if it.first() != mem.first() {
						t.Fatalf("single large jump got %d, want %d", it.first(), mem.first())
					}
				}
			})
		}
	})

	t.Run("EF_LargeBucketBinarySearch", func(t *testing.T) {
		// Create a chunk where 150 keys fall into bucket h=1 (ell=8, lows 0..299 step 2)
		// plus an outlier at the end so EF is chosen and bucket h=1 has > 64 elements.
		keys := make([]uint32, 0, 152)
		keys = append(keys, 100) // h=0, low=0
		for i := uint32(0); i < 150; i++ {
			keys = append(keys, 100+256+i) // 150 keys in bucket h=1 (ell=8 -> 256..511), wait: 256+150=406 < 512!
		}
		keys = append(keys, 100+50000) // forces sparse EF with ell >= 8
		cfg := defaultPEFPartitionConfig()
		cfg.FixCost = 100000
		blob := encodeTestPEF(keys, cfg)
		if kind := pefChunkKind(binary.LittleEndian.Uint32(blob[16:20]) & 3); kind != pefKindEF {
			t.Fatalf("kind = %d, want EF", kind)
		}

		// Jump from start directly into middle of large bucket, then jump again inside same bucket,
		// then jump past bucket end into outlier.
		limits := []uint32{
			99,
			100,       // sequential step into bucket h=1
			100 + 258, // same-bucket jump (>1)
			100 + 380, // same-bucket jump (>64 items left)
			100 + 405, // last item in bucket h=1 (256+149 = 405)
			100 + 406, // miss inside bucket h=1 -> advances to outlier 50100
			100 + 50000,
		}
		assertPEFMatchesInMemory(t, keys, limits, cfg)

		// Also verify direct jump from chunk entry into a >64-element bucket (general jump path).
		assertPEFMatchesInMemory(t, keys, []uint32{100 + 380, 100 + 406}, cfg)
	})

	t.Run("MultiChunkMixed_BoundariesAndGaps", func(t *testing.T) {
		var keys []uint32
		// Chunk 0 region: contiguous AllOnes [100..599]
		for i := uint32(0); i < 500; i++ {
			keys = append(keys, 100+i)
		}
		// Inter-chunk gap (600..9999), then dense Bitmap [10000..10398 step 2]
		for i := uint32(0); i < 200; i++ {
			keys = append(keys, 10000+2*i)
		}
		// Inter-chunk gap, then EF (ell=0) cluster [20000..20299]
		cur := uint32(20000)
		for i := 0; i < 150; i++ {
			keys = append(keys, cur)
			if i%3 == 0 {
				cur += 1
			} else {
				cur += 2
			}
		}
		// Inter-chunk gap, then sparse EF with > 512 zeros
		cur += 50000
		for i := 0; i < 600; i++ {
			keys = append(keys, cur)
			cur += 1000 + uint32(i%7)*100
		}

		cfg := defaultPEFPartitionConfig()
		blob := encodeTestPEF(keys, cfg)
		numChunks := int(binary.LittleEndian.Uint32(blob[4:8]))
		if numChunks < 4 {
			t.Fatalf("expected >= 4 chunks, got %d", numChunks)
		}

		// Collect chunkLast values to construct exact boundary & inter-chunk gap limits.
		limits := []uint32{0, 99, 100}
		for c := 0; c < numChunks; c++ {
			cl := binary.LittleEndian.Uint32(blob[12+4*c : 16+4*c])
			if cl > 1 {
				limits = append(limits, cl-1)
			}
			limits = append(limits, cl, cl+1)
		}
		limits = append(limits, keys[len(keys)-1]+1000, math.MaxUint32-1, math.MaxUint32)
		slices.Sort(limits)
		assertPEFMatchesInMemory(t, keys, limits, cfg)
	})

	t.Run("NearMaxUint32", func(t *testing.T) {
		keys := []uint32{
			100,
			101,
			102,
			math.MaxUint32 - 100000,
			math.MaxUint32 - 10,
			math.MaxUint32 - 9,
			math.MaxUint32 - 2,
			math.MaxUint32 - 1,
		}
		limits := []uint32{
			99,
			100,
			102,
			math.MaxUint32 - 100001,
			math.MaxUint32 - 10,
			math.MaxUint32 - 3,
			math.MaxUint32 - 2,
			math.MaxUint32 - 1,
			math.MaxUint32,
		}
		assertPEFMatchesInMemory(t, keys, limits, defaultPEFPartitionConfig())

		// Also verify direct next(MaxUint32 - 1) and next(MaxUint32) on single chunk ending at MaxUint32-1.
		blob := encodeTestPEF(keys, defaultPEFPartitionConfig())
		it := newPEFPostingIterator(blob, ng)
		it.next(math.MaxUint32 - 2)
		if got := it.first(); got != math.MaxUint32-1 {
			t.Fatalf("after next(MaxUint32-2), got %d, want %d", got, uint32(math.MaxUint32-1))
		}
		it.next(math.MaxUint32 - 1)
		if got := it.first(); got != math.MaxUint32 {
			t.Fatalf("after next(MaxUint32-1), got %d, want MaxUint32", got)
		}

		it2 := newPEFPostingIterator(blob, ng)
		it2.next(math.MaxUint32)
		if got := it2.first(); got != math.MaxUint32 {
			t.Fatalf("after next(MaxUint32), got %d, want MaxUint32", got)
		}
	})

	t.Run("StatsAndSkippingBytesLoaded", func(t *testing.T) {
		var keys []uint32
		for chunk := uint32(0); chunk < 20; chunk++ {
			base := chunk*100000 + 100
			for i := uint32(0); i < 100; i++ {
				keys = append(keys, base+i*37)
			}
		}
		blob := encodeTestPEF(keys, defaultPEFPartitionConfig())

		// Full sequential scan stats.
		seqIt := newPEFPostingIterator(blob, ng)
		seqIt.setNgramLookups(7)
		for seqIt.first() != math.MaxUint32 {
			seqIt.next(seqIt.first())
		}
		var seqStats zoekt.Stats
		seqIt.updateStats(&seqStats)
		if seqStats.NgramLookups != 7 {
			t.Fatalf("NgramLookups = %d, want 7", seqStats.NgramLookups)
		}
		// Calling updateStats again should add 0 (reset).
		seqIt.updateStats(&seqStats)
		if seqStats.NgramLookups != 7 {
			t.Fatalf("NgramLookups after second updateStats = %d, want 7", seqStats.NgramLookups)
		}

		// Skip straight to the last chunk.
		skipIt := newPEFPostingIterator(blob, ng)
		skipIt.next(keys[len(keys)-5])
		var skipStats zoekt.Stats
		skipIt.updateStats(&skipStats)

		if skipStats.IndexBytesLoaded <= 0 {
			t.Fatalf("skipStats.IndexBytesLoaded = %d, want > 0", skipStats.IndexBytesLoaded)
		}
		if skipStats.IndexBytesLoaded >= seqStats.IndexBytesLoaded/2 {
			t.Fatalf("skipping IndexBytesLoaded (%d) should be < half of sequential (%d) (blob len=%d)",
				skipStats.IndexBytesLoaded, seqStats.IndexBytesLoaded, len(blob))
		}
	})
}

func TestPEFPostingIteratorQuick(t *testing.T) {
	cfg := defaultPEFPartitionConfig()
	ng := stringToNGram("abc")

	t.Run("RandomUniform", func(t *testing.T) {
		f := func(nums, limits []uint32) bool {
			if len(nums) == 0 || len(limits) == 0 {
				return true
			}
			nums = sortedUnique(nums)
			slices.Sort(limits)

			blob := encodeTestPEF(nums, cfg)
			want := doHitIterator(&inMemoryIterator{postings: slices.Clone(nums)}, limits)
			got := doHitIterator(newPEFPostingIterator(blob, ng), limits)
			if !reflect.DeepEqual(want, got) {
				t.Log(cmp.Diff(want, got))
				return false
			}
			return true
		}
		if err := quick.Check(f, nil); err != nil {
			t.Error(err)
		}
	})

	t.Run("BimodalClustered", func(t *testing.T) {
		f := func(seed int64, limitSeed int64) bool {
			r := rand.New(rand.NewSource(seed))
			lr := rand.New(rand.NewSource(limitSeed))

			// Generate bimodal sequence: contiguous runs + dense clusters + sparse gaps.
			nClusters := 2 + r.Intn(10)
			var nums []uint32
			cur := uint32(r.Intn(1000))
			for c := 0; c < nClusters; c++ {
				mode := r.Intn(3)
				sz := 10 + r.Intn(150)
				for i := 0; i < sz; i++ {
					if cur > math.MaxUint32-1000000 {
						break
					}
					nums = append(nums, cur)
					switch mode {
					case 0:
						cur++
					case 1:
						cur += uint32(1 + r.Intn(4))
					case 2:
						cur += uint32(10 + r.Intn(5000))
					}
				}
				if cur > math.MaxUint32-1000000 {
					break
				}
				cur += uint32(1000 + r.Intn(50000))
			}
			if len(nums) == 0 {
				return true
			}

			nLimits := 1 + lr.Intn(len(nums)*2)
			limits := make([]uint32, nLimits)
			maxKey := nums[len(nums)-1]
			for i := range limits {
				if lr.Intn(10) == 0 && maxKey > 0 {
					limits[i] = maxKey + uint32(lr.Intn(1000))
				} else if maxKey > 0 {
					limits[i] = uint32(lr.Int63n(int64(maxKey) + 1))
				}
			}
			slices.Sort(limits)

			blob := encodeTestPEF(nums, cfg)
			want := doHitIterator(&inMemoryIterator{postings: slices.Clone(nums)}, limits)
			got := doHitIterator(newPEFPostingIterator(blob, ng), limits)
			if !reflect.DeepEqual(want, got) {
				t.Log(cmp.Diff(want, got))
				return false
			}
			return true
		}
		if err := quick.Check(f, nil); err != nil {
			t.Error(err)
		}
	})
}

func TestPEFPostingIteratorAllocs(t *testing.T) {
	var keys []uint32
	for i := uint32(0); i < 500; i++ {
		keys = append(keys, 100+i)
	}
	for i := uint32(0); i < 200; i++ {
		keys = append(keys, 5000+2*i)
	}
	cur := uint32(20000)
	for i := 0; i < 600; i++ {
		keys = append(keys, cur)
		cur += 500 + uint32(i%13)*70
	}

	blob := encodeTestPEF(keys, defaultPEFPartitionConfig())
	ng := stringToNGram("abc")

	seqAllocs := testing.AllocsPerRun(20, func() {
		it := newPEFPostingIterator(blob, ng)
		for it.first() != math.MaxUint32 {
			it.next(it.first())
		}
	})
	// Expect at most 1 allocation for the iterator struct itself; 0 in first()/next().
	if seqAllocs > 1 {
		t.Fatalf("sequential scan + newPEFPostingIterator allocs = %v, want <= 1", seqAllocs)
	}

	it := newPEFPostingIterator(blob, ng)
	loopAllocs := testing.AllocsPerRun(50, func() {
		// Reset iterator in place via value assignment isn't needed if we create outside;
		// test zero allocs during first/next by skipping through pre-built iterators?
		// Better: measure first()/next() across a fresh slice of iterators or re-init fields.
		it.curVal = it.firstKey
		it.chunkIdx = 0
		it.chunkLast = binary.LittleEndian.Uint32(it.blob[12:16])
		it.chunkLoaded = false
		it.efSeeked = false
		for it.first() != math.MaxUint32 {
			it.next(it.first())
		}
	})
	if loopAllocs != 0 {
		t.Fatalf("first()/next() sequential loop allocs = %v, want 0", loopAllocs)
	}

	skipLimits := make([]uint32, 0, 64)
	for i := 0; i < len(keys); i += 23 {
		skipLimits = append(skipLimits, keys[i]-1)
	}
	skipLimits = append(skipLimits, math.MaxUint32-1, math.MaxUint32)
	skipAllocs := testing.AllocsPerRun(50, func() {
		it.curVal = it.firstKey
		it.chunkIdx = 0
		it.chunkLast = binary.LittleEndian.Uint32(it.blob[12:16])
		it.chunkLoaded = false
		it.efSeeked = false
		for _, lim := range skipLimits {
			it.next(lim)
			_ = it.first()
		}
	})
	if skipAllocs != 0 {
		t.Fatalf("first()/next(limit) skip loop allocs = %v, want 0", skipAllocs)
	}
}

func TestPEFPostingIteratorMalformed(t *testing.T) {
	ng := stringToNGram("abc")

	assertCorruptPanic := func(t *testing.T, f func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("got no panic, want corrupt pef posting panic")
			}
			msg := fmt.Sprint(r)
			if !strings.HasPrefix(msg, "corrupt pef posting:") {
				t.Fatalf("unexpected panic %q, want prefix %q", msg, "corrupt pef posting:")
			}
		}()
		f()
	}

	iterateAll := func(blob []byte) {
		it := newPEFPostingIterator(blob, ng)
		for it.first() != math.MaxUint32 {
			it.next(it.first())
		}
	}

	// Build a valid 2-chunk blob (chunk 0: Bitmap, chunk 1: EF with select0Samples) for mutation cases.
	var multiKeys []uint32
	for i := uint32(0); i < 50; i++ {
		multiKeys = append(multiKeys, 100+2*i)
	}
	cur := uint32(10000)
	for i := 0; i < 400; i++ {
		multiKeys = append(multiKeys, cur)
		cur += 700
	}
	validMulti := encodeTestPEF(multiKeys, defaultPEFPartitionConfig())
	numChunks := int(binary.LittleEndian.Uint32(validMulti[4:8]))
	if numChunks < 2 {
		t.Fatalf("setup failed: expected >= 2 chunks, got %d", numChunks)
	}

	t.Run("TruncatedHeader", func(t *testing.T) {
		for _, n := range []int{0, 1, 11} {
			bad := make([]byte, n)
			assertCorruptPanic(t, func() { newPEFPostingIterator(bad, ng) })
		}
	})

	t.Run("ZeroPostings", func(t *testing.T) {
		bad := slices.Clone(validMulti)
		binary.LittleEndian.PutUint32(bad[0:4], 0)
		assertCorruptPanic(t, func() { newPEFPostingIterator(bad, ng) })
	})

	t.Run("ZeroChunks", func(t *testing.T) {
		bad := slices.Clone(validMulti)
		binary.LittleEndian.PutUint32(bad[4:8], 0)
		assertCorruptPanic(t, func() { newPEFPostingIterator(bad, ng) })
	})

	t.Run("NumChunksExceedsN", func(t *testing.T) {
		bad := slices.Clone(validMulti)
		binary.LittleEndian.PutUint32(bad[0:4], 1)
		assertCorruptPanic(t, func() { newPEFPostingIterator(bad, ng) })
	})

	t.Run("TruncatedDirectory", func(t *testing.T) {
		bad := slices.Clone(validMulti[:12+12*numChunks-1])
		assertCorruptPanic(t, func() { newPEFPostingIterator(bad, ng) })
	})

	t.Run("InvalidChunk0Kind", func(t *testing.T) {
		bad := slices.Clone(validMulti)
		nk := binary.LittleEndian.Uint32(bad[12+4*numChunks : 16+4*numChunks])
		binary.LittleEndian.PutUint32(bad[12+4*numChunks:], (nk&^3)|3)
		assertCorruptPanic(t, func() { newPEFPostingIterator(bad, ng) })
	})

	t.Run("ZeroChunk0Count", func(t *testing.T) {
		bad := slices.Clone(validMulti)
		nk := binary.LittleEndian.Uint32(bad[12+4*numChunks : 16+4*numChunks])
		binary.LittleEndian.PutUint32(bad[12+4*numChunks:], nk&3)
		assertCorruptPanic(t, func() { newPEFPostingIterator(bad, ng) })
	})

	t.Run("OOBChunkOff", func(t *testing.T) {
		bad0 := slices.Clone(validMulti)
		binary.LittleEndian.PutUint32(bad0[12+8*numChunks:], 999999)
		assertCorruptPanic(t, func() { iterateAll(bad0) })

		bad1 := slices.Clone(validMulti)
		binary.LittleEndian.PutUint32(bad1[12+8*numChunks+4:], 999999)
		assertCorruptPanic(t, func() { iterateAll(bad1) })
	})

	t.Run("UnsortedChunkLast", func(t *testing.T) {
		bad := slices.Clone(validMulti)
		last0 := binary.LittleEndian.Uint32(bad[12:16])
		binary.LittleEndian.PutUint32(bad[16:20], last0) // chunkLast[1] == chunkLast[0]
		assertCorruptPanic(t, func() { iterateAll(bad) })

		// Also test galloping into unsorted chunkLast.
		bad2 := slices.Clone(validMulti)
		binary.LittleEndian.PutUint32(bad2[16:20], last0-1)
		assertCorruptPanic(t, func() {
			it := newPEFPostingIterator(bad2, ng)
			it.next(last0 + 10)
		})
	})

	t.Run("TruncatedPayload", func(t *testing.T) {
		bad := slices.Clone(validMulti[:len(validMulti)-1])
		assertCorruptPanic(t, func() { iterateAll(bad) })
	})

	t.Run("CorruptBitmapAllZero", func(t *testing.T) {
		keys := []uint32{10, 12, 14, 16}
		bad := encodeTestPEF(keys, defaultPEFPartitionConfig())
		clear(bad[24:])
		assertCorruptPanic(t, func() { iterateAll(bad) })
	})

	t.Run("CorruptEFSelect0AndHighBits", func(t *testing.T) {
		efKeys, cfg := makeSelect0SampleKeys(300)
		efBlob := encodeTestPEF(efKeys, cfg)

		// Corrupt select0Samples[0] to OOB.
		badSample := slices.Clone(efBlob)
		binary.LittleEndian.PutUint32(badSample[24:28], 999999)
		assertCorruptPanic(t, func() {
			it := newPEFPostingIterator(badSample, ng)
			it.next(efKeys[len(efKeys)-1] - 5)
		})

		// Corrupt highBits to all zeroes.
		badZero := slices.Clone(efBlob)
		clear(badZero[28:])
		assertCorruptPanic(t, func() { iterateAll(badZero) })

		// Corrupt highBits to all ones.
		badOne := slices.Clone(efBlob)
		for idx := 28; idx < len(badOne); idx++ {
			badOne[idx] = 0xff
		}
		assertCorruptPanic(t, func() { iterateAll(badOne) })
	})
}

func FuzzPEFPostingIterator(f *testing.F) {
	// Seed with valid multi-chunk, single-chunk, EF select0, and mutated blobs.
	var mixed []uint32
	for i := uint32(0); i < 64; i++ {
		mixed = append(mixed, 100+i)
	}
	for i := uint32(0); i < 64; i++ {
		mixed = append(mixed, 500+2*i)
	}
	cur := uint32(10000)
	for i := 0; i < 400; i++ {
		mixed = append(mixed, cur)
		cur += 600
	}
	f.Add(encodeTestPEF(mixed, defaultPEFPartitionConfig()), uint32(550))
	f.Add(encodeTestPEF([]uint32{0, 1, 2}, defaultPEFPartitionConfig()), uint32(1))
	efKeys, efCfg := makeSelect0SampleKeys(300)
	f.Add(encodeTestPEF(efKeys, efCfg), uint32(1500))
	f.Add([]byte{0, 1, 2, 3}, uint32(0))

	ng := stringToNGram("abc")

	f.Fuzz(func(t *testing.T, blob []byte, limitSeed uint32) {
		defer func() {
			if r := recover(); r != nil {
				msg := fmt.Sprint(r)
				if !strings.HasPrefix(msg, "corrupt pef posting:") {
					t.Fatalf("unexpected non-corruption panic: %v", r)
				}
			}
		}()

		// 1) Bounded sequential iteration (capped so huge AllOnes blobs terminate quickly).
		it := newPEFPostingIterator(blob, ng)
		prev := it.first()
		for steps := 0; steps < 4096 && prev != math.MaxUint32; steps++ {
			it.next(prev)
			cur := it.first()
			if cur <= prev && cur != math.MaxUint32 {
				t.Fatalf("non-increasing sequential step: cur=%d <= prev=%d", cur, prev)
			}
			prev = cur
		}

		// 2) Skip / NextGEQ jumps including seed-derived jumps and MaxUint32 boundaries.
		it2 := newPEFPostingIterator(blob, ng)
		curVal := it2.first()
		var limits []uint32
		if curVal != math.MaxUint32 {
			limits = append(limits, 0, curVal, curVal+(limitSeed&0xfff))
		}
		limits = append(limits, limitSeed, math.MaxUint32-1, math.MaxUint32)
		slices.Sort(limits)

		prevOut := uint32(0)
		hasPrev := false
		for _, lim := range limits {
			it2.next(lim)
			got := it2.first()
			if lim == math.MaxUint32 {
				if got != math.MaxUint32 {
					t.Fatalf("next(MaxUint32) = %d, want MaxUint32", got)
				}
			} else if got != math.MaxUint32 && got <= lim {
				t.Fatalf("next(%d) = %d, want > %d", lim, got, lim)
			}
			if hasPrev && got < prevOut {
				t.Fatalf("non-monotonic skip output: got=%d < prev=%d", got, prevOut)
			}
			prevOut = got
			hasPrev = true
		}

		var stats zoekt.Stats
		it2.setNgramLookups(3)
		it2.updateStats(&stats)
	})
}
