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
	"math"
	"math/bits"
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestPEFEFBitsizePaperMatchesDivision(t *testing.T) {
	check := func(lastRel, n uint64) {
		t.Helper()
		var want uint64
		if n != 0 {
			var lo int
			if lastRel >= n {
				lo = bits.Len64(lastRel/n) - 1
			}
			want = n + (lastRel >> lo) + 3 + n*uint64(lo)
		}
		if got := pefEFBitsizePaper(lastRel, n); got != want {
			t.Fatalf("pefEFBitsizePaper(%d, %d) = %d, want %d", lastRel, n, got, want)
		}
	}

	check(0, 0)
	check(math.MaxUint64, 0)
	for n := uint64(1); n <= 257; n++ {
		for k := 0; k <= 64-bits.Len64(n); k++ {
			edge := n << k
			check(edge-1, n)
			check(edge, n)
			check(edge+1, n)
		}
	}
	for _, n := range []uint64{1 << 31, math.MaxUint32, 1 << 63, math.MaxUint64} {
		check(0, n)
		check(n-1, n)
		check(n, n)
		check(math.MaxUint64, n)
	}

	rng := rand.New(rand.NewPCG(1, 2))
	for range 10_000 {
		lastRel := rng.Uint64()
		check(lastRel, rng.Uint64())
		check(lastRel, rng.Uint64()>>32)
	}
}

func TestPEFSelectCodec(t *testing.T) {
	cases := []struct {
		name       string
		lastRel, n uint32
		want       pefChunkKind
	}{
		{"single_key", 0, 1, pefKindAllOnes},
		{"contiguous_run", 999, 1000, pefKindAllOnes},
		{"dense_bitmap", 200, 100, pefKindBitmap},
		{"sparse_ef", 100000, 100, pefKindEF},
		{"max_uint32_range", math.MaxUint32, 2, pefKindEF},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := pefSelectCodec(tc.lastRel, tc.n); got != tc.want {
				t.Fatalf("pefSelectCodec(%d, %d) = %d, want %d", tc.lastRel, tc.n, got, tc.want)
			}
		})
	}
}

func TestPEFPartitionEmptyAndSingle(t *testing.T) {
	var s pefPartitionScratch
	cfg := defaultPEFPartitionConfig()

	if got := s.partition(nil, cfg, nil); len(got) != 0 {
		t.Fatalf("empty partition = %v, want empty", got)
	}
	if got := s.partition([]uint32{42}, cfg, nil); !reflect.DeepEqual(got, []uint32{1}) {
		t.Fatalf("single key partition = %v, want [1]", got)
	}
	if got := s.partition([]uint32{math.MaxUint32}, cfg, nil); !reflect.DeepEqual(got, []uint32{1}) {
		t.Fatalf("max uint32 single key partition = %v, want [1]", got)
	}
}

func TestPEFPartitionContiguousAndOutlier(t *testing.T) {
	keys := make([]uint32, 0, 1001)
	for i := uint32(0); i < 1000; i++ {
		keys = append(keys, i)
	}
	keys = append(keys, 100000)

	var s pefPartitionScratch
	got := s.partition(keys, defaultPEFPartitionConfig(), nil)
	want := []uint32{1000, 1001}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("partition = %v, want %v", got, want)
	}
}

func TestPEFPartitionNearMaxUint32(t *testing.T) {
	t.Run("small_near_max", func(t *testing.T) {
		keys := []uint32{
			math.MaxUint32 - 100000,
			math.MaxUint32 - 10,
			math.MaxUint32 - 9,
			math.MaxUint32 - 8,
			math.MaxUint32 - 7,
			math.MaxUint32 - 1,
			math.MaxUint32,
		}
		var s pefPartitionScratch
		got := s.partition(keys, defaultPEFPartitionConfig(), nil)
		assertValidPartition(t, got, uint32(len(keys)))
	})

	t.Run("cross_superblock_near_max", func(t *testing.T) {
		// superSize = 112; 250 keys (> 2*superSize) ending at math.MaxUint32 exercises
		// superBase = superLastKey + 1 across superblock boundaries near MaxUint32.
		const n = 250
		keys := make([]uint32, n)
		cur := uint32(math.MaxUint32) - 5000
		for i := range keys {
			if i%50 == 0 && i > 0 {
				cur += 200
			} else {
				cur++
			}
			keys[i] = cur
		}
		// Shift so the final key lands exactly on math.MaxUint32.
		shift := uint32(math.MaxUint32) - keys[n-1]
		for i := range keys {
			keys[i] += shift
		}

		cfg := pefPartitionConfig{Eps1: 0.03, Eps2: 0.3, Eps3: 1.0, FixCost: 112}
		var s pefPartitionScratch
		got := s.partition(keys, cfg, nil)
		assertValidPartition(t, got, n)
		if len(got) < 2 {
			t.Fatalf("expected multiple chunks across superblocks near MaxUint32, got %v", got)
		}

		// Also exercise superBase = (MaxUint32 - 2) + 1 = MaxUint32 - 1 directly.
		tinyCfg := pefPartitionConfig{Eps1: 0.03, Eps2: 0.3, Eps3: 1.0, FixCost: 2} // superSize = 2
		tinyKeys := []uint32{
			math.MaxUint32 - 3,
			math.MaxUint32 - 2,
			math.MaxUint32 - 1,
			math.MaxUint32,
		}
		gotTiny := s.partition(tinyKeys, tinyCfg, nil)
		assertValidPartition(t, gotTiny, uint32(len(tinyKeys)))
	})
}

func assertValidPartition(t *testing.T, ends []uint32, n uint32) {
	t.Helper()
	if len(ends) == 0 || ends[len(ends)-1] != n {
		t.Fatalf("partition = %v, want non-empty ending at %d", ends, n)
	}
	prev := uint32(0)
	for i, end := range ends {
		if end <= prev || end > n {
			t.Fatalf("invalid end[%d]=%d (prev=%d, n=%d)", i, end, prev, n)
		}
		prev = end
	}
}

func TestPEFPartitionScratchGrowAndReuse(t *testing.T) {
	makeContiguous := func(n int) []uint32 {
		ks := make([]uint32, n)
		for i := range ks {
			ks[i] = uint32(i * 3)
		}
		return ks
	}

	var s pefPartitionScratch
	cfg := defaultPEFPartitionConfig()

	// Size 100 -> need=101; Go rounds minCost ([]uint64) and path ([]uint32)
	// to different allocator size classes (cap(minCost)=112, cap(path)=104).
	// Next call with size 107 (need=108) must reallocate path without panicking.
	out := s.partition(makeContiguous(100), cfg, nil)
	assertValidPartition(t, out, 100)

	out = s.partition(makeContiguous(107), cfg, out[:0])
	assertValidPartition(t, out, 107)

	// Grow step-by-step across all allocator size classes 1..256 on a fresh scratch.
	var growScratch pefPartitionScratch
	var growOut []uint32
	all256 := makeContiguous(256)
	for sz := 1; sz <= 256; sz++ {
		growOut = growScratch.partition(all256[:sz], cfg, growOut[:0])
		assertValidPartition(t, growOut, uint32(sz))
	}

	// Explicitly exercise mismatched capacities (cap(minCost) > cap(path)),
	// which Go's runtime allocator can produce via independent size classes.
	mismatchScratch := pefPartitionScratch{
		minCost: make([]uint64, 0, 200),
		path:    make([]uint32, 0, 100),
	}
	mismatchOut := mismatchScratch.partition(makeContiguous(150), cfg, nil)
	assertValidPartition(t, mismatchOut, 150)

	keys1000 := makeContiguous(1000)
	out = s.partition(keys1000, cfg, out[:0])
	assertValidPartition(t, out, 1000)

	out = s.partition(makeContiguous(500), cfg, out[:0])
	assertValidPartition(t, out, 500)

	// Warm up at 1000 again, then assert 0 allocs on steady-state reuse.
	out = s.partition(keys1000, cfg, out[:0])
	expected := append([]uint32(nil), out...)
	allocs := testing.AllocsPerRun(5, func() {
		out = s.partition(keys1000, cfg, out[:0])
	})
	if allocs != 0 {
		t.Fatalf("steady-state allocs = %v, want 0", allocs)
	}
	if !reflect.DeepEqual(out, expected) {
		t.Fatalf("reused partition output differed")
	}
}

func exactPEFPartitionCost(keys []uint32, fixCost uint64) uint64 {
	n := len(keys)
	if n == 0 {
		return 0
	}
	dp := make([]uint64, n+1)
	for i := 1; i <= n; i++ {
		dp[i] = ^uint64(0)
	}
	for i := 0; i < n; i++ {
		base := uint64(keys[0])
		if i > 0 {
			base = uint64(keys[i-1]) + 1
		}
		for j := i + 1; j <= n; j++ {
			lastRel := uint64(keys[j-1]) - base
			c := dp[i] + pefMinCodecBitsize(lastRel, uint64(j-i)) + fixCost
			if c < dp[j] {
				dp[j] = c
			}
		}
	}
	return dp[n]
}

func evalPEFPartitionCost(keys, ends []uint32, fixCost uint64) uint64 {
	var total uint64
	prev := uint32(0)
	base := uint64(keys[0])
	for _, end := range ends {
		lastKey := uint64(keys[end-1])
		total += pefMinCodecBitsize(lastKey-base, uint64(end-prev)) + fixCost
		prev = end
		base = lastKey + 1
	}
	return total
}

func TestPEFPartitionDPExactComparisonAndTailMerge(t *testing.T) {
	cfg := defaultPEFPartitionConfig()

	// Mixed sequence: contiguous run, dense-with-gaps, sparse run, outlier, another run.
	var mixed []uint32
	for i := uint32(0); i < 60; i++ {
		mixed = append(mixed, i)
	}
	cur := uint32(65)
	for i := 0; i < 50; i++ {
		if i%2 == 0 {
			cur += 2
		} else {
			cur += 3
		}
		mixed = append(mixed, cur)
	}
	cur += 50000
	for i := 0; i < 40; i++ {
		mixed = append(mixed, cur)
		cur += 400
	}
	mixed = append(mixed, cur+1000000)
	cur += 1000001
	for i := uint32(0); i < 50; i++ {
		mixed = append(mixed, cur+i)
	}

	var s pefPartitionScratch
	got := s.partition(mixed, cfg, nil)
	assertValidPartition(t, got, uint32(len(mixed)))

	exactCost := exactPEFPartitionCost(mixed, cfg.FixCost)
	gotCost := evalPEFPartitionCost(mixed, got, cfg.FixCost)
	if gotCost < exactCost {
		t.Fatalf("gotCost=%d < exactCost=%d (impossible)", gotCost, exactCost)
	}
	maxAllowed := float64(exactCost) * (1 + cfg.Eps1) * (1 + cfg.Eps2)
	if float64(gotCost) > maxAllowed {
		t.Fatalf("gotCost=%d exceeds (1+eps1)*(1+eps2)*exactCost=%.1f (exact=%d)", gotCost, maxAllowed, exactCost)
	}

	// Tail-merge boundary checks using superSize = 112 (FixCost=112, Eps3=1.0).
	tailCfg := pefPartitionConfig{Eps1: 0.03, Eps2: 0.3, Eps3: 1.0, FixCost: 112}
	superSize := pefSuperblockSize(tailCfg.FixCost, tailCfg.Eps3)
	if superSize != 112 {
		t.Fatalf("unexpected superSize=%d, want 112", superSize)
	}

	// n = superSize + 1 contiguous keys: tail of 1 is merged into 1 superblock → single AllOnes chunk.
	contigPlusOne := make([]uint32, superSize+1)
	for i := range contigPlusOne {
		contigPlusOne[i] = uint32(i)
	}
	gotMerged := s.partition(contigPlusOne, tailCfg, nil)
	if !reflect.DeepEqual(gotMerged, []uint32{uint32(superSize + 1)}) {
		t.Fatalf("n=superSize+1 contiguous partition = %v, want [%d]", gotMerged, superSize+1)
	}

	// n = 2*superSize contiguous keys: remaining tail == superSize is NOT merged → splits at superSize.
	contigTwoSupers := make([]uint32, 2*superSize)
	for i := range contigTwoSupers {
		contigTwoSupers[i] = uint32(i)
	}
	gotTwo := s.partition(contigTwoSupers, tailCfg, nil)
	wantTwo := []uint32{uint32(superSize), uint32(2 * superSize)}
	if !reflect.DeepEqual(gotTwo, wantTwo) {
		t.Fatalf("n=2*superSize contiguous partition = %v, want %v", gotTwo, wantTwo)
	}
}

func TestPEFPartitionCrossSuperblock(t *testing.T) {
	const N = 25000
	keys := make([]uint32, 0, N)
	var cur uint32
	for i := 0; i < N; i++ {
		if i%500 == 0 && i > 0 {
			cur += 10000
		} else {
			cur++
		}
		keys = append(keys, cur)
	}

	var s pefPartitionScratch
	cfg := defaultPEFPartitionConfig()
	out := s.partition(keys, cfg, nil)

	if len(out) < 2 {
		t.Fatalf("expected multiple chunks across %d keys, got %v", N, out)
	}
	if out[len(out)-1] != N {
		t.Fatalf("last end = %d, want %d", out[len(out)-1], N)
	}
	prev := uint32(0)
	for i, end := range out {
		if end <= prev || end > N {
			t.Fatalf("out[%d] = %d not strictly ascending after %d (N=%d)", i, end, prev, N)
		}
		prev = end
	}

	// Also verify multi-superblock (>2 superblocks) with smaller Eps3.
	cfgMulti := cfg
	cfgMulti.Eps3 = 0.02 // superSize = 5600 → 4 superblocks (5600, 5600, 5600, 8200)
	outMulti := s.partition(keys, cfgMulti, out[:0])
	if outMulti[len(outMulti)-1] != N {
		t.Fatalf("multi-superblock last end = %d, want %d", outMulti[len(outMulti)-1], N)
	}
	prev = 0
	for i, end := range outMulti {
		if end <= prev || end > N {
			t.Fatalf("outMulti[%d] = %d not strictly ascending after %d (N=%d)", i, end, prev, N)
		}
		prev = end
	}

	// Verify steady-state 0 allocations when reusing scratch and out.
	expected := append([]uint32(nil), outMulti...)
	allocs := testing.AllocsPerRun(5, func() {
		outMulti = s.partition(keys, cfgMulti, outMulti[:0])
	})
	if allocs != 0 {
		t.Fatalf("steady-state allocs = %v, want 0", allocs)
	}
	if !reflect.DeepEqual(outMulti, expected) {
		t.Fatalf("reused partition output differed from first run")
	}
}

func TestPEFPartitionConfigValidationAndEdgeCases(t *testing.T) {
	invalidConfigs := []struct {
		name string
		cfg  pefPartitionConfig
	}{
		{"Eps1NaN", pefPartitionConfig{Eps1: math.NaN(), Eps2: 0.3, Eps3: 0.01, FixCost: 112}},
		{"Eps1Negative", pefPartitionConfig{Eps1: -0.1, Eps2: 0.3, Eps3: 0.01, FixCost: 112}},
		{"Eps1One", pefPartitionConfig{Eps1: 1.0, Eps2: 0.3, Eps3: 0.01, FixCost: 112}},
		{"Eps2NaN", pefPartitionConfig{Eps1: 0.03, Eps2: math.NaN(), Eps3: 0.01, FixCost: 112}},
		{"Eps2PlusInf", pefPartitionConfig{Eps1: 0.03, Eps2: math.Inf(1), Eps3: 0.01, FixCost: 112}},
		{"Eps2Zero", pefPartitionConfig{Eps1: 0.03, Eps2: 0, Eps3: 0.01, FixCost: 112}},
		{"Eps3NaN", pefPartitionConfig{Eps1: 0.03, Eps2: 0.3, Eps3: math.NaN(), FixCost: 112}},
		{"Eps3PlusInf", pefPartitionConfig{Eps1: 0.03, Eps2: 0.3, Eps3: math.Inf(1), FixCost: 112}},
		{"Eps3Negative", pefPartitionConfig{Eps1: 0.03, Eps2: 0.3, Eps3: -0.01, FixCost: 112}},
		{"FixCostZero", pefPartitionConfig{Eps1: 0.03, Eps2: 0.3, Eps3: 0.01, FixCost: 0}},
		{"FixCostTooLarge", pefPartitionConfig{Eps1: 0.03, Eps2: 0.3, Eps3: 0.01, FixCost: (1 << 30) + 1}},
		{"FixCostMaxUint64", pefPartitionConfig{Eps1: 0.03, Eps2: 0.3, Eps3: 0.01, FixCost: math.MaxUint64}},
	}
	for _, tc := range invalidConfigs {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.cfg.validate(); err == nil {
				t.Fatalf("expected validate() error for %+v", tc.cfg)
			}
		})
	}

	if got := pefSuperblockSize(1<<30, 1e-10); got != 1<<30 {
		t.Fatalf("pefSuperblockSize overflow clamp = %d, want %d", got, 1<<30)
	}
	if got := pefSuperblockSize(112, 0); got != 1<<30 {
		t.Fatalf("pefSuperblockSize eps3=0 = %d, want %d", got, 1<<30)
	}

	// Huge finite Eps2 must terminate cleanly without float-to-uint64 overflow or infinite loops.
	hugeEps2Cfg := pefPartitionConfig{Eps1: 0.03, Eps2: 1e20, Eps3: 0.01, FixCost: 112}
	if err := hugeEps2Cfg.validate(); err != nil {
		t.Fatalf("huge finite Eps2 validate() unexpected error: %v", err)
	}
	keys := make([]uint32, 200)
	for i := range keys {
		keys[i] = uint32(i * 100)
	}
	var s pefPartitionScratch
	got := s.partition(keys, hugeEps2Cfg, nil)
	assertValidPartition(t, got, uint32(len(keys)))
}
