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
	"reflect"
	"testing"
)

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
	if len(got) == 0 || got[len(got)-1] != uint32(len(keys)) {
		t.Fatalf("partition = %v, want non-empty ending at %d", got, len(keys))
	}
	prev := uint32(0)
	for i, end := range got {
		if end <= prev || end > uint32(len(keys)) {
			t.Fatalf("invalid end[%d]=%d (prev=%d, n=%d)", i, end, prev, len(keys))
		}
		prev = end
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
