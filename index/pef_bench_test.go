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
	"context"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/sourcegraph/zoekt"
	"github.com/sourcegraph/zoekt/query"
)

// Package-level sinks prevent the compiler from eliminating benchmark loops.
var (
	pefBenchSinkU32   uint32
	pefBenchSinkBytes int
)

// genSyntheticPostings generates m strictly increasing uint32 posting positions
// for the named distribution ("uniform", "clustered_code", "all_ones", "sparse").
func genSyntheticPostings(dist string, m int, seed int64) []uint32 {
	if m <= 0 {
		return nil
	}
	rng := rand.New(rand.NewSource(seed))
	keys := make([]uint32, m)

	switch dist {
	case "uniform":
		// Uniform spacing with mean gap ~64 (1..127), keeping 1M keys within ~64M.
		cur := uint32(100 + rng.Intn(64))
		for i := range keys {
			keys[i] = cur
			cur += uint32(1 + rng.Intn(127))
		}
	case "clustered_code":
		// Simulates a common code trigram: dense clusters (e.g. files/blocks of
		// 16..128 occurrences with gaps 1..16) separated by inter-file gaps (1K..10K).
		cur := uint32(200)
		clusterLeft := 0
		for i := range keys {
			keys[i] = cur
			if clusterLeft <= 0 {
				clusterLeft = 16 + rng.Intn(113)
				cur += uint32(1000 + rng.Intn(9000))
			} else {
				clusterLeft--
				cur += uint32(1 + rng.Intn(16))
			}
		}
	case "all_ones":
		// Pure contiguous run of positions (best-case AllOnes chunking).
		const base = uint32(1000)
		for i := range keys {
			keys[i] = base + uint32(i)
		}
	case "sparse":
		// Large gaps (~1000..9000) typical of multi-byte varints.
		cur := uint32(1000 + rng.Intn(4000))
		for i := range keys {
			keys[i] = cur
			cur += uint32(1000 + rng.Intn(8000))
		}
	default:
		panic(fmt.Sprintf("unknown synthetic distribution %q", dist))
	}
	return keys
}

// genSortedLimits generates k sorted skip limits spanning [0, maxKey], so skips
// exercise the full range of the posting list without exhausting on step 0.
func genSortedLimits(k int, maxKey uint32, seed int64) []uint32 {
	if k <= 0 || maxKey == 0 {
		return nil
	}
	rng := rand.New(rand.NewSource(seed))
	limits := make([]uint32, k)
	span := uint64(maxKey) + 1
	for i := range limits {
		limits[i] = uint32(rng.Int63n(int64(span)))
	}
	slices.Sort(limits)
	return limits
}

func BenchmarkPEFEncode(b *testing.B) {
	dists := []string{"uniform", "clustered_code", "all_ones", "sparse"}
	sizes := []int{128, 1024, 16384, 262144}

	runCase := func(b *testing.B, dist string, m int, encode func([]uint32) []byte) {
		keys := genSyntheticPostings(dist, m, int64(m)*31+17)
		varintLen := len(toDeltas(keys))
		// Warm up reusable scratch buffers before timing.
		blob := encode(keys)
		pefLen := len(blob)

		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			blob = encode(keys)
			pefBenchSinkBytes = len(blob)
		}
		elapsed := b.Elapsed().Seconds()
		if elapsed > 0 {
			mpostPerSec := float64(m) * float64(b.N) / elapsed / 1e6
			b.ReportMetric(mpostPerSec, "M-post/s")
		}
		b.ReportMetric(float64(pefLen*8)/float64(m), "bpk")
		b.ReportMetric(float64(pefLen)/float64(varintLen), "ratio-vs-varint")
	}

	defCfg := defaultPEFPartitionConfig()
	for _, dist := range dists {
		for _, m := range sizes {
			b.Run(fmt.Sprintf("default/dist=%s/m=%d", dist, m), func(b *testing.B) {
				runCase(b, dist, m, newPEFEncoder(defCfg).encodePositions)
			})
			for _, shift := range []uint8{7, 8} {
				b.Run(fmt.Sprintf("Uniform%d/dist=%s/m=%d", 1<<shift, dist, m), func(b *testing.B) {
					runCase(b, dist, m, newPEFUniformEncoder(shift).encodePositions)
				})
			}
		}
	}

	fixCosts := []uint64{64, 112, 256}
	eps1s := []float64{0.03, 0.1}
	for _, m := range []int{16384, 262144} {
		for _, fix := range fixCosts {
			for _, eps1 := range eps1s {
				cfg := defCfg
				cfg.FixCost = fix
				cfg.Eps1 = eps1
				b.Run(fmt.Sprintf("sweep/dist=clustered_code/m=%d/fix=%d/eps1=%.2f", m, fix, eps1), func(b *testing.B) {
					runCase(b, "clustered_code", m, newPEFEncoder(cfg).encodePositions)
				})
			}
		}
	}
}

func BenchmarkPEFUniformIterator(b *testing.B) {
	const size = 262144
	keys := genSyntheticPostings("clustered_code", size, int64(size)*31+17)
	limits := genSortedLimits(100, keys[len(keys)-1], 17)
	ng := stringToNGram("abc")
	for _, shift := range []uint8{7, 8} {
		enc := newPEFUniformEncoder(shift)
		blob := enc.encodePositions(keys)
		name := fmt.Sprintf("Uniform%d/m=%d", 1<<shift, size)
		b.Run(name+"/seq", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				it := newPEFPostingIteratorWithBlockShift(blob, ng, shift)
				var sum uint32
				for cur := it.first(); cur != math.MaxUint32; cur = it.first() {
					sum += cur
					it.next(cur)
				}
				pefBenchSinkU32 = sum
			}
		})
		b.Run(name+"/skip", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				it := newPEFPostingIteratorWithBlockShift(blob, ng, shift)
				var sum uint32
				for _, limit := range limits {
					it.next(limit)
					sum += it.first()
				}
				pefBenchSinkU32 = sum
			}
		})
	}
}

func BenchmarkPostingIterator_VarintVsPEF(b *testing.B) {
	ng := stringToNGram("abc")
	dists := []string{"uniform", "clustered_code"}
	skipCases := []struct {
		size      int
		limitSize int
	}{
		{10_000, 100},
		{100_000, 100},
		{100_000, 10_000},
		{1_000_000, 100},
		{1_000_000, 1_000_000},
	}

	for _, dist := range dists {
		for _, tc := range skipCases {
			keys := genSyntheticPostings(dist, tc.size, int64(tc.size)*13+7)
			limits := genSortedLimits(tc.limitSize, keys[len(keys)-1], int64(tc.limitSize)*97+3)
			deltas := toDeltas(keys)
			enc := newPEFEncoder(defaultPEFPartitionConfig())
			pefBlob := slices.Clone(enc.encodePositions(keys))

			name := fmt.Sprintf("skip/dist=%s/m=%d/limits=%d", dist, tc.size, tc.limitSize)
			b.Run(name+"/Varint", func(b *testing.B) {
				var totalLoaded int64
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					it := newCompressedPostingIterator(deltas, ng)
					var sum uint32
					for _, lim := range limits {
						it.next(lim)
						sum += it.first()
					}
					pefBenchSinkU32 = sum
					var s zoekt.Stats
					it.updateStats(&s)
					totalLoaded += s.IndexBytesLoaded
				}
				b.ReportMetric(float64(totalLoaded)/float64(b.N), "idx-bytes/op")
			})

			b.Run(name+"/PEF", func(b *testing.B) {
				var totalLoaded int64
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					it := newPEFPostingIterator(pefBlob, ng)
					var sum uint32
					for _, lim := range limits {
						it.next(lim)
						sum += it.first()
					}
					pefBenchSinkU32 = sum
					var s zoekt.Stats
					it.updateStats(&s)
					totalLoaded += s.IndexBytesLoaded
				}
				b.ReportMetric(float64(totalLoaded)/float64(b.N), "idx-bytes/op")
			})
		}

		for _, size := range []int{100_000, 1_000_000} {
			keys := genSyntheticPostings(dist, size, int64(size)*19+11)
			deltas := toDeltas(keys)
			enc := newPEFEncoder(defaultPEFPartitionConfig())
			pefBlob := slices.Clone(enc.encodePositions(keys))

			name := fmt.Sprintf("seq/dist=%s/m=%d", dist, size)
			b.Run(name+"/Varint", func(b *testing.B) {
				var totalLoaded int64
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					it := newCompressedPostingIterator(deltas, ng)
					var sum uint32
					for cur := it.first(); cur != math.MaxUint32; cur = it.first() {
						sum += cur
						it.next(cur)
					}
					pefBenchSinkU32 = sum
					var s zoekt.Stats
					it.updateStats(&s)
					totalLoaded += s.IndexBytesLoaded
				}
				b.ReportMetric(float64(totalLoaded)/float64(b.N), "idx-bytes/op")
			})

			b.Run(name+"/PEF", func(b *testing.B) {
				var totalLoaded int64
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					it := newPEFPostingIterator(pefBlob, ng)
					var sum uint32
					for cur := it.first(); cur != math.MaxUint32; cur = it.first() {
						sum += cur
						it.next(cur)
					}
					pefBenchSinkU32 = sum
					var s zoekt.Stats
					it.updateStats(&s)
					totalLoaded += s.IndexBytesLoaded
				}
				b.ReportMetric(float64(totalLoaded)/float64(b.N), "idx-bytes/op")
			})
		}
	}
}

// genDistancePair generates two clustered-code posting lists over the same
// universe, planting occasional matches at cur+dist so distanceHitIterator
// exercises both skipping across gaps and matching hits.
func genDistancePair(m1, m2 int, dist uint32, seed int64) (keys1, keys2 []uint32) {
	keys2 = genSyntheticPostings("clustered_code", m2, seed)
	rng := rand.New(rand.NewSource(seed + 101))
	keys1 = make([]uint32, m1)
	step := m2 / m1
	if step < 1 {
		step = 1
	}
	for i := range keys1 {
		idx2 := (i * step) % m2
		anchor := keys2[idx2]
		if i%4 == 0 && anchor >= dist {
			// Exact distance hit at anchor - dist.
			keys1[i] = anchor - dist
		} else {
			// Near-miss offset inside the same region.
			off := uint32(1 + rng.Intn(64))
			if anchor > dist+off {
				keys1[i] = anchor - dist - off
			} else {
				keys1[i] = anchor + off
			}
		}
	}
	keys1 = sortedUnique(keys1)
	return keys1, keys2
}

func BenchmarkDistanceHitIterator_VarintVsPEF(b *testing.B) {
	ng1 := stringToNGram("abc")
	ng2 := stringToNGram("def")
	const dist = uint32(3)

	cases := []struct {
		name string
		m1   int
		m2   int
	}{
		{name: "rare_50x500K", m1: 50, m2: 500_000},
		{name: "medium_2Kx500K", m1: 2_000, m2: 500_000},
		{name: "frequent_100Kx500K", m1: 100_000, m2: 500_000},
	}

	for i, tc := range cases {
		keys1, keys2 := genDistancePair(tc.m1, tc.m2, dist, int64(i+1)*1009)
		deltas1 := toDeltas(keys1)
		deltas2 := toDeltas(keys2)

		enc := newPEFEncoder(defaultPEFPartitionConfig())
		var pef1 []byte
		if len(keys1) >= defaultPEFMinPostings {
			pef1 = slices.Clone(enc.encodePositions(keys1))
		}
		pef2 := slices.Clone(enc.encodePositions(keys2))

		runIter := func(b *testing.B, usePEF bool) {
			var totalLoaded int64
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				var it1, it2 hitIterator
				if usePEF && pef1 != nil {
					it1 = newPEFPostingIterator(pef1, ng1)
				} else {
					it1 = newCompressedPostingIterator(deltas1, ng1)
				}
				if usePEF {
					it2 = newPEFPostingIterator(pef2, ng2)
				} else {
					it2 = newCompressedPostingIterator(deltas2, ng2)
				}
				dit := &distanceHitIterator{i1: it1, i2: it2, distance: dist}
				var hits uint32
				for cur := dit.first(); cur != math.MaxUint32; cur = dit.first() {
					hits += cur
					dit.next(cur)
				}
				pefBenchSinkU32 = hits
				var s zoekt.Stats
				dit.updateStats(&s)
				totalLoaded += s.IndexBytesLoaded
			}
			b.ReportMetric(float64(totalLoaded)/float64(b.N), "idx-bytes/op")
		}

		b.Run(tc.name+"/Varint", func(b *testing.B) {
			runIter(b, false)
		})
		b.Run(tc.name+"/PEF", func(b *testing.B) {
			runIter(b, true)
		})
	}
}

func buildSmallRepoDocs(numFiles, funcsPerFile int) []Document {
	docs := make([]Document, 0, numFiles)
	for i := range numFiles {
		var sb strings.Builder
		fmt.Fprintf(&sb, "package svc%d\n\n", i)
		for j := range funcsPerFile {
			fmt.Fprintf(&sb, "func processTask%d_%d(input string) int {\n", i, j)
			fmt.Fprintf(&sb, "\t// commonLoopMarker handles step_%d\n", j)
			fmt.Fprintf(&sb, "\tv%d := len(input) + defaultOffsetValue\n", j)
			fmt.Fprintf(&sb, "\treturn v%d\n", j)
			sb.WriteString("}\n")
		}
		if i == numFiles/2 {
			sb.WriteString("// uniqueAnchorNeedle997 target marker\n")
		}
		docs = append(docs, Document{
			Name:     fmt.Sprintf("pkg/sub%d/file_%04d.go", i%10, i),
			Content:  []byte(sb.String()),
			Branches: []string{"main"},
		})
	}
	return docs
}

func buildMemShardWithPEF(tb testing.TB, docs []Document, pefMinPostings int) []byte {
	tb.Helper()
	sb, err := NewShardBuilder(&zoekt.Repository{
		Name:     "small-bench-repo",
		Branches: []zoekt.RepositoryBranch{{Name: "main", Version: "v1"}},
	})
	if err != nil {
		tb.Fatalf("NewShardBuilder: %v", err)
	}
	sb.pefMinPostings = pefMinPostings
	sb.IndexTime = time.Unix(1700000000, 0).UTC()
	sb.ID = "bench-small-repo-0001"
	for _, doc := range docs {
		if err := sb.Add(doc); err != nil {
			tb.Fatalf("ShardBuilder.Add(%s): %v", doc.Name, err)
		}
	}
	var buf bytes.Buffer
	if err := sb.Write(&buf); err != nil {
		tb.Fatalf("ShardBuilder.Write: %v", err)
	}
	return buf.Bytes()
}

func BenchmarkSmallRepo_VarintVsPEF(b *testing.B) {
	repos := []struct {
		name         string
		numFiles     int
		funcsPerFile int
	}{
		{name: "files=10", numFiles: 10, funcsPerFile: 4},
		{name: "files=100", numFiles: 100, funcsPerFile: 4},
		{name: "files=500", numFiles: 500, funcsPerFile: 10},
	}

	variants := []struct {
		name           string
		pefMinPostings int
	}{
		{name: "Varint", pefMinPostings: -1},
		{name: "PEF", pefMinPostings: defaultPEFMinPostings},
	}

	queries := []struct {
		name string
		q    query.Q
	}{
		{
			name: "common",
			q:    &query.Substring{Pattern: "commonLoopMarker", Content: true, CaseSensitive: true},
		},
		{
			name: "rare",
			q:    &query.Substring{Pattern: "uniqueAnchorNeedle997", Content: true, CaseSensitive: true},
		},
	}

	ctx := context.Background()
	sOpts := &zoekt.SearchOptions{}

	for _, repo := range repos {
		docs := buildSmallRepoDocs(repo.numFiles, repo.funcsPerFile)

		for _, v := range variants {
			b.Run(fmt.Sprintf("%s/index/%s", repo.name, v.name), func(b *testing.B) {
				var buf bytes.Buffer
				var lastSize int
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					buf.Reset()
					sb, err := NewShardBuilder(&zoekt.Repository{
						Name:     "small-bench-repo",
						Branches: []zoekt.RepositoryBranch{{Name: "main", Version: "v1"}},
					})
					if err != nil {
						b.Fatalf("NewShardBuilder: %v", err)
					}
					sb.pefMinPostings = v.pefMinPostings
					for _, doc := range docs {
						if err := sb.Add(doc); err != nil {
							b.Fatalf("Add: %v", err)
						}
					}
					if err := sb.Write(&buf); err != nil {
						b.Fatalf("Write: %v", err)
					}
					lastSize = buf.Len()
				}
				pefBenchSinkBytes = lastSize
				b.ReportMetric(float64(lastSize), "shard-bytes")
			})

			shardBytes := buildMemShardWithPEF(b, docs, v.pefMinPostings)
			searcher, err := NewSearcher(&memSeeker{data: shardBytes})
			if err != nil {
				b.Fatalf("NewSearcher: %v", err)
			}

			for _, qtc := range queries {
				b.Run(fmt.Sprintf("%s/search_%s/%s", repo.name, qtc.name, v.name), func(b *testing.B) {
					var totalLoaded int64
					b.ReportAllocs()
					b.ResetTimer()
					for b.Loop() {
						res, err := searcher.Search(ctx, qtc.q, sOpts)
						if err != nil {
							b.Fatalf("Search: %v", err)
						}
						pefBenchSinkBytes = len(res.Files)
						totalLoaded += res.Stats.IndexBytesLoaded
					}
					b.ReportMetric(float64(totalLoaded)/float64(b.N), "idx-bytes/op")
				})
			}
			searcher.Close()
		}
	}
}

func TestPEFSmallRepoAntiPathology(t *testing.T) {
	ctx := context.Background()
	sOpts := &zoekt.SearchOptions{}

	openMem := func(t *testing.T, shardBytes []byte) (zoekt.Searcher, *indexTOC) {
		t.Helper()
		f := &memSeeker{data: shardBytes}
		rd := &reader{r: f}
		var toc indexTOC
		if err := rd.readTOC(&toc); err != nil {
			t.Fatalf("readTOC: %v", err)
		}
		s, err := NewSearcher(f)
		if err != nil {
			t.Fatalf("NewSearcher: %v", err)
		}
		t.Cleanup(s.Close)
		return s, &toc
	}

	assertParity := func(t *testing.T, varintS, pefS zoekt.Searcher, queries []query.Q, requireNonEmpty bool) {
		t.Helper()
		for _, q := range queries {
			want, err := varintS.Search(ctx, q, sOpts)
			if err != nil {
				t.Fatalf("varint search %s: %v", q, err)
			}
			got, err := pefS.Search(ctx, q, sOpts)
			if err != nil {
				t.Fatalf("PEF search %s: %v", q, err)
			}
			if requireNonEmpty && len(want.Files) == 0 {
				t.Fatalf("expected non-empty matches for %s", q)
			}
			if diff := cmp.Diff(want.Files, got.Files); diff != "" {
				t.Fatalf("search mismatch for %s (-varint +pef):\n%s", q, diff)
			}
		}
	}

	t.Run("AllOnes_100KB_AAAA", func(t *testing.T) {
		const size = 100 * 1024
		docs := []Document{{
			Name:     "aaaa.txt",
			Content:  bytes.Repeat([]byte("a"), size),
			Branches: []string{"main"},
		}}
		varintBytes := buildMemShardWithPEF(t, docs, -1)
		pefBytes := buildMemShardWithPEF(t, docs, defaultPEFMinPostings)

		varintS, varintTOC := openMem(t, varintBytes)
		pefS, pefTOC := openMem(t, pefBytes)

		if varintTOC.postingsPEFBitmap.sz != 0 {
			t.Fatalf("varint postingsPEFBitmap.sz = %d, want 0", varintTOC.postingsPEFBitmap.sz)
		}
		if pefTOC.postingsPEFBitmap.sz == 0 {
			t.Fatalf("PEF postingsPEFBitmap.sz = 0 for 100KB 'a' run, want > 0")
		}
		// Pure AllOnes compresses ~100K postings into a few bytes vs ~100KB varints.
		if len(pefBytes) >= len(varintBytes) {
			t.Fatalf("expected PEF shard (%d B) < varint shard (%d B) on 100KB AllOnes", len(pefBytes), len(varintBytes))
		}
		assertParity(t, varintS, pefS, []query.Q{
			&query.Substring{Pattern: "aaa", Content: true, CaseSensitive: true},
			&query.Substring{Pattern: "aaaaaa", Content: true, CaseSensitive: true},
		}, true)
	})

	t.Run("HighEntropyRandomASCII_100KB", func(t *testing.T) {
		const size = 100 * 1024
		rng := rand.New(rand.NewSource(42))
		raw := make([]byte, size)
		for i := range raw {
			raw[i] = byte(32 + rng.Intn(95))
		}
		docs := []Document{{
			Name:     "random.txt",
			Content:  raw,
			Branches: []string{"main"},
		}}
		varintBytes := buildMemShardWithPEF(t, docs, -1)
		pefBytes := buildMemShardWithPEF(t, docs, defaultPEFMinPostings)

		varintS, _ := openMem(t, varintBytes)
		pefS, pefTOC := openMem(t, pefBytes)

		if pefTOC.postingsPEFBitmap.sz != 0 {
			t.Fatalf("high-entropy 100KB postingsPEFBitmap.sz = %d, want 0", pefTOC.postingsPEFBitmap.sz)
		}
		if !bytes.Equal(pefBytes, varintBytes) {
			t.Fatalf("high-entropy 100KB shard differs from varint baseline: len(pef)=%d, len(varint)=%d", len(pefBytes), len(varintBytes))
		}
		needle := string(raw[1000:1012])
		assertParity(t, varintS, pefS, []query.Q{
			&query.Substring{Pattern: needle, Content: true, CaseSensitive: true},
		}, true)
	})

	for _, numFiles := range []int{10, 100} {
		t.Run(fmt.Sprintf("TinyRepo_%dFiles", numFiles), func(t *testing.T) {
			docs := buildSmallRepoDocs(numFiles, 4)
			varintBytes := buildMemShardWithPEF(t, docs, -1)
			pefBytes := buildMemShardWithPEF(t, docs, defaultPEFMinPostings)

			varintS, _ := openMem(t, varintBytes)
			pefS, pefTOC := openMem(t, pefBytes)

			if pefTOC.postingsPEFBitmap.sz != 0 {
				t.Fatalf("tiny %d-file repo postingsPEFBitmap.sz = %d, want 0", numFiles, pefTOC.postingsPEFBitmap.sz)
			}
			if !bytes.Equal(pefBytes, varintBytes) {
				t.Fatalf("tiny %d-file shard differs from varint baseline: len(pef)=%d, len(varint)=%d", numFiles, len(pefBytes), len(varintBytes))
			}
			assertParity(t, varintS, pefS, []query.Q{
				&query.Substring{Pattern: "commonLoopMarker", Content: true, CaseSensitive: true},
				&query.Substring{Pattern: "uniqueAnchorNeedle997", Content: true, CaseSensitive: true},
				&query.Substring{Pattern: "defaultoffsetvalue", Content: true, CaseSensitive: false},
			}, true)
		})
	}
}
