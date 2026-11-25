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
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/sourcegraph/zoekt"
	"github.com/sourcegraph/zoekt/query"
)

func buildTestCorpusDocs() []Document {
	docs := make([]Document, 0, 32)
	for i := range 30 {
		var sb strings.Builder
		fmt.Fprintf(&sb, "package pkg%d\n\n", i)
		for j := range 12 {
			fmt.Fprintf(&sb, "func handleItem%d_%d() int {\n", i, j)
			sb.WriteString("\tval := commonTokenAlpha + mixedCaseWord\n")
			sb.WriteString("\treturn val\n")
			sb.WriteString("}\n")
		}
		if i%6 == 0 {
			sb.WriteString("// MIXEDCASEWORD appears rarely in uppercase\n")
			sb.WriteString("// COMMONTOKENBETA sparse token\n")
		}
		if i == 3 || i == 14 || i == 27 {
			sb.WriteString("// rareNeedleXYZ123 marker\n")
		}
		if i == 5 || i == 22 {
			sb.WriteString("// rarePrefixcommonTokenAlpha combined hit\n")
		}
		docs = append(docs, Document{
			Name:     fmt.Sprintf("dir/file_%02d_alpha.go", i),
			Content:  []byte(sb.String()),
			Branches: []string{"main"},
		})
	}
	return docs
}

func buildTestShardBytes(t *testing.T, repoName string, docs []Document, pefMinPostings int, indexTime time.Time, id string) ([]byte, string) {
	t.Helper()
	dir := t.TempDir()
	opts := Options{
		IndexDir: dir,
		RepositoryDescription: zoekt.Repository{
			Name:     repoName,
			Branches: []zoekt.RepositoryBranch{{Name: "main", Version: "v1"}},
		},
		DisableCTags:   true,
		Parallelism:    1,
		PEFMinPostings: pefMinPostings,
	}
	opts.SetDefaults()

	b, err := NewBuilder(opts)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	b.indexTime = indexTime
	b.id = id
	for _, doc := range docs {
		if err := b.Add(doc); err != nil {
			t.Fatalf("Builder.Add(%s): %v", doc.Name, err)
		}
	}
	if err := b.Finish(); err != nil {
		t.Fatalf("Builder.Finish: %v", err)
	}
	shardPath := opts.shardName(0)
	raw, err := os.ReadFile(shardPath)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", shardPath, err)
	}
	return raw, shardPath
}

func openTestSearcher(t *testing.T, shardPath string) (zoekt.Searcher, *indexTOC) {
	t.Helper()
	f, err := os.Open(shardPath)
	if err != nil {
		t.Fatalf("os.Open(%s): %v", shardPath, err)
	}
	inf, err := NewIndexFile(f)
	if err != nil {
		f.Close()
		t.Fatalf("NewIndexFile(%s): %v", shardPath, err)
	}
	t.Cleanup(inf.Close)

	rd := &reader{r: inf}
	var toc indexTOC
	if err := rd.readTOC(&toc); err != nil {
		t.Fatalf("readTOC(%s): %v", shardPath, err)
	}
	s, err := NewSearcher(inf)
	if err != nil {
		t.Fatalf("NewSearcher(%s): %v", shardPath, err)
	}
	return s, &toc
}

func TestPEFShardSearchParity(t *testing.T) {
	for _, blockSize := range []string{"", "128", "256"} {
		name := "Opt"
		if blockSize != "" {
			name = "Uniform" + blockSize
		}
		t.Run(name, func(t *testing.T) {
			t.Setenv("ZOEKT_PEF_UNIFORM_BLOCK_SIZE", blockSize)
			testPEFShardSearchParity(t, blockSize != "")
		})
	}
}

func testPEFShardSearchParity(t *testing.T, uniform bool) {
	t.Helper()
	docs := buildTestCorpusDocs()
	fixedTime := time.Unix(1700000000, 0).UTC()

	_, varintPath := buildTestShardBytes(t, "parity-repo", docs, -1, fixedTime, "parity00000000000001")
	_, pefPath := buildTestShardBytes(t, "parity-repo", docs, 128, fixedTime, "parity00000000000001")

	varintSearcher, varintTOC := openTestSearcher(t, varintPath)
	pefSearcher, pefTOC := openTestSearcher(t, pefPath)

	if varintTOC.postingsPEFBitmap.sz != 0 {
		t.Fatalf("varint shard postingsPEFBitmap.sz = %d, want 0", varintTOC.postingsPEFBitmap.sz)
	}
	pefData := pefSearcher.(*indexData)
	wantVersion := 13
	if uniform {
		wantVersion = 14
	}
	if pefData.metaData.IndexMinReaderVersion != wantVersion || pefData.metaData.IndexFeatureVersion != wantVersion {
		t.Fatalf("PEF shard reader/feature versions = %d/%d, want %d", pefData.metaData.IndexMinReaderVersion, pefData.metaData.IndexFeatureVersion, wantVersion)
	}
	if (pefTOC.postingsPEFBitmap.sz == 0) != uniform {
		t.Fatalf("legacy PEF bitmap size = %d with uniform=%v", pefTOC.postingsPEFBitmap.sz, uniform)
	}
	if varintSearcher.(*indexData).metaData.IndexMinReaderVersion != WriteMinFeatureVersion {
		t.Fatal("disabled PEF changed the minimum reader version")
	}
	// Verify namePostings stay varint and content has both PEF and varint trigrams.
	_, peIdx := pefData.contentNgrams.GetWithIndex(stringToNGram("pha"))
	if !pefData.isPEFPosting(peIdx) {
		t.Fatalf("expected frequent trigram 'pha' to be PEF-encoded")
	}
	_, rareIdx := pefData.contentNgrams.GetWithIndex(stringToNGram("XYZ"))
	if pefData.isPEFPosting(rareIdx) {
		t.Fatalf("expected rare trigram 'XYZ' to remain varint")
	}

	queries := []struct {
		name string
		q    query.Q
	}{
		{
			name: "case_sensitive_single_pef_trigram",
			q:    &query.Substring{Pattern: "pha", Content: true, CaseSensitive: true},
		},
		{
			name: "case_sensitive_substring_pef",
			q:    &query.Substring{Pattern: "commonTokenAlpha", Content: true, CaseSensitive: true},
		},
		{
			name: "case_sensitive_rare_substring",
			q:    &query.Substring{Pattern: "rareNeedleXYZ123", Content: true, CaseSensitive: true},
		},
		{
			name: "case_insensitive_merging_mixed_pef_and_varint",
			q:    &query.Substring{Pattern: "mixedcaseword", Content: true, CaseSensitive: false},
		},
		{
			name: "distance_hit_rare_plus_frequent",
			q:    &query.Substring{Pattern: "rarePrefixcommonTokenAlpha", Content: true, CaseSensitive: true},
		},
		{
			name: "boolean_and",
			q: query.NewAnd(
				&query.Substring{Pattern: "commonTokenAlpha", Content: true, CaseSensitive: true},
				&query.Substring{Pattern: "rareNeedleXYZ123", Content: true, CaseSensitive: true},
			),
		},
		{
			name: "boolean_or",
			q: query.NewOr(
				&query.Substring{Pattern: "rareNeedleXYZ123", Content: true, CaseSensitive: true},
				&query.Substring{Pattern: "COMMONTOKENBETA", Content: true, CaseSensitive: true},
			),
		},
		{
			name: "boolean_and_not",
			q: query.NewAnd(
				&query.Substring{Pattern: "commonTokenAlpha", Content: true, CaseSensitive: true},
				&query.Not{Child: &query.Substring{Pattern: "rareNeedleXYZ123", Content: true, CaseSensitive: true}},
			),
		},
		{
			name: "regexp_query",
			q:    &query.Regexp{Regexp: mustParseRE(`handleItem[0-9]+_11\(\)`), Content: true, CaseSensitive: true},
		},
		{
			name: "filename_query",
			q:    &query.Substring{Pattern: "file_14_alpha", FileName: true, CaseSensitive: true},
		},
	}

	ctx := context.Background()
	sOpts := &zoekt.SearchOptions{}
	for _, tc := range queries {
		t.Run(tc.name, func(t *testing.T) {
			wantRes, err := varintSearcher.Search(ctx, tc.q, sOpts)
			if err != nil {
				t.Fatalf("varint search error: %v", err)
			}
			gotRes, err := pefSearcher.Search(ctx, tc.q, sOpts)
			if err != nil {
				t.Fatalf("PEF search error: %v", err)
			}
			if len(wantRes.Files) == 0 {
				t.Fatalf("expected non-empty matches for query %q", tc.name)
			}
			if diff := cmp.Diff(wantRes.Files, gotRes.Files); diff != "" {
				t.Fatalf("SearchResult.Files mismatch (-varint +pef):\n%s", diff)
			}
			if tc.name == "case_sensitive_single_pef_trigram" {
				if gotRes.Stats.NgramLookups == 0 || gotRes.Stats.IndexBytesLoaded == 0 {
					t.Fatalf("expected non-zero stats on single-variant PEF trigram, got %+v", gotRes.Stats)
				}
			}
		})
	}
}

func TestPEFSmallRepoZeroOverhead(t *testing.T) {
	t.Setenv("ZOEKT_PEF_MIN_POSTINGS", "")
	t.Setenv("ZOEKT_PEF_UNIFORM_BLOCK_SIZE", "128")
	docs := []Document{
		{Name: "a.go", Content: []byte("package main\nfunc main() { println(\"hello world\") }\n")},
		{Name: "b.go", Content: []byte("package main\nfunc helper() int { return 42 }\n")},
	}
	fixedTime := time.Unix(1700000000, 0).UTC()

	varintBytes, varintPath := buildTestShardBytes(t, "small-repo", docs, -1, fixedTime, "small000000000000001")
	defaultBytes, defaultPath := buildTestShardBytes(t, "small-repo", docs, 0, fixedTime, "small000000000000001")

	_, defaultTOC := openTestSearcher(t, defaultPath)
	defaultSearcher, _ := openTestSearcher(t, defaultPath)
	_, _ = openTestSearcher(t, varintPath)

	if defaultTOC.postingsPEFBitmap.sz != 0 {
		t.Fatalf("postingsPEFBitmap.sz = %d, want 0", defaultTOC.postingsPEFBitmap.sz)
	}
	d := defaultSearcher.(*indexData)
	if d.metaData.IndexMinReaderVersion != WriteMinFeatureVersion {
		t.Fatalf("IndexMinReaderVersion = %d, want %d", d.metaData.IndexMinReaderVersion, WriteMinFeatureVersion)
	}
	if len(d.postingsPEFBitmap) != 0 {
		t.Fatalf("len(d.postingsPEFBitmap) = %d, want 0", len(d.postingsPEFBitmap))
	}
	if !bytes.Equal(defaultBytes, varintBytes) {
		t.Fatalf("small repo shard differs from varint baseline: len(default)=%d, len(varint)=%d", len(defaultBytes), len(varintBytes))
	}
}

func TestPEFUniformShardMetadata(t *testing.T) {
	docs := buildTestCorpusDocs()
	fixedTime := time.Unix(1700000000, 0).UTC()
	const id = "uniform0000000000001"

	t.Setenv("ZOEKT_PEF_UNIFORM_BLOCK_SIZE", "")
	opt, _ := buildTestShardBytes(t, "uniform", docs, 128, fixedTime, id)
	t.Setenv("ZOEKT_PEF_UNIFORM_BLOCK_SIZE", "not-a-size")
	fallback, _ := buildTestShardBytes(t, "uniform", docs, 128, fixedTime, id)
	if !bytes.Equal(opt, fallback) {
		t.Fatal("invalid block size changed the default format")
	}
	t.Setenv("ZOEKT_PEF_UNIFORM_BLOCK_SIZE", "128")
	raw, path := buildTestShardBytes(t, "uniform", docs, 128, fixedTime, id)
	t.Setenv("ZOEKT_PEF_UNIFORM_BLOCK_SIZE", "")
	s, toc := openTestSearcher(t, path)
	if toc.postingsPEFUniform.sz == 0 || s.(*indexData).pefUniformBlockShift != 7 {
		t.Fatal("reader did not use the shard's Uniform metadata")
	}
	for _, tc := range []struct {
		name string
		edit func([]byte)
	}{
		{"unsupported_block_size", func(blob []byte) { blob[toc.postingsPEFUniform.off] = 6 }},
		{"legacy_reader_version", func(blob []byte) {
			metadata := blob[toc.metaData.off : toc.metaData.off+toc.metaData.sz]
			copy(metadata, bytes.Replace(metadata, []byte(`"IndexMinReaderVersion":14`), []byte(`"IndexMinReaderVersion":13`), 1))
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := bytes.Clone(raw)
			tc.edit(bad)
			if _, err := NewSearcher(&memSeeker{bad}); err == nil || !strings.Contains(err.Error(), "Uniform") {
				t.Fatalf("expected Uniform metadata error, got %v", err)
			}
		})
	}
}

func TestPEFUniformIndexStateAcceptsExistingShards(t *testing.T) {
	docs := buildTestCorpusDocs()
	for _, tc := range []struct {
		name, writtenWith, checkedWith string
	}{
		{"keep_opt", "", "128"},
		{"keep_uniform", "128", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("ZOEKT_PEF_UNIFORM_BLOCK_SIZE", tc.writtenWith)
			_, path := buildTestShardBytes(t, "uniform", docs, 128, time.Unix(1700000000, 0), "uniform0000000000001")
			t.Setenv("ZOEKT_PEF_UNIFORM_BLOCK_SIZE", tc.checkedWith)
			opts := Options{
				IndexDir:       filepath.Dir(path),
				DisableCTags:   true,
				Parallelism:    1,
				PEFMinPostings: 128,
				RepositoryDescription: zoekt.Repository{
					Name:     "uniform",
					Branches: []zoekt.RepositoryBranch{{Name: "main", Version: "v1"}},
				},
			}
			opts.SetDefaults()
			if state, _ := opts.IndexState(); state != IndexStateEqual {
				t.Fatalf("IndexState = %v, want %v", state, IndexStateEqual)
			}
		})
	}
}

func TestPEFMergeWithLegacyShard(t *testing.T) {
	t.Setenv("ZOEKT_PEF_UNIFORM_BLOCK_SIZE", "")
	fixedTime := time.Unix(1700000000, 0).UTC()
	docs1 := buildTestCorpusDocs()
	docs2 := []Document{
		{
			Name:     "legacy.go",
			Content:  []byte("package legacy\nfunc legacyEntry() string { return \"commonTokenAlpha rareNeedleXYZ123\" }\n"),
			Branches: []string{"main"},
		},
	}

	_, pefShardPath := buildTestShardBytes(t, "repo-pef", docs1, 128, fixedTime, "merge000000000000001")
	_, legacyShardPath := buildTestShardBytes(t, "repo-legacy", docs2, -1, fixedTime, "merge000000000000002")

	openInf := func(p string) IndexFile {
		f, err := os.Open(p)
		if err != nil {
			t.Fatalf("Open(%s): %v", p, err)
		}
		inf, err := NewIndexFile(f)
		if err != nil {
			f.Close()
			t.Fatalf("NewIndexFile(%s): %v", p, err)
		}
		t.Cleanup(inf.Close)
		return inf
	}

	t.Setenv("ZOEKT_PEF_MIN_POSTINGS", "128")
	t.Setenv("ZOEKT_PEF_UNIFORM_BLOCK_SIZE", "256")
	mergeDir := t.TempDir()
	tmpName, dstName, err := Merge(mergeDir, openInf(pefShardPath), openInf(legacyShardPath))
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if err := os.Rename(tmpName, dstName); err != nil {
		t.Fatalf("Rename: %v", err)
	}

	mergedSearcher, mergedTOC := openTestSearcher(t, dstName)
	if mergedTOC.postingsPEFBitmap.sz != 0 || mergedSearcher.(*indexData).metaData.IndexMinReaderVersion != 14 {
		t.Fatal("merged shard was not written in the Uniform format")
	}

	pefSearcher, _ := openTestSearcher(t, pefShardPath)
	legacySearcher, _ := openTestSearcher(t, legacyShardPath)

	ctx := context.Background()
	for _, q := range []query.Q{
		&query.Substring{Pattern: "commonTokenAlpha", Content: true, CaseSensitive: true},
		&query.Substring{Pattern: "rareNeedleXYZ123", Content: true, CaseSensitive: true},
		&query.Substring{Pattern: "legacyEntry", Content: true, CaseSensitive: true},
		&query.Substring{Pattern: "mixedcaseword", Content: true, CaseSensitive: false},
	} {
		mRes, err := mergedSearcher.Search(ctx, q, &zoekt.SearchOptions{})
		if err != nil {
			t.Fatalf("merged search %s: %v", q, err)
		}
		r1, err := pefSearcher.Search(ctx, q, &zoekt.SearchOptions{})
		if err != nil {
			t.Fatalf("pef search %s: %v", q, err)
		}
		r2, err := legacySearcher.Search(ctx, q, &zoekt.SearchOptions{})
		if err != nil {
			t.Fatalf("legacy search %s: %v", q, err)
		}
		wantCount := len(r1.Files) + len(r2.Files)
		if len(mRes.Files) != wantCount {
			t.Fatalf("query %s: merged files=%d, want %d (%d+%d)", q, len(mRes.Files), wantCount, len(r1.Files), len(r2.Files))
		}
	}

	// Exploding the merged shard should also preserve search results and respect ZOEKT_PEF_MIN_POSTINGS=0.
	t.Setenv("ZOEKT_PEF_MIN_POSTINGS", "0")
	explodeDir := t.TempDir()
	exploded, err := explode(explodeDir, openInf(dstName))
	if err != nil {
		t.Fatalf("explode: %v", err)
	}
	if len(exploded) != 2 {
		t.Fatalf("exploded %d shards, want 2", len(exploded))
	}
	var explodedMatches int
	explodeQuery := &query.Substring{Pattern: "commonTokenAlpha", Content: true, CaseSensitive: true}
	for tmp, final := range exploded {
		if err := os.Rename(tmp, final); err != nil {
			t.Fatalf("Rename(%s): %v", final, err)
		}
		expSearcher, expTOC := openTestSearcher(t, final)
		if expTOC.postingsPEFBitmap.sz != 0 || expSearcher.(*indexData).metaData.IndexMinReaderVersion != WriteMinFeatureVersion {
			t.Fatalf("exploded shard %s used PEF with ZOEKT_PEF_MIN_POSTINGS=0", filepath.Base(final))
		}
		res, err := expSearcher.Search(ctx, explodeQuery, &zoekt.SearchOptions{})
		if err != nil {
			t.Fatalf("exploded search: %v", err)
		}
		explodedMatches += len(res.Files)
	}
	mergedRes, err := mergedSearcher.Search(ctx, explodeQuery, &zoekt.SearchOptions{})
	if err != nil {
		t.Fatalf("merged search: %v", err)
	}
	if explodedMatches != len(mergedRes.Files) {
		t.Fatalf("exploded matches=%d, want %d", explodedMatches, len(mergedRes.Files))
	}
}

func TestPEFOldShardBackwardsCompatibility(t *testing.T) {
	// Load legacy shards committed before postingsPEFBitmap existed.
	for _, tc := range []struct {
		path    string
		pattern string
	}{
		{path: "../testdata/shards/ctagsrepo_v16.00000.zoekt", pattern: "func main"},
		{path: "../testdata/backcompat/static_toc_v16.00000.zoekt", pattern: "abcde"},
	} {
		t.Run(filepath.Base(tc.path), func(t *testing.T) {
			s, toc := openTestSearcher(t, tc.path)
			if toc.postingsPEFBitmap.sz != 0 {
				t.Fatalf("legacy shard has postingsPEFBitmap.sz=%d, want 0", toc.postingsPEFBitmap.sz)
			}
			d := s.(*indexData)
			if len(d.postingsPEFBitmap) != 0 {
				t.Fatalf("len(d.postingsPEFBitmap) = %d, want 0", len(d.postingsPEFBitmap))
			}
			if d.isPEFPosting(0) || d.isPEFPosting(100) || d.isPEFPosting(-1) {
				t.Fatalf("isPEFPosting returned true on legacy shard")
			}

			res, err := s.Search(context.Background(), &query.Substring{Pattern: tc.pattern, Content: true}, &zoekt.SearchOptions{})
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			if len(res.Files) == 0 {
				t.Fatalf("expected match for %q in %s", tc.pattern, tc.path)
			}
		})
	}

	// Verify reader rejects shards whose IndexMinReaderVersion > FeatureVersion.
	sb, err := NewShardBuilder(&zoekt.Repository{Name: "future"})
	if err != nil {
		t.Fatalf("NewShardBuilder: %v", err)
	}
	if err := sb.AddFile("a.txt", []byte("hello world")); err != nil {
		t.Fatalf("AddFile: %v", err)
	}
	var buf bytes.Buffer
	if err := sb.Write(&buf); err != nil {
		t.Fatalf("Write: %v", err)
	}
	// Corrupting IndexMinReaderVersion in-place is awkward due to TOC offsets; instead
	// verify via FeatureVersion constant check in readIndexData on synthetic metadata.
	f := &memSeeker{buf.Bytes()}
	rd := &reader{r: f}
	var toc indexTOC
	if err := rd.readTOC(&toc); err != nil {
		t.Fatalf("readTOC: %v", err)
	}
	mdSlice := bytes.Replace(buf.Bytes(),
		[]byte(fmt.Sprintf(`"IndexMinReaderVersion":%d`, WriteMinFeatureVersion)),
		[]byte(fmt.Sprintf(`"IndexMinReaderVersion":%d`, FeatureVersion+1)),
		1,
	)
	if bytes.Equal(mdSlice, buf.Bytes()) {
		t.Fatal("failed to patch IndexMinReaderVersion in test shard")
	}
	if len(mdSlice) != len(buf.Bytes()) {
		t.Fatalf("IndexMinReaderVersion patch changed byte length: %d vs %d", len(mdSlice), len(buf.Bytes()))
	}
	if _, err := NewSearcher(&memSeeker{mdSlice}); err == nil || !strings.Contains(err.Error(), "needs read feature version") {
		t.Fatalf("expected NewSearcher to reject IndexMinReaderVersion > FeatureVersion, got %v", err)
	}
}

func TestPEFConfigAndEnvOverrides(t *testing.T) {
	t.Run("MinPostings", func(t *testing.T) {
		t.Setenv("ZOEKT_PEF_MIN_POSTINGS", "")
		if got := resolvePEFMinPostings(0); got != defaultPEFMinPostings {
			t.Fatalf("unset env: got %d, want %d", got, defaultPEFMinPostings)
		}
		if got := resolvePEFMinPostings(-1); got != -1 {
			t.Fatalf("explicit disable: got %d, want -1", got)
		}
		if got := resolvePEFMinPostings(256); got != 256 {
			t.Fatalf("explicit value: got %d, want 256", got)
		}

		t.Setenv("ZOEKT_PEF_MIN_POSTINGS", "0")
		if got := resolvePEFMinPostings(0); got != -1 {
			t.Fatalf("env=0: got %d, want -1", got)
		}
		if got := resolvePEFMinPostings(128); got != 128 {
			t.Fatalf("explicit option overrides env=0: got %d, want 128", got)
		}

		t.Setenv("ZOEKT_PEF_MIN_POSTINGS", "64")
		if got := resolvePEFMinPostings(0); got != 64 {
			t.Fatalf("env=64: got %d, want 64", got)
		}
		if got := resolvePEFMinPostings(-1); got != -1 {
			t.Fatalf("explicit disable overrides env=64: got %d, want -1", got)
		}

		t.Setenv("ZOEKT_PEF_MIN_POSTINGS", "not-an-int")
		if got := resolvePEFMinPostings(0); got != defaultPEFMinPostings {
			t.Fatalf("invalid env: got %d, want %d", got, defaultPEFMinPostings)
		}
	})

	t.Run("PartitionConfigEnv", func(t *testing.T) {
		t.Run("ValidOverrides", func(t *testing.T) {
			t.Setenv("ZOEKT_PEF_FIX_COST", "160")
			t.Setenv("ZOEKT_PEF_EPS1", "0.1")
			t.Setenv("ZOEKT_PEF_EPS2", "0.5")
			got := resolvePEFPartitionConfig()
			if got.FixCost != 160 || got.Eps1 != 0.1 || got.Eps2 != 0.5 {
				t.Fatalf("got %+v, want FixCost=160 Eps1=0.1 Eps2=0.5", got)
			}
		})

		// Invalid values fallback cleanly to default without panicking.
		invalidCases := []struct {
			name string
			env  string
			val  string
		}{
			{"FixCostZero", "ZOEKT_PEF_FIX_COST", "0"},
			{"FixCostMaxUint64", "ZOEKT_PEF_FIX_COST", "18446744073709551615"},
			{"Eps1NaN", "ZOEKT_PEF_EPS1", "NaN"},
			{"Eps2NaN", "ZOEKT_PEF_EPS2", "NaN"},
			{"Eps2PlusInf", "ZOEKT_PEF_EPS2", "+Inf"},
		}
		for _, tc := range invalidCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Setenv("ZOEKT_PEF_FIX_COST", "")
				t.Setenv("ZOEKT_PEF_EPS1", "")
				t.Setenv("ZOEKT_PEF_EPS2", "")
				t.Setenv(tc.env, tc.val)
				got := resolvePEFPartitionConfig()
				if got != defaultPEFPartitionConfig() {
					t.Fatalf("invalid %s=%s: got %+v, want %+v", tc.env, tc.val, got, defaultPEFPartitionConfig())
				}
			})
		}
	})

	t.Run("EndToEndEnvDisable", func(t *testing.T) {
		t.Setenv("ZOEKT_PEF_MIN_POSTINGS", "0")
		docs := buildTestCorpusDocs()
		fixedTime := time.Unix(1700000000, 0).UTC()
		_, shardPath := buildTestShardBytes(t, "env-disabled", docs, 0, fixedTime, "env00000000000000001")
		_, toc := openTestSearcher(t, shardPath)
		if toc.postingsPEFBitmap.sz != 0 {
			t.Fatalf("postingsPEFBitmap.sz = %d with ZOEKT_PEF_MIN_POSTINGS=0, want 0", toc.postingsPEFBitmap.sz)
		}
	})
}
