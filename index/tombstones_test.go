package index

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/OgurtsovAndrei/zoekt"
	"github.com/OgurtsovAndrei/zoekt/query"
)

func TestSetTombstone(t *testing.T) {
	mockRepos = mkRepos("r1", "r2", "r3")

	readMeta := func(shard string) []byte {
		blob, err := os.ReadFile(shard + ".meta")
		if err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		return blob
	}

	dir := t.TempDir()
	ghostShard := filepath.Join(dir, "test.zoekt")

	isAlive := func(alive []bool) {
		t.Helper()
		blob := readMeta(ghostShard)
		ghostRepos := []*zoekt.Repository{}
		if err := json.Unmarshal(blob, &ghostRepos); err != nil {
			t.Fatal(err)
		}
		for i, repo := range ghostRepos {
			if repo.Tombstone == alive[i] {
				t.Fatalf("r%d: want %t, got %t\n", i+1, alive[i], repo.Tombstone)
			}
		}
	}

	if err := SetTombstone(ghostShard, 2); err != nil {
		t.Fatal(err)
	}
	isAlive([]bool{true, false, true})

	if err := SetTombstone(ghostShard, 1); err != nil {
		t.Fatal(err)
	}
	isAlive([]bool{false, false, true})

	if err := UnsetTombstone(ghostShard, 2); err != nil {
		t.Fatal(err)
	}
	isAlive([]bool{false, true, true})
}

func mkRepos(repoNames ...string) []*zoekt.Repository {
	ret := make([]*zoekt.Repository, 0, len(repoNames))
	for i, n := range repoNames {
		ret = append(ret, &zoekt.Repository{ID: uint32(i + 1), Name: n})
	}
	return ret
}

func TestInitTombstonedDocsReleasesMapAndFiltersSearch(t *testing.T) {
	dir := t.TempDir()
	repo := zoekt.Repository{
		Name:     "repo",
		ID:       1,
		Branches: []zoekt.RepositoryBranch{{Name: "main", Version: "v1"}},
	}

	opts := Options{
		IndexDir:              dir,
		RepositoryDescription: repo,
		DisableCTags:          true,
	}
	opts.SetDefaults()

	b, err := NewBuilder(opts)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	const totalDocs = 70
	for i := range totalDocs {
		name := fmt.Sprintf("file_%03d.go", i)
		if err := b.Add(Document{
			Name:     name,
			Content:  []byte(fmt.Sprintf("package p\nconst token_%03d = \"needle\"\n", i)),
			Branches: []string{"main"},
		}); err != nil {
			t.Fatalf("Add(%s): %v", name, err)
		}
	}
	if err := b.Finish(); err != nil {
		t.Fatalf("Finish: %v", err)
	}

	shards := opts.FindAllShards()
	if len(shards) != 1 {
		t.Fatalf("expected 1 shard, got %d", len(shards))
	}
	shardPath := shards[0]

	metaRepo := repo
	metaRepo.FileTombstones = map[string]struct{}{
		"file_000.go": {},
		"file_065.go": {},
	}
	for i := range 3000 {
		metaRepo.FileTombstones[fmt.Sprintf("unrelated_shard_file_with_long_prefix_%04d.go", i)] = struct{}{}
	}
	metaBytes, err := json.Marshal(&metaRepo)
	if err != nil {
		t.Fatalf("Marshal meta: %v", err)
	}
	if len(metaBytes) < minSharedMetaBytes {
		t.Fatalf("expected metaBytes >= %d, got %d", minSharedMetaBytes, len(metaBytes))
	}
	if err := os.WriteFile(shardPath+".meta", metaBytes, 0o600); err != nil {
		t.Fatalf("WriteFile meta: %v", err)
	}

	// Open two searchers sequentially against the same >=64 KiB .meta file so the
	// second NewSearcher exercises globalMetaDeduplicator cache reuse end-to-end.
	for attempt := range 2 {
		f, err := os.Open(shardPath)
		if err != nil {
			t.Fatalf("attempt %d: Open shard: %v", attempt, err)
		}

		inf, err := NewIndexFile(f)
		if err != nil {
			_ = f.Close()
			t.Fatalf("attempt %d: NewIndexFile: %v", attempt, err)
		}

		s, err := NewSearcher(inf)
		if err != nil {
			inf.Close()
			t.Fatalf("attempt %d: NewSearcher: %v", attempt, err)
		}

		d := s.(*indexData)
		if d.repoMetaData[0].FileTombstones != nil {
			s.Close()
			t.Fatalf("attempt %d: expected d.repoMetaData[0].FileTombstones to be nil after initTombstonedDocs, got len=%d", attempt, len(d.repoMetaData[0].FileTombstones))
		}
		if d.repoListEntry[0].Repository.FileTombstones != nil {
			s.Close()
			t.Fatalf("attempt %d: expected d.repoListEntry[0].Repository.FileTombstones to be nil, got len=%d", attempt, len(d.repoListEntry[0].Repository.FileTombstones))
		}
		if len(d.tombstonedDocs) != 2 {
			s.Close()
			t.Fatalf("attempt %d: expected tombstonedDocs bitset length 2 for %d docs, got %d", attempt, totalDocs, len(d.tombstonedDocs))
		}
		if d.isDocTombstoned(999) {
			s.Close()
			t.Fatalf("attempt %d: out-of-range doc should not be tombstoned", attempt)
		}

		sr, err := s.Search(context.Background(), &query.Substring{Pattern: "needle"}, &zoekt.SearchOptions{})
		if err != nil {
			s.Close()
			t.Fatalf("attempt %d: Search: %v", attempt, err)
		}
		if len(sr.Files) != totalDocs-2 {
			s.Close()
			t.Fatalf("attempt %d: expected %d matching files, got %d", attempt, totalDocs-2, len(sr.Files))
		}
		for _, fm := range sr.Files {
			if fm.FileName == "file_000.go" || fm.FileName == "file_065.go" {
				s.Close()
				t.Fatalf("attempt %d: tombstoned file %q returned in search results", attempt, fm.FileName)
			}
		}
		s.Close()
	}
}

func TestBuilderDeltaPrunesUnrelatedFileTombstones(t *testing.T) {
	dir := t.TempDir()
	repo := zoekt.Repository{
		Name:     "repo",
		ID:       1,
		Branches: []zoekt.RepositoryBranch{{Name: "main", Version: "v1"}},
	}

	opts := Options{
		IndexDir:              dir,
		RepositoryDescription: repo,
		DisableCTags:          true,
		ShardMax:              100,
	}
	opts.SetDefaults()
	opts.ShardMax = 100

	b, err := NewBuilder(opts)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	docs := []Document{
		{Name: "shard0_keep.go", Content: []byte(strings.Repeat("a", 60)), Branches: []string{"main"}},
		{Name: "shard0_drop.go", Content: []byte(strings.Repeat("b", 60)), Branches: []string{"main"}},
		{Name: "shard1_drop.go", Content: []byte(strings.Repeat("c", 60)), Branches: []string{"main"}},
	}
	for _, doc := range docs {
		if err := b.Add(doc); err != nil {
			t.Fatalf("Add(%s): %v", doc.Name, err)
		}
	}
	if err := b.Finish(); err != nil {
		t.Fatalf("Finish initial: %v", err)
	}

	oldShards := opts.FindAllShards()
	if len(oldShards) < 2 {
		t.Fatalf("expected at least 2 shards, got %d", len(oldShards))
	}

	// Pre-populate shard 0 metadata with stale unrelated tombstones to verify compaction on update.
	repos0, _, err := ReadMetadataPathAlive(oldShards[0])
	if err != nil {
		t.Fatalf("ReadMetadataPathAlive: %v", err)
	}
	repos0[0].FileTombstones = map[string]struct{}{
		"stale_foreign_file.go": {},
	}
	tmp, final, err := JsonMarshalRepoMetaTemp(oldShards[0], repos0[0])
	if err != nil {
		t.Fatalf("JsonMarshalRepoMetaTemp: %v", err)
	}
	if err := os.Rename(tmp, final); err != nil {
		t.Fatalf("Rename meta: %v", err)
	}

	deltaOpts := opts
	deltaOpts.IsDelta = true
	deltaOpts.RepositoryDescription.Branches = []zoekt.RepositoryBranch{{Name: "main", Version: "v2"}}

	db, err := NewBuilder(deltaOpts)
	if err != nil {
		t.Fatalf("NewBuilder delta: %v", err)
	}
	db.MarkFileAsChangedOrRemoved("shard0_drop.go")
	db.MarkFileAsChangedOrRemoved("shard1_drop.go")
	db.MarkFileAsChangedOrRemoved("deleted_never_indexed.go")
	if err := db.Add(Document{
		Name:     "shard0_drop.go",
		Content:  []byte("updated content"),
		Branches: []string{"main"},
	}); err != nil {
		t.Fatalf("Add delta doc: %v", err)
	}
	if err := db.Finish(); err != nil {
		t.Fatalf("Finish delta: %v", err)
	}

	wantPerShard := []map[string]struct{}{
		{"shard0_drop.go": {}},
		{"shard1_drop.go": {}},
	}
	if len(oldShards) != len(wantPerShard) {
		t.Fatalf("expected %d old shards, got %d", len(wantPerShard), len(oldShards))
	}
	for i, shard := range oldShards {
		repos, _, err := ReadMetadataPathAlive(shard)
		if err != nil {
			t.Fatalf("ReadMetadataPathAlive(%s): %v", shard, err)
		}
		got := repos[0].FileTombstones
		if !reflect.DeepEqual(got, wantPerShard[i]) {
			t.Fatalf("shard %s FileTombstones = %v, want %v", shard, got, wantPerShard[i])
		}
	}
}

func TestMergeAndExplodePreserveFileTombstones(t *testing.T) {
	dir := t.TempDir()
	repo := zoekt.Repository{
		Name:     "repo-merge",
		ID:       7,
		Branches: []zoekt.RepositoryBranch{{Name: "main", Version: "v1"}},
	}
	opts := Options{
		IndexDir:              dir,
		RepositoryDescription: repo,
		DisableCTags:          true,
	}
	opts.SetDefaults()

	b, err := NewBuilder(opts)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for _, name := range []string{"alive.go", "tombstoned.go"} {
		if err := b.Add(Document{
			Name:     name,
			Content:  []byte("package p\nconst sharedNeedle = 1\n"),
			Branches: []string{"main"},
		}); err != nil {
			t.Fatalf("Add(%s): %v", name, err)
		}
	}
	if err := b.Finish(); err != nil {
		t.Fatalf("Finish: %v", err)
	}

	shards := opts.FindAllShards()
	if len(shards) != 1 {
		t.Fatalf("expected 1 shard, got %d", len(shards))
	}
	shardPath := shards[0]

	repos, _, err := ReadMetadataPathAlive(shardPath)
	if err != nil {
		t.Fatalf("ReadMetadataPathAlive: %v", err)
	}
	repos[0].FileTombstones = map[string]struct{}{"tombstoned.go": {}}
	tmp, final, err := JsonMarshalRepoMetaTemp(shardPath, repos[0])
	if err != nil {
		t.Fatalf("JsonMarshalRepoMetaTemp: %v", err)
	}
	if err := os.Rename(tmp, final); err != nil {
		t.Fatalf("Rename meta: %v", err)
	}

	f, err := os.Open(shardPath)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	inf, err := NewIndexFile(f)
	if err != nil {
		_ = f.Close()
		t.Fatalf("NewIndexFile: %v", err)
	}
	defer inf.Close()

	mergeDir := t.TempDir()
	tmpCompound, dstCompound, err := Merge(mergeDir, inf)
	if err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if err := os.Rename(tmpCompound, dstCompound); err != nil {
		t.Fatalf("Rename compound: %v", err)
	}

	explodeDir := t.TempDir()
	if err := Explode(explodeDir, dstCompound); err != nil {
		t.Fatalf("Explode: %v", err)
	}

	exploded, err := filepath.Glob(filepath.Join(explodeDir, "*.zoekt"))
	if err != nil || len(exploded) != 1 {
		t.Fatalf("expected 1 exploded shard, got %v (err=%v)", exploded, err)
	}

	ef, err := os.Open(exploded[0])
	if err != nil {
		t.Fatalf("Open exploded: %v", err)
	}
	einf, err := NewIndexFile(ef)
	if err != nil {
		_ = ef.Close()
		t.Fatalf("NewIndexFile exploded: %v", err)
	}
	es, err := NewSearcher(einf)
	if err != nil {
		einf.Close()
		t.Fatalf("NewSearcher exploded: %v", err)
	}
	defer es.Close()

	sr, err := es.Search(context.Background(), &query.Substring{Pattern: "sharedNeedle"}, &zoekt.SearchOptions{})
	if err != nil {
		t.Fatalf("Search exploded: %v", err)
	}
	if len(sr.Files) != 1 || sr.Files[0].FileName != "alive.go" {
		t.Fatalf("expected only alive.go after Merge+Explode, got %+v", sr.Files)
	}
}

func TestMetaBlobDeduplicatorConcurrentAndEviction(t *testing.T) {
	var lastIdleFire func()
	dedup := &metaBlobDeduplicator{
		inflight: make(map[sharedMetaKey]*sharedMetaCall),
		onTimerScheduled: func(fire func()) {
			lastIdleFire = fire
		},
	}

	makeLargeBlob := func(repoName string, count int) []byte {
		r := zoekt.Repository{
			Name:           repoName,
			ID:             1,
			FileTombstones: make(map[string]struct{}, count),
		}
		for i := range count {
			r.FileTombstones[fmt.Sprintf("dir/subdir/file_number_%06d.go", i)] = struct{}{}
		}
		b, err := json.Marshal(&r)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		if len(b) < minSharedMetaBytes {
			t.Fatalf("expected blob >= %d bytes, got %d", minSharedMetaBytes, len(b))
		}
		return b
	}

	blob := makeLargeBlob("big-repo", 2500)

	const workers = 16
	var wg sync.WaitGroup
	wg.Add(workers)
	errs := make(chan error, workers)
	for range workers {
		go func() {
			defer wg.Done()
			repos, err := dedup.unmarshalShared(blob, false)
			if err != nil {
				errs <- err
				return
			}
			if len(repos) != 1 || repos[0].Name != "big-repo" || len(repos[0].FileTombstones) != 2500 {
				errs <- fmt.Errorf("unexpected repos: %+v", repos)
				return
			}
			// Clearing FileTombstones on the returned struct copy must not mutate the cached entry.
			repos[0].FileTombstones = nil
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("worker error: %v", err)
	}

	// Verify cached entry still has all 2500 tombstones after callers cleared their struct copies.
	repos, err := dedup.unmarshalShared(blob, false)
	if err != nil {
		t.Fatalf("unmarshalShared cached: %v", err)
	}
	if len(repos[0].FileTombstones) != 2500 {
		t.Fatalf("expected 2500 cached tombstones, got %d", len(repos[0].FileTombstones))
	}

	// Eviction beyond maxSharedMetaEntries and LRU promotion.
	blobs := make([][]byte, maxSharedMetaEntries+2)
	for i := range maxSharedMetaEntries + 2 {
		blobs[i] = makeLargeBlob(fmt.Sprintf("repo-%d", i), 2500)
		if _, err := dedup.unmarshalShared(blobs[i], false); err != nil {
			t.Fatalf("unmarshalShared(%d): %v", i, err)
		}
	}
	staleFire := lastIdleFire
	// Access a non-head cached entry to exercise LRU move-to-front.
	if _, err := dedup.unmarshalShared(blobs[maxSharedMetaEntries], false); err != nil {
		t.Fatalf("unmarshalShared LRU hit: %v", err)
	}
	dedup.mu.Lock()
	entryCount := len(dedup.entries)
	headName := ""
	if entryCount > 0 && len(dedup.entries[0].repos) > 0 {
		headName = dedup.entries[0].repos[0].Name
	}
	dedup.mu.Unlock()
	if entryCount > maxSharedMetaEntries {
		t.Fatalf("expected at most %d cached entries, got %d", maxSharedMetaEntries, entryCount)
	}
	wantHead := fmt.Sprintf("repo-%d", maxSharedMetaEntries)
	if headName != wantHead {
		t.Fatalf("expected LRU promoted head entry %q, got %q", wantHead, headName)
	}

	// A stale timer callback from an earlier generation must not clear entries.
	if staleFire != nil {
		staleFire()
	}
	dedup.mu.Lock()
	afterStale := len(dedup.entries)
	dedup.mu.Unlock()
	if afterStale == 0 {
		t.Fatalf("stale timer callback unexpectedly cleared active cache entries")
	}

	// Invalid large JSON must return an error without caching.
	invalidLarge := []byte(strings.Repeat("{", minSharedMetaBytes+10))
	if _, err := dedup.unmarshalShared(invalidLarge, false); err == nil {
		t.Fatalf("expected error on invalid large JSON")
	}

	// Fire the latest scheduled idle callback deterministically and verify entries are cleared.
	if lastIdleFire == nil {
		t.Fatalf("expected idle callback to be scheduled")
	}
	lastIdleFire()
	dedup.mu.Lock()
	afterIdle := len(dedup.entries)
	dedup.mu.Unlock()
	if afterIdle != 0 {
		t.Fatalf("expected 0 entries after idle callback, got %d", afterIdle)
	}
}
