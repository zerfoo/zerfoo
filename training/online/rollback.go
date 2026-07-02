package online

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// RollbackConfig holds parameters for the rollback manager.
type RollbackConfig struct {
	// MaxVersions is the maximum number of snapshots to retain.
	MaxVersions int
	// StoragePath is the directory where snapshot files are stored.
	StoragePath string
}

// snapshotEntry tracks a snapshot's ID and creation order.
type snapshotEntry struct {
	ID  string
	Seq int64
}

// RollbackManager manages versioned model weight snapshots on disk,
// supporting snapshot creation, rollback, listing, and pruning.
type RollbackManager struct {
	cfg     RollbackConfig
	mu      sync.Mutex
	entries []snapshotEntry
	nextSeq int64
}

// NewRollbackManager creates a RollbackManager, creating StoragePath if it
// does not exist. It scans the directory for existing snapshots so that
// persistence across restarts is maintained.
func NewRollbackManager(cfg RollbackConfig) (*RollbackManager, error) {
	if err := os.MkdirAll(cfg.StoragePath, 0o755); err != nil {
		return nil, fmt.Errorf("rollback: create storage path: %w", err)
	}

	m := &RollbackManager{cfg: cfg}

	// Scan for existing snapshots to restore state.
	matches, err := filepath.Glob(filepath.Join(cfg.StoragePath, "*.gob"))
	if err != nil {
		return nil, fmt.Errorf("rollback: scan snapshots: %w", err)
	}

	type fileInfo struct {
		id      string
		modTime int64
	}
	var files []fileInfo
	for _, p := range matches {
		base := filepath.Base(p)
		id := base[:len(base)-len(".gob")]
		info, err := os.Stat(p)
		if err != nil {
			continue
		}
		files = append(files, fileInfo{id: id, modTime: info.ModTime().UnixNano()})
	}

	// Sort by modification time ascending (oldest first).
	sort.Slice(files, func(i, j int) bool {
		return files[i].modTime < files[j].modTime
	})

	for i, f := range files {
		m.entries = append(m.entries, snapshotEntry{ID: f.id, Seq: int64(i)})
	}
	m.nextSeq = int64(len(files))

	return m, nil
}

// Snapshot serializes weights to a gob file and evicts the oldest snapshot
// if the number of snapshots exceeds MaxVersions.
func (m *RollbackManager) Snapshot(id string, weights map[string][]float32) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	path := filepath.Join(m.cfg.StoragePath, id+".gob")
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("rollback: create snapshot file: %w", err)
	}
	if err := gob.NewEncoder(f).Encode(weights); err != nil {
		f.Close()
		return fmt.Errorf("rollback: encode snapshot: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("rollback: close snapshot file: %w", err)
	}

	// Remove any existing entry with the same ID before appending.
	entries := m.entries[:0]
	for _, e := range m.entries {
		if e.ID != id {
			entries = append(entries, e)
		}
	}
	m.entries = append(entries, snapshotEntry{ID: id, Seq: m.nextSeq})
	m.nextSeq++

	// Evict oldest if over limit.
	if m.cfg.MaxVersions > 0 && len(m.entries) > m.cfg.MaxVersions {
		m.evictOldest(len(m.entries) - m.cfg.MaxVersions)
	}

	return nil
}

// Rollback deserializes and returns the weights for the given snapshot ID.
func (m *RollbackManager) Rollback(id string) (map[string][]float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	path := filepath.Join(m.cfg.StoragePath, id+".gob")
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("rollback: open snapshot: %w", err)
	}
	defer f.Close()

	var weights map[string][]float32
	if err := gob.NewDecoder(f).Decode(&weights); err != nil {
		return nil, fmt.Errorf("rollback: decode snapshot: %w", err)
	}
	return weights, nil
}

// ListSnapshots returns snapshot IDs sorted by creation time, newest first.
func (m *RollbackManager) ListSnapshots() []string {
	m.mu.Lock()
	defer m.mu.Unlock()

	ids := make([]string, len(m.entries))
	for i, e := range m.entries {
		ids[len(m.entries)-1-i] = e.ID
	}
	return ids
}

// Prune deletes the oldest snapshots beyond MaxVersions.
func (m *RollbackManager) Prune() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.cfg.MaxVersions <= 0 || len(m.entries) <= m.cfg.MaxVersions {
		return nil
	}
	return m.evictOldest(len(m.entries) - m.cfg.MaxVersions)
}

// Close is a no-op that satisfies resource cleanup conventions.
func (m *RollbackManager) Close() error {
	return nil
}

// evictOldest removes the n oldest entries from disk and memory.
// Must be called with m.mu held.
func (m *RollbackManager) evictOldest(n int) error {
	if n <= 0 {
		return nil
	}
	if n > len(m.entries) {
		n = len(m.entries)
	}
	for _, e := range m.entries[:n] {
		path := filepath.Join(m.cfg.StoragePath, e.ID+".gob")
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("rollback: evict snapshot %s: %w", e.ID, err)
		}
	}
	m.entries = m.entries[n:]
	return nil
}
