package online

import (
	"fmt"
	"testing"
)

func TestSnapshot(t *testing.T) {
	dir := t.TempDir()
	m, err := NewRollbackManager(RollbackConfig{
		MaxVersions: 10,
		StoragePath: dir,
	})
	if err != nil {
		t.Fatalf("NewRollbackManager: %v", err)
	}

	for i := range 3 {
		id := fmt.Sprintf("v%d", i)
		weights := map[string][]float32{"layer0": {float32(i)}}
		if err := m.Snapshot(id, weights); err != nil {
			t.Fatalf("Snapshot(%s): %v", id, err)
		}
	}

	ids := m.ListSnapshots()
	if len(ids) != 3 {
		t.Fatalf("expected 3 snapshots, got %d", len(ids))
	}
	// Newest first.
	if ids[0] != "v2" || ids[1] != "v1" || ids[2] != "v0" {
		t.Fatalf("unexpected order: %v", ids)
	}
}

func TestRollbackWeights(t *testing.T) {
	dir := t.TempDir()
	m, err := NewRollbackManager(RollbackConfig{
		MaxVersions: 10,
		StoragePath: dir,
	})
	if err != nil {
		t.Fatalf("NewRollbackManager: %v", err)
	}

	want := map[string][]float32{
		"layer0": {1.0, 2.0, 3.0},
		"layer1": {4.0, 5.0},
	}
	if err := m.Snapshot("snap1", want); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	got, err := m.Rollback("snap1")
	if err != nil {
		t.Fatalf("Rollback: %v", err)
	}

	for name, wantW := range want {
		gotW, ok := got[name]
		if !ok {
			t.Fatalf("missing layer %s", name)
		}
		if len(gotW) != len(wantW) {
			t.Fatalf("layer %s: length mismatch %d vs %d", name, len(gotW), len(wantW))
		}
		for i := range wantW {
			if gotW[i] != wantW[i] {
				t.Fatalf("layer %s[%d]: got %f, want %f", name, i, gotW[i], wantW[i])
			}
		}
	}
}

func TestPrune(t *testing.T) {
	dir := t.TempDir()
	maxVersions := 3
	m, err := NewRollbackManager(RollbackConfig{
		MaxVersions: maxVersions,
		StoragePath: dir,
	})
	if err != nil {
		t.Fatalf("NewRollbackManager: %v", err)
	}

	// Write MaxVersions + 2 snapshots without auto-eviction by using a
	// high initial MaxVersions, then reset and prune.
	m.cfg.MaxVersions = maxVersions + 10 // prevent auto-eviction
	for i := range maxVersions + 2 {
		id := fmt.Sprintf("v%d", i)
		if err := m.Snapshot(id, map[string][]float32{"l": {float32(i)}}); err != nil {
			t.Fatalf("Snapshot(%s): %v", id, err)
		}
	}

	if len(m.ListSnapshots()) != maxVersions+2 {
		t.Fatalf("expected %d snapshots before prune, got %d", maxVersions+2, len(m.ListSnapshots()))
	}

	m.cfg.MaxVersions = maxVersions
	if err := m.Prune(); err != nil {
		t.Fatalf("Prune: %v", err)
	}

	ids := m.ListSnapshots()
	if len(ids) != maxVersions {
		t.Fatalf("expected %d snapshots after prune, got %d", maxVersions, len(ids))
	}

	// Oldest two (v0, v1) should be gone; newest three remain.
	for _, id := range ids {
		if id == "v0" || id == "v1" {
			t.Fatalf("snapshot %s should have been pruned", id)
		}
	}
}

func TestPersistence(t *testing.T) {
	dir := t.TempDir()
	cfg := RollbackConfig{MaxVersions: 10, StoragePath: dir}

	m1, err := NewRollbackManager(cfg)
	if err != nil {
		t.Fatalf("NewRollbackManager (1st): %v", err)
	}

	want := map[string][]float32{"layer0": {7.0, 8.0, 9.0}}
	if err := m1.Snapshot("persist1", want); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	if err := m1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Re-open with a new manager pointing to the same directory.
	m2, err := NewRollbackManager(cfg)
	if err != nil {
		t.Fatalf("NewRollbackManager (2nd): %v", err)
	}

	ids := m2.ListSnapshots()
	if len(ids) != 1 || ids[0] != "persist1" {
		t.Fatalf("expected [persist1], got %v", ids)
	}

	got, err := m2.Rollback("persist1")
	if err != nil {
		t.Fatalf("Rollback after reopen: %v", err)
	}
	for i, v := range want["layer0"] {
		if got["layer0"][i] != v {
			t.Fatalf("weight mismatch at %d: got %f, want %f", i, got["layer0"][i], v)
		}
	}
}

func TestSnapshotAutoEviction(t *testing.T) {
	dir := t.TempDir()
	m, err := NewRollbackManager(RollbackConfig{
		MaxVersions: 2,
		StoragePath: dir,
	})
	if err != nil {
		t.Fatalf("NewRollbackManager: %v", err)
	}

	for i := range 4 {
		id := fmt.Sprintf("v%d", i)
		if err := m.Snapshot(id, map[string][]float32{"l": {float32(i)}}); err != nil {
			t.Fatalf("Snapshot(%s): %v", id, err)
		}
	}

	ids := m.ListSnapshots()
	if len(ids) != 2 {
		t.Fatalf("expected 2 snapshots after auto-eviction, got %d: %v", len(ids), ids)
	}
	if ids[0] != "v3" || ids[1] != "v2" {
		t.Fatalf("expected [v3 v2], got %v", ids)
	}

	// Evicted snapshots should not be rollback-able.
	_, err = m.Rollback("v0")
	if err == nil {
		t.Fatal("expected error rolling back evicted snapshot v0")
	}
}

func TestRollbackNotFound(t *testing.T) {
	dir := t.TempDir()
	m, err := NewRollbackManager(RollbackConfig{
		MaxVersions: 5,
		StoragePath: dir,
	})
	if err != nil {
		t.Fatalf("NewRollbackManager: %v", err)
	}

	_, err = m.Rollback("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent snapshot")
	}
}

func TestPruneNoop(t *testing.T) {
	dir := t.TempDir()
	m, err := NewRollbackManager(RollbackConfig{
		MaxVersions: 5,
		StoragePath: dir,
	})
	if err != nil {
		t.Fatalf("NewRollbackManager: %v", err)
	}

	if err := m.Snapshot("v0", map[string][]float32{"l": {1.0}}); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// Prune should be a no-op when under MaxVersions.
	if err := m.Prune(); err != nil {
		t.Fatalf("Prune: %v", err)
	}
	if len(m.ListSnapshots()) != 1 {
		t.Fatalf("expected 1 snapshot, got %d", len(m.ListSnapshots()))
	}
}

func TestSnapshotOverwrite(t *testing.T) {
	dir := t.TempDir()
	m, err := NewRollbackManager(RollbackConfig{
		MaxVersions: 5,
		StoragePath: dir,
	})
	if err != nil {
		t.Fatalf("NewRollbackManager: %v", err)
	}

	if err := m.Snapshot("v0", map[string][]float32{"l": {1.0}}); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	// Overwrite with new weights.
	if err := m.Snapshot("v0", map[string][]float32{"l": {2.0}}); err != nil {
		t.Fatalf("Snapshot overwrite: %v", err)
	}

	ids := m.ListSnapshots()
	if len(ids) != 1 {
		t.Fatalf("expected 1 snapshot after overwrite, got %d", len(ids))
	}

	got, err := m.Rollback("v0")
	if err != nil {
		t.Fatalf("Rollback: %v", err)
	}
	if got["l"][0] != 2.0 {
		t.Fatalf("expected overwritten weight 2.0, got %f", got["l"][0])
	}
}
