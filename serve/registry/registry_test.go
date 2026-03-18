package registry

import (
	"path/filepath"
	"testing"
	"time"
)

func newTestRegistry(t *testing.T) *Registry {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "registry.db")
	r, err := NewRegistry(dbPath)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	t.Cleanup(func() { r.Close() })
	return r
}

func TestRegisterAndActivate(t *testing.T) {
	r := newTestRegistry(t)

	v1 := ModelVersion{
		ID:        "gemma-v1",
		Name:      "gemma",
		Version:   "1.0",
		Path:      "/models/gemma-v1.gguf",
		Format:    "gguf",
		CreatedAt: time.Now().Truncate(time.Second),
		Metrics:   map[string]float64{"tok_s": 245},
		Active:    false,
	}
	v2 := ModelVersion{
		ID:        "gemma-v2",
		Name:      "gemma",
		Version:   "2.0",
		Path:      "/models/gemma-v2.gguf",
		Format:    "gguf",
		CreatedAt: time.Now().Truncate(time.Second),
		Active:    false,
	}

	if err := r.Register(v1); err != nil {
		t.Fatalf("Register v1: %v", err)
	}
	if err := r.Register(v2); err != nil {
		t.Fatalf("Register v2: %v", err)
	}

	// Duplicate register must fail.
	if err := r.Register(v1); err != errAlreadyExist {
		t.Fatalf("duplicate Register: got %v, want %v", err, errAlreadyExist)
	}

	// Activate v1.
	if err := r.Activate(v1.ID); err != nil {
		t.Fatalf("Activate v1: %v", err)
	}
	active, err := r.GetActive("gemma")
	if err != nil {
		t.Fatalf("GetActive: %v", err)
	}
	if active.ID != v1.ID {
		t.Fatalf("GetActive: got %s, want %s", active.ID, v1.ID)
	}
	if active.Metrics["tok_s"] != 245 {
		t.Fatalf("Metrics not preserved: got %v", active.Metrics)
	}

	// Activate v2 — v1 should be deactivated.
	if err := r.Activate(v2.ID); err != nil {
		t.Fatalf("Activate v2: %v", err)
	}
	active, err = r.GetActive("gemma")
	if err != nil {
		t.Fatalf("GetActive after switch: %v", err)
	}
	if active.ID != v2.ID {
		t.Fatalf("GetActive after switch: got %s, want %s", active.ID, v2.ID)
	}

	// Old version must no longer be active.
	versions, err := r.List("gemma")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	for _, mv := range versions {
		if mv.ID == v1.ID && mv.Active {
			t.Fatal("v1 should be deactivated after activating v2")
		}
	}
}

func TestListVersions(t *testing.T) {
	r := newTestRegistry(t)

	models := []ModelVersion{
		{ID: "a-v1", Name: "a", Version: "1"},
		{ID: "a-v2", Name: "a", Version: "2"},
		{ID: "b-v1", Name: "b", Version: "1"},
	}
	for _, mv := range models {
		if err := r.Register(mv); err != nil {
			t.Fatalf("Register %s: %v", mv.ID, err)
		}
	}

	listA, err := r.List("a")
	if err != nil {
		t.Fatalf("List a: %v", err)
	}
	if len(listA) != 2 {
		t.Fatalf("List a: got %d, want 2", len(listA))
	}

	listB, err := r.List("b")
	if err != nil {
		t.Fatalf("List b: %v", err)
	}
	if len(listB) != 1 {
		t.Fatalf("List b: got %d, want 1", len(listB))
	}

	listC, err := r.List("c")
	if err != nil {
		t.Fatalf("List c: %v", err)
	}
	if len(listC) != 0 {
		t.Fatalf("List c: got %d, want 0", len(listC))
	}
}

func TestDeleteVersion(t *testing.T) {
	r := newTestRegistry(t)

	mv := ModelVersion{ID: "del-1", Name: "x", Version: "1"}
	if err := r.Register(mv); err != nil {
		t.Fatalf("Register: %v", err)
	}

	if err := r.Delete("del-1"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	list, err := r.List("x")
	if err != nil {
		t.Fatalf("List after delete: %v", err)
	}
	if len(list) != 0 {
		t.Fatalf("List after delete: got %d, want 0", len(list))
	}

	// Delete non-existent must return errNotFound.
	if err := r.Delete("del-1"); err != errNotFound {
		t.Fatalf("Delete non-existent: got %v, want %v", err, errNotFound)
	}
}

func TestNewRegistryBadPath(t *testing.T) {
	_, err := NewRegistry(filepath.Join(t.TempDir(), "no", "such", "dir", "db"))
	if err == nil {
		t.Fatal("expected error for bad path")
	}
}

func TestValidationErrors(t *testing.T) {
	r := newTestRegistry(t)

	// Empty ID.
	if err := r.Register(ModelVersion{Name: "x"}); err != errNilID {
		t.Fatalf("Register empty ID: got %v, want %v", err, errNilID)
	}
	// Empty Name.
	if err := r.Register(ModelVersion{ID: "x"}); err != errNilName {
		t.Fatalf("Register empty Name: got %v, want %v", err, errNilName)
	}
	// Activate empty ID.
	if err := r.Activate(""); err != errNilID {
		t.Fatalf("Activate empty ID: got %v, want %v", err, errNilID)
	}
	// Activate non-existent.
	if err := r.Activate("nope"); err != errNotFound {
		t.Fatalf("Activate non-existent: got %v, want %v", err, errNotFound)
	}
	// Delete empty ID.
	if err := r.Delete(""); err != errNilID {
		t.Fatalf("Delete empty ID: got %v, want %v", err, errNilID)
	}
	// GetActive with no active version.
	if err := r.Register(ModelVersion{ID: "z1", Name: "z"}); err != nil {
		t.Fatalf("Register z1: %v", err)
	}
	if _, err := r.GetActive("z"); err != errNotFound {
		t.Fatalf("GetActive no active: got %v, want %v", err, errNotFound)
	}
	// GetActive unknown model.
	if _, err := r.GetActive("unknown"); err != errNotFound {
		t.Fatalf("GetActive unknown: got %v, want %v", err, errNotFound)
	}
}

func TestActivateIsolatesByName(t *testing.T) {
	r := newTestRegistry(t)

	if err := r.Register(ModelVersion{ID: "m1-v1", Name: "m1", Version: "1"}); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(ModelVersion{ID: "m2-v1", Name: "m2", Version: "1"}); err != nil {
		t.Fatal(err)
	}
	if err := r.Activate("m1-v1"); err != nil {
		t.Fatal(err)
	}
	if err := r.Activate("m2-v1"); err != nil {
		t.Fatal(err)
	}
	// Both should be independently active.
	a1, err := r.GetActive("m1")
	if err != nil {
		t.Fatalf("GetActive m1: %v", err)
	}
	if a1.ID != "m1-v1" {
		t.Fatalf("got %s, want m1-v1", a1.ID)
	}
	a2, err := r.GetActive("m2")
	if err != nil {
		t.Fatalf("GetActive m2: %v", err)
	}
	if a2.ID != "m2-v1" {
		t.Fatalf("got %s, want m2-v1", a2.ID)
	}
}

func TestActivateAlreadyActive(t *testing.T) {
	r := newTestRegistry(t)

	if err := r.Register(ModelVersion{ID: "aa-1", Name: "aa", Version: "1"}); err != nil {
		t.Fatal(err)
	}
	if err := r.Activate("aa-1"); err != nil {
		t.Fatal(err)
	}
	// Re-activate should be a no-op.
	if err := r.Activate("aa-1"); err != nil {
		t.Fatalf("re-Activate: %v", err)
	}
	active, err := r.GetActive("aa")
	if err != nil {
		t.Fatal(err)
	}
	if active.ID != "aa-1" {
		t.Fatalf("got %s, want aa-1", active.ID)
	}
}

func TestPersistenceAcrossReopen(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "persist.db")

	// Open, register, activate, close.
	r1, err := NewRegistry(dbPath)
	if err != nil {
		t.Fatalf("NewRegistry (1st open): %v", err)
	}
	mv := ModelVersion{
		ID:        "persist-v1",
		Name:      "persist-model",
		Version:   "1.0",
		Path:      "/models/persist.gguf",
		Format:    "gguf",
		CreatedAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		Metrics:   map[string]float64{"loss": 0.42},
	}
	if err := r1.Register(mv); err != nil {
		t.Fatalf("Register: %v", err)
	}
	if err := r1.Activate(mv.ID); err != nil {
		t.Fatalf("Activate: %v", err)
	}
	if err := r1.Close(); err != nil {
		t.Fatalf("Close (1st): %v", err)
	}

	// Reopen and verify data survived.
	r2, err := NewRegistry(dbPath)
	if err != nil {
		t.Fatalf("NewRegistry (2nd open): %v", err)
	}
	defer r2.Close()

	active, err := r2.GetActive("persist-model")
	if err != nil {
		t.Fatalf("GetActive after reopen: %v", err)
	}
	if active.ID != "persist-v1" {
		t.Fatalf("GetActive after reopen: got %s, want persist-v1", active.ID)
	}
	if active.Path != "/models/persist.gguf" {
		t.Fatalf("Path after reopen: got %s", active.Path)
	}
	if !active.CreatedAt.Equal(time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)) {
		t.Fatalf("CreatedAt after reopen: got %v", active.CreatedAt)
	}
	if active.Metrics["loss"] != 0.42 {
		t.Fatalf("Metrics after reopen: got %v", active.Metrics)
	}

	list, err := r2.List("persist-model")
	if err != nil {
		t.Fatalf("List after reopen: %v", err)
	}
	if len(list) != 1 {
		t.Fatalf("List after reopen: got %d, want 1", len(list))
	}
}
