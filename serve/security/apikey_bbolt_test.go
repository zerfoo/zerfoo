package security

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestBboltKeyStoreBackend(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "keys.db")

	backend, err := NewBboltKeyStoreBackend(dbPath)
	if err != nil {
		t.Fatalf("open bbolt backend: %v", err)
	}
	defer backend.Close()

	ks := NewKeyStore(WithBackend(backend))

	// Create a key.
	raw, key, err := ks.Create("bbolt-key", []Scope{ScopeInference, ScopeAdmin}, time.Time{})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if key.ID != "bbolt-key" {
		t.Fatalf("expected ID bbolt-key, got %s", key.ID)
	}

	// Lookup by raw key.
	found := ks.Lookup(raw)
	if found == nil {
		t.Fatal("expected to find key after create")
	}
	if found.ID != "bbolt-key" {
		t.Fatal("lookup returned wrong key")
	}

	// Unknown key returns nil.
	if ks.Lookup("zf_unknown") != nil {
		t.Fatal("expected nil for unknown key")
	}

	// Revoke.
	if err := ks.Revoke("bbolt-key"); err != nil {
		t.Fatalf("revoke: %v", err)
	}
	revoked := ks.Lookup(raw)
	if revoked == nil {
		t.Fatal("expected to find revoked key")
	}
	if !revoked.Revoked {
		t.Fatal("expected key to be revoked")
	}
	if revoked.Valid(time.Now()) {
		t.Fatal("revoked key should not be valid")
	}

	// List should exclude revoked keys.
	list := ks.List()
	if len(list) != 0 {
		t.Fatalf("expected 0 active keys, got %d", len(list))
	}

	// Create a second key, then rotate it.
	raw2, _, err := ks.Create("rot-key", []Scope{ScopeTraining}, time.Now().Add(time.Hour))
	if err != nil {
		t.Fatalf("create rot-key: %v", err)
	}
	newRaw, newKey, err := ks.Rotate("rot-key", time.Now().Add(24*time.Hour))
	if err != nil {
		t.Fatalf("rotate: %v", err)
	}
	if raw2 == newRaw {
		t.Fatal("rotated key should differ from original")
	}
	if newKey.ID != "rot-key" {
		t.Fatalf("expected rotated key ID rot-key, got %s", newKey.ID)
	}
	if ks.Lookup(raw2) != nil {
		t.Fatal("old key should be removed after rotation")
	}
	if ks.Lookup(newRaw) == nil {
		t.Fatal("new key should be found after rotation")
	}

	// List round-trip.
	list = ks.List()
	if len(list) != 1 {
		t.Fatalf("expected 1 active key, got %d", len(list))
	}
	if list[0].ID != "rot-key" {
		t.Fatalf("expected rot-key, got %s", list[0].ID)
	}
}

func TestBboltKeyStorePersistence(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "keys.db")

	// Create a key and close the backend.
	backend, err := NewBboltKeyStoreBackend(dbPath)
	if err != nil {
		t.Fatalf("open bbolt backend: %v", err)
	}
	ks := NewKeyStore(WithBackend(backend))
	raw, _, err := ks.Create("persist-key", []Scope{ScopeInference}, time.Time{})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	backend.Close()

	// Reopen and verify key is still present.
	backend2, err := NewBboltKeyStoreBackend(dbPath)
	if err != nil {
		t.Fatalf("reopen bbolt backend: %v", err)
	}
	defer backend2.Close()

	ks2 := NewKeyStore(WithBackend(backend2))
	found := ks2.Lookup(raw)
	if found == nil {
		t.Fatal("expected key to persist across reopen")
	}
	if found.ID != "persist-key" {
		t.Fatalf("expected persist-key, got %s", found.ID)
	}
}

func TestBboltKeyStoreBackendOpenError(t *testing.T) {
	// Attempt to open a directory as a database file.
	dir := t.TempDir()
	_, err := NewBboltKeyStoreBackend(dir)
	if err == nil {
		t.Fatal("expected error opening directory as bbolt db")
	}
}

func TestBboltKeyStoreBackendInvalidPath(t *testing.T) {
	_, err := NewBboltKeyStoreBackend(filepath.Join(os.DevNull, "nonexistent", "keys.db"))
	if err == nil {
		t.Fatal("expected error for invalid path")
	}
}
