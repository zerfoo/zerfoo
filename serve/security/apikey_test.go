package security

import (
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestKeyStoreCreateAndLookup(t *testing.T) {
	ks := NewKeyStore()
	raw, key, err := ks.Create("test-key", []Scope{ScopeInference}, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(raw, "zf_") {
		t.Fatalf("expected zf_ prefix, got %s", raw[:4])
	}
	if key.ID != "test-key" {
		t.Fatalf("expected ID test-key, got %s", key.ID)
	}

	found := ks.Lookup(raw)
	if found == nil {
		t.Fatal("expected to find key")
	}
	if found.ID != "test-key" {
		t.Fatal("lookup returned wrong key")
	}

	if ks.Lookup("zf_invalid") != nil {
		t.Fatal("expected nil for unknown key")
	}
}

func TestKeyStoreCreateDuplicateID(t *testing.T) {
	ks := NewKeyStore()
	_, _, err := ks.Create("dup", []Scope{ScopeAdmin}, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = ks.Create("dup", []Scope{ScopeAdmin}, time.Time{})
	if err == nil {
		t.Fatal("expected error for duplicate ID")
	}
}

func TestKeyStoreRevoke(t *testing.T) {
	ks := NewKeyStore()
	raw, _, err := ks.Create("rev-key", []Scope{ScopeInference}, time.Time{})
	if err != nil {
		t.Fatal(err)
	}

	if err := ks.Revoke("rev-key"); err != nil {
		t.Fatal(err)
	}

	key := ks.Lookup(raw)
	if key == nil {
		t.Fatal("expected to find revoked key")
	}
	if !key.Revoked {
		t.Fatal("expected key to be revoked")
	}
	if key.Valid(time.Now()) {
		t.Fatal("revoked key should not be valid")
	}
}

func TestKeyStoreRotate(t *testing.T) {
	ks := NewKeyStore()
	oldRaw, _, err := ks.Create("rot-key", []Scope{ScopeInference, ScopeTraining}, time.Time{})
	if err != nil {
		t.Fatal(err)
	}

	newRaw, newKey, err := ks.Rotate("rot-key", time.Now().Add(24*time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if oldRaw == newRaw {
		t.Fatal("rotated key should differ")
	}
	if len(newKey.Scopes) != 2 {
		t.Fatalf("expected 2 scopes, got %d", len(newKey.Scopes))
	}

	if ks.Lookup(oldRaw) != nil {
		t.Fatal("old key should be removed after rotation")
	}
	if ks.Lookup(newRaw) == nil {
		t.Fatal("new key should be found")
	}
}

func TestAPIKeyExpiry(t *testing.T) {
	key := &APIKey{
		ExpiresAt: time.Now().Add(-time.Hour),
	}
	if key.Valid(time.Now()) {
		t.Fatal("expired key should not be valid")
	}

	key.ExpiresAt = time.Now().Add(time.Hour)
	if !key.Valid(time.Now()) {
		t.Fatal("non-expired key should be valid")
	}
}

func TestAPIKeyHasScope(t *testing.T) {
	key := &APIKey{Scopes: []Scope{ScopeInference, ScopeReadOnly}}
	if !key.HasScope(ScopeInference) {
		t.Fatal("expected inference scope")
	}
	if key.HasScope(ScopeAdmin) {
		t.Fatal("should not have admin scope")
	}
}

func TestKeyStoreList(t *testing.T) {
	ks := NewKeyStore()
	ks.Create("a", []Scope{ScopeInference}, time.Time{})
	ks.Create("b", []Scope{ScopeAdmin}, time.Time{})
	ks.Revoke("b")

	list := ks.List()
	if len(list) != 1 {
		t.Fatalf("expected 1 active key, got %d", len(list))
	}
	if list[0].ID != "a" {
		t.Fatalf("expected key 'a', got '%s'", list[0].ID)
	}
}

func TestKeyStoreRevokeNotFound(t *testing.T) {
	ks := NewKeyStore()
	if err := ks.Revoke("nonexistent"); err == nil {
		t.Fatal("expected error for nonexistent key")
	}
}

func TestKeyStoreCreateEmptyID(t *testing.T) {
	ks := NewKeyStore()
	_, _, err := ks.Create("", []Scope{ScopeInference}, time.Time{})
	if err == nil {
		t.Fatal("expected error for empty ID")
	}
}

// TestKeyStoreLookupRaceConcurrentRevoke reproduces CONC-M2: a reader
// pattern shaped exactly like authMiddleware (server.go) -- Lookup, then
// Valid()/HasScope() with no lock held -- running concurrently with Revoke,
// which mutates the live APIKey record's Revoked/RevokedAt fields under
// KeyStore's mutex.
//
// Before the fix, Lookup returned the shared *APIKey stored in the backend,
// so this raced under `go test -race`: Revoke's locked write to k.Revoked
// raced with this goroutine's lock-free k.Valid() read of the same field.
// Lookup now returns a value copy taken under the read lock, so the reader
// never touches the live, concurrently-mutated struct.
func TestKeyStoreLookupRaceConcurrentRevoke(t *testing.T) {
	ks := NewKeyStore()
	raw, _, err := ks.Create("race-key", []Scope{ScopeInference}, time.Time{})
	if err != nil {
		t.Fatal(err)
	}

	const iterations = 500
	var wg sync.WaitGroup

	// Writer: repeatedly revoke, mimicking an admin revoking a key while
	// live traffic is in flight for it.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			_ = ks.Revoke("race-key")
		}
	}()

	// Reader: the authMiddleware pattern -- Lookup, then read Valid()/
	// HasScope() with no lock held.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			key := ks.Lookup(raw)
			if key == nil {
				continue
			}
			_ = key.Valid(time.Now())
			_ = key.HasScope(ScopeInference)
		}
	}()

	wg.Wait()
}

// TestKeyStoreLookupRaceConcurrentRotate is the Rotate analogue of
// TestKeyStoreLookupRaceConcurrentRevoke: Rotate revokes the old record and
// publishes a new raw key/record under the same ID, concurrently with
// authMiddleware-style Lookup+Valid()/HasScope() reads.
//
// The current raw key is published via atomic.Value so the reader picks up
// rotations without racing on the Go string variable itself -- any race
// here must come from the KeyStore, not from this test's bookkeeping.
func TestKeyStoreLookupRaceConcurrentRotate(t *testing.T) {
	ks := NewKeyStore()
	raw, _, err := ks.Create("rotate-race-key", []Scope{ScopeInference}, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	var currentRaw atomic.Value
	currentRaw.Store(raw)

	const iterations = 500
	var wg sync.WaitGroup

	// Writer: repeatedly rotate, mimicking an admin rotating a key while
	// live traffic is in flight for it.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			newRaw, _, err := ks.Rotate("rotate-race-key", time.Time{})
			if err != nil {
				return
			}
			currentRaw.Store(newRaw)
		}
	}()

	// Reader: the authMiddleware pattern -- Lookup, then read Valid()/
	// HasScope() with no lock held.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			key := ks.Lookup(currentRaw.Load().(string))
			if key == nil {
				continue
			}
			_ = key.Valid(time.Now())
			_ = key.HasScope(ScopeInference)
		}
	}()

	wg.Wait()
}
