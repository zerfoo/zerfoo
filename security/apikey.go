package security

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"sync"
	"time"
)

// Scope defines the permissions granted to an API key.
type Scope string

const (
	ScopeInference Scope = "inference"
	ScopeTraining  Scope = "training"
	ScopeAdmin     Scope = "admin"
	ScopeReadOnly  Scope = "read_only"
)

// APIKey represents a managed API key with scopes and lifecycle metadata.
type APIKey struct {
	ID        string
	Hash      string // SHA-256 hash of the raw key; raw key is never stored
	Scopes    []Scope
	CreatedAt time.Time
	ExpiresAt time.Time // zero value means no expiry
	Revoked   bool
	RevokedAt time.Time
}

// HasScope reports whether the key has the given scope.
func (k *APIKey) HasScope(s Scope) bool {
	for _, sc := range k.Scopes {
		if sc == s {
			return true
		}
	}
	return false
}

// Valid reports whether the key is usable: not revoked and not expired.
func (k *APIKey) Valid(now time.Time) bool {
	if k.Revoked {
		return false
	}
	if !k.ExpiresAt.IsZero() && now.After(k.ExpiresAt) {
		return false
	}
	return true
}

// KeyStore manages API keys. It is safe for concurrent use.
type KeyStore struct {
	mu   sync.RWMutex
	keys map[string]*APIKey // keyed by hash
	byID map[string]*APIKey
}

// NewKeyStore returns an empty KeyStore.
func NewKeyStore() *KeyStore {
	return &KeyStore{
		keys: make(map[string]*APIKey),
		byID: make(map[string]*APIKey),
	}
}

// Create generates a new API key with the given scopes and optional expiry.
// It returns the raw key (shown once) and the stored APIKey record.
func (s *KeyStore) Create(id string, scopes []Scope, expiresAt time.Time) (rawKey string, key *APIKey, err error) {
	if id == "" {
		return "", nil, errors.New("security: key ID must not be empty")
	}

	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", nil, err
	}
	rawKey = "zf_" + hex.EncodeToString(raw)

	h := sha256.Sum256([]byte(rawKey))
	hash := hex.EncodeToString(h[:])

	key = &APIKey{
		ID:        id,
		Hash:      hash,
		Scopes:    scopes,
		CreatedAt: time.Now(),
		ExpiresAt: expiresAt,
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.byID[id]; exists {
		return "", nil, errors.New("security: duplicate key ID")
	}

	s.keys[hash] = key
	s.byID[id] = key
	return rawKey, key, nil
}

// Lookup finds a key by its raw value. Returns nil if not found.
func (s *KeyStore) Lookup(rawKey string) *APIKey {
	h := sha256.Sum256([]byte(rawKey))
	hash := hex.EncodeToString(h[:])

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.keys[hash]
}

// Revoke marks a key as revoked by its ID.
func (s *KeyStore) Revoke(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	k, ok := s.byID[id]
	if !ok {
		return errors.New("security: key not found")
	}
	k.Revoked = true
	k.RevokedAt = time.Now()
	return nil
}

// Rotate revokes the old key and creates a new one with the same scopes.
// Returns the new raw key and key record.
func (s *KeyStore) Rotate(id string, newExpiry time.Time) (rawKey string, key *APIKey, err error) {
	s.mu.Lock()
	old, ok := s.byID[id]
	if !ok {
		s.mu.Unlock()
		return "", nil, errors.New("security: key not found")
	}
	scopes := make([]Scope, len(old.Scopes))
	copy(scopes, old.Scopes)
	old.Revoked = true
	old.RevokedAt = time.Now()
	delete(s.byID, id)
	delete(s.keys, old.Hash)
	s.mu.Unlock()

	return s.Create(id, scopes, newExpiry)
}

// List returns all non-revoked keys.
func (s *KeyStore) List() []*APIKey {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var result []*APIKey
	for _, k := range s.byID {
		if !k.Revoked {
			result = append(result, k)
		}
	}
	return result
}
