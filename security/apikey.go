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

// KeyStoreBackend abstracts storage operations for API keys.
type KeyStoreBackend interface {
	Store(hash string, key *APIKey) error
	Load(hash string) *APIKey
	Delete(hash string) error
	List() []*APIKey
}

// memoryBackend is the default in-memory implementation of KeyStoreBackend.
type memoryBackend struct {
	keys map[string]*APIKey
}

func newMemoryBackend() *memoryBackend {
	return &memoryBackend{keys: make(map[string]*APIKey)}
}

func (m *memoryBackend) Store(hash string, key *APIKey) error {
	m.keys[hash] = key
	return nil
}

func (m *memoryBackend) Load(hash string) *APIKey {
	return m.keys[hash]
}

func (m *memoryBackend) Delete(hash string) error {
	delete(m.keys, hash)
	return nil
}

func (m *memoryBackend) List() []*APIKey {
	var result []*APIKey
	for _, k := range m.keys {
		result = append(result, k)
	}
	return result
}

// KeyStoreOption configures a KeyStore.
type KeyStoreOption func(*KeyStore)

// WithBackend sets the storage backend for a KeyStore.
func WithBackend(b KeyStoreBackend) KeyStoreOption {
	return func(s *KeyStore) {
		s.backend = b
	}
}

// KeyStore manages API keys. It is safe for concurrent use.
type KeyStore struct {
	mu      sync.RWMutex
	byID    map[string]*APIKey
	backend KeyStoreBackend
}

// NewKeyStore returns a KeyStore configured with the given options.
// If no backend is provided, an in-memory backend is used.
func NewKeyStore(opts ...KeyStoreOption) *KeyStore {
	s := &KeyStore{
		byID: make(map[string]*APIKey),
	}
	for _, o := range opts {
		o(s)
	}
	if s.backend == nil {
		s.backend = newMemoryBackend()
	}
	// Populate byID index from backend for persistent backends.
	for _, k := range s.backend.List() {
		s.byID[k.ID] = k
	}
	return s
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

	if err := s.backend.Store(hash, key); err != nil {
		return "", nil, err
	}
	s.byID[id] = key
	return rawKey, key, nil
}

// Lookup finds a key by its raw value. Returns nil if not found.
func (s *KeyStore) Lookup(rawKey string) *APIKey {
	h := sha256.Sum256([]byte(rawKey))
	hash := hex.EncodeToString(h[:])

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.backend.Load(hash)
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
	if err := s.backend.Store(k.Hash, k); err != nil {
		return err
	}
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
	_ = s.backend.Delete(old.Hash)
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
