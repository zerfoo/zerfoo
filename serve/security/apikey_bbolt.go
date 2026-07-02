package security

import (
	"encoding/json"
	"fmt"

	bolt "go.etcd.io/bbolt"
)

var bucketName = []byte("apikeys")

// BboltKeyStoreBackend is a persistent KeyStoreBackend backed by a bbolt database.
type BboltKeyStoreBackend struct {
	db *bolt.DB
}

// NewBboltKeyStoreBackend opens or creates a bbolt database at path and returns
// a backend ready for use with KeyStore.
func NewBboltKeyStoreBackend(path string) (*BboltKeyStoreBackend, error) {
	db, err := bolt.Open(path, 0600, nil)
	if err != nil {
		return nil, fmt.Errorf("security: open bbolt db: %w", err)
	}
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(bucketName)
		return err
	}); err != nil {
		db.Close()
		return nil, fmt.Errorf("security: create bucket: %w", err)
	}
	return &BboltKeyStoreBackend{db: db}, nil
}

// Close closes the underlying bbolt database.
func (b *BboltKeyStoreBackend) Close() error {
	return b.db.Close()
}

// Store persists an API key under the given hash.
func (b *BboltKeyStoreBackend) Store(hash string, key *APIKey) error {
	data, err := json.Marshal(key)
	if err != nil {
		return fmt.Errorf("security: marshal key: %w", err)
	}
	return b.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(bucketName).Put([]byte(hash), data)
	})
}

// Load retrieves an API key by its hash. Returns nil if not found.
func (b *BboltKeyStoreBackend) Load(hash string) *APIKey {
	var key APIKey
	err := b.db.View(func(tx *bolt.Tx) error {
		data := tx.Bucket(bucketName).Get([]byte(hash))
		if data == nil {
			return nil
		}
		return json.Unmarshal(data, &key)
	})
	if err != nil {
		return nil
	}
	if key.ID == "" {
		return nil
	}
	return &key
}

// Delete removes an API key by its hash.
func (b *BboltKeyStoreBackend) Delete(hash string) error {
	return b.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(bucketName).Delete([]byte(hash))
	})
}

// List returns all stored API keys.
func (b *BboltKeyStoreBackend) List() []*APIKey {
	var keys []*APIKey
	b.db.View(func(tx *bolt.Tx) error {
		return tx.Bucket(bucketName).ForEach(func(k, v []byte) error {
			var key APIKey
			if err := json.Unmarshal(v, &key); err != nil {
				return nil // skip corrupt entries
			}
			keys = append(keys, &key)
			return nil
		})
	})
	return keys
}
