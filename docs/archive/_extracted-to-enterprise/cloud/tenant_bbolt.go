package cloud

import (
	"encoding/json"
	"fmt"

	bolt "go.etcd.io/bbolt"
)

var tenantsBucket = []byte("tenants")

// BboltTenantStoreBackend is a persistent TenantStoreBackend backed by a bbolt database.
type BboltTenantStoreBackend struct {
	db *bolt.DB
}

// NewBboltTenantStoreBackend opens or creates a bbolt database at path and
// returns a backend ready for use with NewTenantManager(WithTenantBackend(...)).
func NewBboltTenantStoreBackend(path string) (*BboltTenantStoreBackend, error) {
	db, err := bolt.Open(path, 0600, nil)
	if err != nil {
		return nil, fmt.Errorf("cloud: open bbolt db: %w", err)
	}
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(tenantsBucket)
		return err
	}); err != nil {
		db.Close()
		return nil, fmt.Errorf("cloud: create bucket: %w", err)
	}
	return &BboltTenantStoreBackend{db: db}, nil
}

// Close closes the underlying bbolt database.
func (b *BboltTenantStoreBackend) Close() error {
	return b.db.Close()
}

// SaveTenant persists a tenant configuration as JSON keyed by its ID.
func (b *BboltTenantStoreBackend) SaveTenant(id string, cfg TenantConfig) error {
	data, err := json.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("cloud: marshal tenant %s: %w", id, err)
	}
	return b.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(tenantsBucket).Put([]byte(id), data)
	})
}

// LoadTenant retrieves a tenant configuration by ID.
func (b *BboltTenantStoreBackend) LoadTenant(id string) (*TenantConfig, bool) {
	var cfg TenantConfig
	found := false
	b.db.View(func(tx *bolt.Tx) error {
		data := tx.Bucket(tenantsBucket).Get([]byte(id))
		if data == nil {
			return nil
		}
		if err := json.Unmarshal(data, &cfg); err != nil {
			return err
		}
		found = true
		return nil
	})
	if !found {
		return nil, false
	}
	return &cfg, true
}

// DeleteTenant removes a tenant by ID.
func (b *BboltTenantStoreBackend) DeleteTenant(id string) error {
	return b.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(tenantsBucket).Delete([]byte(id))
	})
}

// ListTenants returns all stored tenant configurations.
func (b *BboltTenantStoreBackend) ListTenants() []TenantConfig {
	var result []TenantConfig
	b.db.View(func(tx *bolt.Tx) error {
		return tx.Bucket(tenantsBucket).ForEach(func(k, v []byte) error {
			var cfg TenantConfig
			if err := json.Unmarshal(v, &cfg); err != nil {
				return nil // skip corrupt entries
			}
			result = append(result, cfg)
			return nil
		})
	})
	return result
}
