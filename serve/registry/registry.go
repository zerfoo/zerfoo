// Package registry provides a bbolt-backed model version registry for tracking,
// activating, and managing model versions used by the serving layer.
package registry

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	bolt "go.etcd.io/bbolt"
)

var (
	bucketName      = []byte("models")
	errNotFound     = errors.New("registry: model version not found")
	errNilID        = errors.New("registry: ID must not be empty")
	errNilName      = errors.New("registry: Name must not be empty")
	errAlreadyExist = errors.New("registry: model version already exists")
)

// ModelVersion describes a single registered model version.
type ModelVersion struct {
	ID        string             `json:"id"`
	Name      string             `json:"name"`
	Version   string             `json:"version"`
	Path      string             `json:"path"`
	Format    string             `json:"format"`
	CreatedAt time.Time          `json:"created_at"`
	Metrics   map[string]float64 `json:"metrics,omitempty"`
	Active    bool               `json:"active"`
}

// Registry is a bbolt-backed store for model versions.
type Registry struct {
	db *bolt.DB
}

// NewRegistry opens (or creates) a bbolt database at dbPath and returns a Registry.
func NewRegistry(dbPath string) (*Registry, error) {
	db, err := bolt.Open(dbPath, 0600, &bolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return nil, fmt.Errorf("registry: open db: %w", err)
	}
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(bucketName)
		return err
	}); err != nil {
		db.Close()
		return nil, fmt.Errorf("registry: init bucket: %w", err)
	}
	return &Registry{db: db}, nil
}

// Register stores a new model version. It returns an error if a version with the
// same ID already exists.
func (r *Registry) Register(mv ModelVersion) error {
	if mv.ID == "" {
		return errNilID
	}
	if mv.Name == "" {
		return errNilName
	}
	data, err := json.Marshal(mv)
	if err != nil {
		return err
	}
	return r.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketName)
		if b.Get([]byte(mv.ID)) != nil {
			return errAlreadyExist
		}
		return b.Put([]byte(mv.ID), data)
	})
}

// Activate marks the version with the given id as active and deactivates all
// other versions that share the same Name.
func (r *Registry) Activate(id string) error {
	if id == "" {
		return errNilID
	}
	return r.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketName)

		// Load the target version to learn its Name.
		raw := b.Get([]byte(id))
		if raw == nil {
			return errNotFound
		}
		var target ModelVersion
		if err := json.Unmarshal(raw, &target); err != nil {
			return err
		}

		// Scan all versions: deactivate siblings, activate target.
		c := b.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			var mv ModelVersion
			if err := json.Unmarshal(v, &mv); err != nil {
				return err
			}
			if mv.Name != target.Name {
				continue
			}
			changed := false
			if mv.ID == id && !mv.Active {
				mv.Active = true
				changed = true
			} else if mv.ID != id && mv.Active {
				mv.Active = false
				changed = true
			}
			if changed {
				data, err := json.Marshal(mv)
				if err != nil {
					return err
				}
				if err := b.Put(k, data); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

// GetActive returns the currently active version for the given model name.
// It returns errNotFound if no active version exists.
func (r *Registry) GetActive(name string) (*ModelVersion, error) {
	var result *ModelVersion
	err := r.db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket(bucketName).Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			var mv ModelVersion
			if err := json.Unmarshal(v, &mv); err != nil {
				return err
			}
			if mv.Name == name && mv.Active {
				result = &mv
				return nil
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, errNotFound
	}
	return result, nil
}

// List returns all versions registered under the given model name.
func (r *Registry) List(name string) ([]ModelVersion, error) {
	var out []ModelVersion
	err := r.db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket(bucketName).Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			var mv ModelVersion
			if err := json.Unmarshal(v, &mv); err != nil {
				return err
			}
			if mv.Name == name {
				out = append(out, mv)
			}
		}
		return nil
	})
	return out, err
}

// Delete removes the model version with the given id.
// It returns errNotFound if the id does not exist.
func (r *Registry) Delete(id string) error {
	if id == "" {
		return errNilID
	}
	return r.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketName)
		if b.Get([]byte(id)) == nil {
			return errNotFound
		}
		return b.Delete([]byte(id))
	})
}

// Close closes the underlying bbolt database.
func (r *Registry) Close() error {
	return r.db.Close()
}
