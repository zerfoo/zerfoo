package support

import (
	"encoding/json"
	"fmt"

	bolt "go.etcd.io/bbolt"
)

var ticketsBucket = []byte("tickets")

// BboltStoreBackend persists tickets in a bbolt database.
type BboltStoreBackend struct {
	db *bolt.DB
}

// NewBboltStoreBackend opens or creates a bbolt database at path and returns
// a backend ready for use with NewStore(WithStoreBackend(...)).
func NewBboltStoreBackend(path string) (*BboltStoreBackend, error) {
	db, err := bolt.Open(path, 0600, nil)
	if err != nil {
		return nil, fmt.Errorf("bbolt open %s: %w", path, err)
	}
	err = db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(ticketsBucket)
		return err
	})
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("bbolt create bucket: %w", err)
	}
	return &BboltStoreBackend{db: db}, nil
}

// Save persists a ticket as JSON keyed by its ID.
func (b *BboltStoreBackend) Save(ticket *Ticket) error {
	data, err := json.Marshal(ticket)
	if err != nil {
		return fmt.Errorf("marshal ticket %s: %w", ticket.ID, err)
	}
	return b.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(ticketsBucket).Put([]byte(ticket.ID), data)
	})
}

// Get retrieves a ticket by ID.
func (b *BboltStoreBackend) Get(id string) (*Ticket, bool) {
	var t Ticket
	found := false
	b.db.View(func(tx *bolt.Tx) error {
		data := tx.Bucket(ticketsBucket).Get([]byte(id))
		if data == nil {
			return nil
		}
		if err := json.Unmarshal(data, &t); err != nil {
			return err
		}
		found = true
		return nil
	})
	if !found {
		return nil, false
	}
	return &t, true
}

// ListByCustomer returns all tickets belonging to the given customer.
func (b *BboltStoreBackend) ListByCustomer(customerID string) []*Ticket {
	var result []*Ticket
	b.db.View(func(tx *bolt.Tx) error {
		return tx.Bucket(ticketsBucket).ForEach(func(k, v []byte) error {
			var t Ticket
			if err := json.Unmarshal(v, &t); err != nil {
				return err
			}
			if t.CustomerID == customerID {
				result = append(result, &t)
			}
			return nil
		})
	})
	return result
}

// Delete removes a ticket by ID.
func (b *BboltStoreBackend) Delete(id string) error {
	return b.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(ticketsBucket).Delete([]byte(id))
	})
}

// All returns every ticket in the database.
func (b *BboltStoreBackend) All() []*Ticket {
	var result []*Ticket
	b.db.View(func(tx *bolt.Tx) error {
		return tx.Bucket(ticketsBucket).ForEach(func(k, v []byte) error {
			var t Ticket
			if err := json.Unmarshal(v, &t); err != nil {
				return err
			}
			result = append(result, &t)
			return nil
		})
	})
	return result
}

// Close closes the underlying bbolt database.
func (b *BboltStoreBackend) Close() error {
	return b.db.Close()
}
