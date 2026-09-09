// zerfoo-create exposes model construction to coding agents on user hardware.
package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/zerfoo/zerfoo/tabular"
	bolt "go.etcd.io/bbolt"
)

type service struct {
	state, data, library string
	launch               func(string) error
}
type project struct {
	ID        string `json:"id"`
	Objective string `json:"objective"`
}
type datasetRecord struct {
	ID       string                  `json:"id"`
	Project  string                  `json:"project"`
	Manifest tabular.DatasetManifest `json:"manifest"`
}
type plan struct {
	DefinitionSHA256 string                   `json:"definition_sha256,omitempty"`
	Version          int                      `json:"version"`
	EvidenceCards    []evidenceCard           `json:"evidence_cards"`
	ID               string                   `json:"id"`
	Project          string                   `json:"project"`
	Dataset          string                   `json:"dataset"`
	Rationale        string                   `json:"rationale"`
	Evidence         []string                 `json:"evidence"`
	Config           tabular.ClassifierConfig `json:"config"`
	Options          tabular.FitOptions       `json:"options"`
}
type run struct {
	ID         string                   `json:"id"`
	Plan       string                   `json:"plan"`
	Status     string                   `json:"status"`
	Cancel     bool                     `json:"cancel_requested"`
	Error      string                   `json:"error,omitempty"`
	Progress   tabular.TrainingProgress `json:"progress"`
	Artifact   string                   `json:"artifact,omitempty"`
	Hash       string                   `json:"artifact_sha256,omitempty"`
	Validation *tabular.Evaluation      `json:"validation,omitempty"`
	Created    time.Time                `json:"created"`
	Updated    time.Time                `json:"updated"`
}

func newService(state, data, library string) (*service, error) {
	if err := os.MkdirAll(state, 0700); err != nil {
		return nil, err
	}
	absolute, err := filepath.Abs(state)
	if err != nil {
		return nil, err
	}
	return &service{state: absolute, data: data, library: library}, nil
}
func newID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}
func (s *service) transaction(fn func(*bolt.Bucket) error) error {
	db, err := bolt.Open(filepath.Join(s.state, "projects.db"), 0600, &bolt.Options{Timeout: 5 * time.Second})
	if err != nil {
		return err
	}
	err = db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists([]byte("objects"))
		if err != nil {
			return err
		}
		return fn(b)
	})
	return errors.Join(err, db.Close())
}
func put(b *bolt.Bucket, key string, value any) error {
	raw, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return b.Put([]byte(key), raw)
}
func get(b *bolt.Bucket, key string, value any) error {
	raw := b.Get([]byte(key))
	if raw == nil {
		return fmt.Errorf("not found: %s", key)
	}
	return json.Unmarshal(raw, value)
}
func (s *service) load(key string, value any) error {
	return s.transaction(func(b *bolt.Bucket) error { return get(b, key, value) })
}
func (s *service) save(key string, value any) error {
	return s.transaction(func(b *bolt.Bucket) error { return put(b, key, value) })
}
func decode(raw json.RawMessage, value any) error {
	return strictJSON(raw, value)
}
func (s *service) dataset(ctx context.Context, id string) (*tabular.Dataset, error) {
	var record datasetRecord
	if err := s.load("dataset/"+id, &record); err != nil {
		return nil, err
	}
	root, err := os.OpenRoot(s.state)
	if err != nil {
		return nil, err
	}
	file, err := root.Open(record.ID + ".csv")
	if err != nil {
		return nil, errors.Join(err, root.Close())
	}
	dataset, err := tabular.ReplayCSV(ctx, file, record.Manifest)
	return dataset, errors.Join(err, file.Close(), root.Close())
}
