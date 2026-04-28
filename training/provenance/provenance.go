package provenance

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"sync"
	"time"
)

// EventType identifies the kind of lifecycle event.
type EventType string

const (
	EventTraining   EventType = "training"
	EventDataset    EventType = "dataset"
	EventEvaluation EventType = "evaluation"
)

// Event is a single node in the provenance DAG.
type Event struct {
	Hash      string
	ParentID  string
	Type      EventType
	Timestamp time.Time
	Payload   any
}

// TrainingRecord captures a training run.
type TrainingRecord struct {
	RunID           string
	ModelName       string
	ModelVersion    string
	Hyperparameters map[string]string
	StartedAt       time.Time
	FinishedAt      time.Time
	ParentID        string
}

// DatasetRecord captures dataset provenance.
type DatasetRecord struct {
	Name     string
	Version  string
	Checksum string
	NumRows  int
	ParentID string
}

// EvaluationRecord captures evaluation results.
type EvaluationRecord struct {
	ParentID string
	Metrics  map[string]float64
	Dataset  string
	Split    string
}

// Tracker maintains a cryptographic hash chain of model lifecycle events.
type Tracker struct {
	mu     sync.RWMutex
	events map[string]*Event
}

// NewTracker creates a new provenance tracker.
func NewTracker() *Tracker {
	return &Tracker{
		events: make(map[string]*Event),
	}
}

// RecordTraining records a training run event and returns its hash.
func (t *Tracker) RecordTraining(rec TrainingRecord) (string, error) {
	if rec.RunID == "" {
		return "", fmt.Errorf("provenance: RunID is required")
	}
	if rec.ModelName == "" {
		return "", fmt.Errorf("provenance: ModelName is required")
	}

	parentHash := rec.ParentID
	if parentHash != "" {
		if err := t.validateParent(parentHash); err != nil {
			return "", err
		}
	}

	now := time.Now()
	canonical := fmt.Sprintf("training|%s|%s|%s|%s|%s",
		parentHash, rec.RunID, rec.ModelName, rec.ModelVersion,
		canonicalMap(rec.Hyperparameters))

	hash := computeHash(parentHash, canonical, now)

	ev := &Event{
		Hash:      hash,
		ParentID:  parentHash,
		Type:      EventTraining,
		Timestamp: now,
		Payload:   rec,
	}

	t.mu.Lock()
	t.events[hash] = ev
	t.mu.Unlock()

	return hash, nil
}

// RecordDataset records a dataset event and returns its hash.
func (t *Tracker) RecordDataset(rec DatasetRecord) (string, error) {
	if rec.Name == "" {
		return "", fmt.Errorf("provenance: dataset Name is required")
	}

	parentHash := rec.ParentID
	if parentHash != "" {
		if err := t.validateParent(parentHash); err != nil {
			return "", err
		}
	}

	now := time.Now()
	canonical := fmt.Sprintf("dataset|%s|%s|%s|%s|%d",
		parentHash, rec.Name, rec.Version, rec.Checksum, rec.NumRows)

	hash := computeHash(parentHash, canonical, now)

	ev := &Event{
		Hash:      hash,
		ParentID:  parentHash,
		Type:      EventDataset,
		Timestamp: now,
		Payload:   rec,
	}

	t.mu.Lock()
	t.events[hash] = ev
	t.mu.Unlock()

	return hash, nil
}

// RecordEvaluation records an evaluation event and returns its hash.
func (t *Tracker) RecordEvaluation(rec EvaluationRecord) (string, error) {
	parentHash := rec.ParentID
	if parentHash != "" {
		if err := t.validateParent(parentHash); err != nil {
			return "", err
		}
	}

	now := time.Now()
	canonical := fmt.Sprintf("evaluation|%s|%s|%s|%s",
		parentHash, rec.Dataset, rec.Split, canonicalMetrics(rec.Metrics))

	hash := computeHash(parentHash, canonical, now)

	ev := &Event{
		Hash:      hash,
		ParentID:  parentHash,
		Type:      EventEvaluation,
		Timestamp: now,
		Payload:   rec,
	}

	t.mu.Lock()
	t.events[hash] = ev
	t.mu.Unlock()

	return hash, nil
}

// Trace traverses the DAG from the given hash back to the root(s),
// returning events in reverse chronological order (leaf first).
func (t *Tracker) Trace(hash string) ([]Event, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var chain []Event
	current := hash
	visited := make(map[string]bool)

	for current != "" {
		if visited[current] {
			return nil, fmt.Errorf("provenance: cycle detected at %s", current)
		}
		visited[current] = true

		ev, ok := t.events[current]
		if !ok {
			return nil, fmt.Errorf("provenance: event %s not found", current)
		}
		chain = append(chain, *ev)
		current = ev.ParentID
	}

	return chain, nil
}

// Verify checks that the hash chain from the given event back to the root
// is intact. It recomputes each event's hash and compares it to the stored
// value.
func (t *Tracker) Verify(hash string) (bool, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	current := hash
	visited := make(map[string]bool)

	for current != "" {
		if visited[current] {
			return false, fmt.Errorf("provenance: cycle detected at %s", current)
		}
		visited[current] = true

		ev, ok := t.events[current]
		if !ok {
			return false, fmt.Errorf("provenance: event %s not found", current)
		}

		recomputed := recomputeHash(ev)
		if recomputed != ev.Hash {
			return false, nil
		}
		current = ev.ParentID
	}

	return true, nil
}

// Events returns all tracked events. The caller must not modify the returned
// slice.
func (t *Tracker) Events() []Event {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]Event, 0, len(t.events))
	for _, ev := range t.events {
		out = append(out, *ev)
	}
	return out
}

func (t *Tracker) validateParent(hash string) error {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if _, ok := t.events[hash]; !ok {
		return fmt.Errorf("provenance: parent event %s not found", hash)
	}
	return nil
}

func computeHash(parentHash, canonical string, ts time.Time) string {
	h := sha256.New()
	h.Write([]byte(parentHash))
	h.Write([]byte(canonical))
	h.Write([]byte(ts.UTC().Format(time.RFC3339Nano)))
	return hex.EncodeToString(h.Sum(nil))
}

func recomputeHash(ev *Event) string {
	var canonical string
	switch rec := ev.Payload.(type) {
	case TrainingRecord:
		canonical = fmt.Sprintf("training|%s|%s|%s|%s|%s",
			ev.ParentID, rec.RunID, rec.ModelName, rec.ModelVersion,
			canonicalMap(rec.Hyperparameters))
	case DatasetRecord:
		canonical = fmt.Sprintf("dataset|%s|%s|%s|%s|%d",
			ev.ParentID, rec.Name, rec.Version, rec.Checksum, rec.NumRows)
	case EvaluationRecord:
		canonical = fmt.Sprintf("evaluation|%s|%s|%s|%s",
			ev.ParentID, rec.Dataset, rec.Split, canonicalMetrics(rec.Metrics))
	default:
		return ""
	}
	return computeHash(ev.ParentID, canonical, ev.Timestamp)
}

// canonicalMap produces a deterministic string from a map by sorting keys.
func canonicalMap(m map[string]string) string {
	if len(m) == 0 {
		return ""
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var s string
	for i, k := range keys {
		if i > 0 {
			s += ","
		}
		s += k + "=" + m[k]
	}
	return s
}

// canonicalMetrics produces a deterministic string from metric values.
func canonicalMetrics(m map[string]float64) string {
	if len(m) == 0 {
		return ""
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var s string
	for i, k := range keys {
		if i > 0 {
			s += ","
		}
		s += fmt.Sprintf("%s=%g", k, m[k])
	}
	return s
}
