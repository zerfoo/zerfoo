// Package features provides a feature store for the Wolf time-series ML platform.
// It supports offline CSV loading and online ring-buffer ingestion with
// point-in-time correctness guarantees (no future data leakage).
package features

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"sync"
	"time"
)

const defaultRingCap = 500

// Tick represents a single timestamped feature vector.
type Tick struct {
	Timestamp time.Time
	Features  []float64
}

// FeatureStore manages offline (CSV-loaded) and online (ring-buffer) feature
// data per asset, enforcing point-in-time correctness on reads.
type FeatureStore struct {
	mu      sync.RWMutex
	ringBuf map[string]*ringBuffer
	offline map[string][]Tick
}

// ringBuffer is a fixed-capacity circular buffer of Ticks.
type ringBuffer struct {
	data  []Tick
	head  int // next write position
	count int
	cap   int
}

func newRingBuffer(cap int) *ringBuffer {
	return &ringBuffer{
		data: make([]Tick, cap),
		cap:  cap,
	}
}

func (rb *ringBuffer) append(t Tick) {
	rb.data[rb.head] = t
	rb.head = (rb.head + 1) % rb.cap
	if rb.count < rb.cap {
		rb.count++
	}
}

// ordered returns all ticks in insertion order.
func (rb *ringBuffer) ordered() []Tick {
	if rb.count == 0 {
		return nil
	}
	out := make([]Tick, rb.count)
	if rb.count < rb.cap {
		copy(out, rb.data[:rb.count])
	} else {
		// head points to the oldest element when full
		n := copy(out, rb.data[rb.head:rb.cap])
		copy(out[n:], rb.data[:rb.head])
	}
	return out
}

// NewFeatureStore creates a FeatureStore ready for use.
func NewFeatureStore() *FeatureStore {
	return &FeatureStore{
		ringBuf: make(map[string]*ringBuffer),
		offline: make(map[string][]Tick),
	}
}

// LoadOffline reads a CSV file with columns: timestamp, f1, f2, ...
// and stores rows whose timestamp falls within [start, end].
// Timestamps must be in RFC3339 format.
func (fs *FeatureStore) LoadOffline(asset, csvPath string, start, end time.Time) error {
	if end.Before(start) {
		return errors.New("end must not be before start")
	}

	f, err := os.Open(csvPath)
	if err != nil {
		return fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)

	// Skip header row.
	if _, err := r.Read(); err != nil {
		return fmt.Errorf("read header: %w", err)
	}

	var ticks []Tick
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read row: %w", err)
		}
		if len(record) < 2 {
			return fmt.Errorf("row has %d columns, need at least 2", len(record))
		}

		ts, err := time.Parse(time.RFC3339, record[0])
		if err != nil {
			return fmt.Errorf("parse timestamp %q: %w", record[0], err)
		}

		if ts.Before(start) || ts.After(end) {
			continue
		}

		features := make([]float64, len(record)-1)
		for i, v := range record[1:] {
			features[i], err = strconv.ParseFloat(v, 64)
			if err != nil {
				return fmt.Errorf("parse feature column %d value %q: %w", i+1, v, err)
			}
		}

		ticks = append(ticks, Tick{Timestamp: ts, Features: features})
	}

	fs.mu.Lock()
	fs.offline[asset] = ticks
	fs.mu.Unlock()

	return nil
}

// UpdateOnline appends a tick to the asset's ring buffer. The buffer evicts
// the oldest entry when it reaches capacity (500).
func (fs *FeatureStore) UpdateOnline(asset string, tick Tick) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	rb, ok := fs.ringBuf[asset]
	if !ok {
		rb = newRingBuffer(defaultRingCap)
		fs.ringBuf[asset] = rb
	}
	rb.append(tick)
	return nil
}

// GetFeatures returns all ticks for the asset with timestamp <= asOf,
// combining offline and online data in time order. This enforces
// point-in-time correctness: no future data is ever returned.
func (fs *FeatureStore) GetFeatures(asset string, asOf time.Time) ([]Tick, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	var result []Tick

	// Collect from offline data.
	if off, ok := fs.offline[asset]; ok {
		for _, t := range off {
			if !t.Timestamp.After(asOf) {
				result = append(result, t)
			}
		}
	}

	// Collect from online ring buffer.
	if rb, ok := fs.ringBuf[asset]; ok {
		for _, t := range rb.ordered() {
			if !t.Timestamp.After(asOf) {
				result = append(result, t)
			}
		}
	}

	return result, nil
}
