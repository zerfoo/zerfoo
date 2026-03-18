package online

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// FeedbackSignal represents a single feedback observation comparing a model's
// prediction against the actual outcome.
type FeedbackSignal struct {
	Timestamp    time.Time `json:"timestamp"`
	PredictionID string    `json:"prediction_id"`
	Predicted    []float32 `json:"predicted"`
	Actual       []float32 `json:"actual"`
	Score        float64   `json:"score"`
}

// FeedbackConfig holds parameters for the FeedbackCollector.
type FeedbackConfig struct {
	// BufferSize is the maximum number of signals buffered in memory
	// before an automatic flush to disk.
	BufferSize int

	// FlushInterval is the duration between periodic background flushes.
	FlushInterval time.Duration

	// StoragePath is the directory where JSONL feedback files are written.
	StoragePath string
}

// FeedbackCollector buffers feedback signals in memory and periodically
// flushes them to JSONL files on disk.
type FeedbackCollector struct {
	cfg    FeedbackConfig
	mu     sync.Mutex
	buffer []FeedbackSignal
	seq    int
	cancel context.CancelFunc
	done   chan struct{}
}

// NewFeedbackCollector creates a new FeedbackCollector, creating the
// StoragePath directory if it does not exist.
func NewFeedbackCollector(cfg FeedbackConfig) (*FeedbackCollector, error) {
	if err := os.MkdirAll(cfg.StoragePath, 0755); err != nil {
		return nil, fmt.Errorf("feedback: create storage path: %w", err)
	}
	return &FeedbackCollector{
		cfg:    cfg,
		buffer: make([]FeedbackSignal, 0, cfg.BufferSize),
	}, nil
}

// Record appends a signal to the in-memory buffer. If the buffer reaches
// BufferSize, it is automatically flushed to disk.
func (fc *FeedbackCollector) Record(signal FeedbackSignal) error {
	fc.mu.Lock()
	fc.buffer = append(fc.buffer, signal)
	if fc.cfg.BufferSize > 0 && len(fc.buffer) >= fc.cfg.BufferSize {
		signals := fc.buffer
		fc.buffer = make([]FeedbackSignal, 0, fc.cfg.BufferSize)
		fc.mu.Unlock()
		return fc.writeSignals(signals)
	}
	fc.mu.Unlock()
	return nil
}

// Flush writes all buffered signals to a JSONL file on disk, returns them,
// and clears the buffer.
func (fc *FeedbackCollector) Flush() ([]FeedbackSignal, error) {
	fc.mu.Lock()
	if len(fc.buffer) == 0 {
		fc.mu.Unlock()
		return nil, nil
	}
	signals := fc.buffer
	fc.buffer = make([]FeedbackSignal, 0, fc.cfg.BufferSize)
	fc.mu.Unlock()

	if err := fc.writeSignals(signals); err != nil {
		return nil, err
	}
	return signals, nil
}

// Start begins a background goroutine that flushes buffered signals every
// FlushInterval.
func (fc *FeedbackCollector) Start(ctx context.Context) {
	ctx, fc.cancel = context.WithCancel(ctx)
	fc.done = make(chan struct{})
	go func() {
		defer close(fc.done)
		ticker := time.NewTicker(fc.cfg.FlushInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				fc.Flush()
			}
		}
	}()
}

// Stop stops the background goroutine and performs a final flush.
func (fc *FeedbackCollector) Stop() {
	if fc.cancel != nil {
		fc.cancel()
		<-fc.done
	}
	fc.Flush()
}

// Close releases resources held by the collector.
func (fc *FeedbackCollector) Close() error {
	fc.Stop()
	return nil
}

// ReadAll reads all feedback signals from JSONL files in the storage directory.
func (fc *FeedbackCollector) ReadAll() ([]FeedbackSignal, error) {
	entries, err := os.ReadDir(fc.cfg.StoragePath)
	if err != nil {
		return nil, fmt.Errorf("feedback: read storage dir: %w", err)
	}

	var all []FeedbackSignal
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		signals, err := fc.readFile(filepath.Join(fc.cfg.StoragePath, entry.Name()))
		if err != nil {
			return nil, err
		}
		all = append(all, signals...)
	}
	return all, nil
}

func (fc *FeedbackCollector) writeSignals(signals []FeedbackSignal) error {
	fc.mu.Lock()
	seq := fc.seq
	fc.seq++
	fc.mu.Unlock()

	name := filepath.Join(fc.cfg.StoragePath, fmt.Sprintf("feedback_%06d.jsonl", seq))
	f, err := os.Create(name)
	if err != nil {
		return fmt.Errorf("feedback: create file: %w", err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for _, s := range signals {
		if err := enc.Encode(s); err != nil {
			return fmt.Errorf("feedback: encode signal: %w", err)
		}
	}
	return f.Sync()
}

func (fc *FeedbackCollector) readFile(path string) ([]FeedbackSignal, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("feedback: open file: %w", err)
	}
	defer f.Close()

	var signals []FeedbackSignal
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var s FeedbackSignal
		if err := json.Unmarshal(scanner.Bytes(), &s); err != nil {
			return nil, fmt.Errorf("feedback: unmarshal signal: %w", err)
		}
		signals = append(signals, s)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("feedback: scan file: %w", err)
	}
	return signals, nil
}
