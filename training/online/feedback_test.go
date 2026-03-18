package online

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func makeFeedbackSignal(id string, score float64) FeedbackSignal {
	return FeedbackSignal{
		Timestamp:    time.Now(),
		PredictionID: id,
		Predicted:    []float32{0.1, 0.2, 0.3},
		Actual:       []float32{0.15, 0.25, 0.35},
		Score:        score,
	}
}

func TestRecord(t *testing.T) {
	dir := t.TempDir()
	fc, err := NewFeedbackCollector(FeedbackConfig{
		BufferSize:    10,
		FlushInterval: time.Minute,
		StoragePath:   dir,
	})
	if err != nil {
		t.Fatalf("NewFeedbackCollector: %v", err)
	}
	defer fc.Close()

	for i := range 5 {
		if err := fc.Record(makeFeedbackSignal(string(rune('a'+i)), float64(i))); err != nil {
			t.Fatalf("Record(%d): %v", i, err)
		}
	}

	fc.mu.Lock()
	n := len(fc.buffer)
	fc.mu.Unlock()

	if n != 5 {
		t.Errorf("buffer length = %d, want 5", n)
	}
}

func TestFlush(t *testing.T) {
	dir := t.TempDir()
	fc, err := NewFeedbackCollector(FeedbackConfig{
		BufferSize:    10,
		FlushInterval: time.Minute,
		StoragePath:   dir,
	})
	if err != nil {
		t.Fatalf("NewFeedbackCollector: %v", err)
	}
	defer fc.Close()

	for i := range 3 {
		if err := fc.Record(makeFeedbackSignal(string(rune('a'+i)), float64(i))); err != nil {
			t.Fatalf("Record(%d): %v", i, err)
		}
	}

	signals, err := fc.Flush()
	if err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if len(signals) != 3 {
		t.Errorf("Flush returned %d signals, want 3", len(signals))
	}

	// Buffer should be cleared.
	fc.mu.Lock()
	n := len(fc.buffer)
	fc.mu.Unlock()
	if n != 0 {
		t.Errorf("buffer length after flush = %d, want 0", n)
	}

	// JSONL file should exist.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 JSONL file, got %d", len(entries))
	}
}

func TestAutoFlush(t *testing.T) {
	dir := t.TempDir()
	bufSize := 3
	fc, err := NewFeedbackCollector(FeedbackConfig{
		BufferSize:    bufSize,
		FlushInterval: time.Minute,
		StoragePath:   dir,
	})
	if err != nil {
		t.Fatalf("NewFeedbackCollector: %v", err)
	}
	defer fc.Close()

	// Record exactly BufferSize signals to trigger auto-flush.
	for i := range bufSize {
		if err := fc.Record(makeFeedbackSignal(string(rune('a'+i)), float64(i))); err != nil {
			t.Fatalf("Record(%d): %v", i, err)
		}
	}

	// Buffer should be empty after auto-flush.
	fc.mu.Lock()
	n := len(fc.buffer)
	fc.mu.Unlock()
	if n != 0 {
		t.Errorf("buffer length after auto-flush = %d, want 0", n)
	}

	// JSONL file should have been written.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 JSONL file after auto-flush, got %d", len(entries))
	}

	// Verify the file contains the right number of signals.
	all, err := fc.ReadAll()
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(all) != bufSize {
		t.Errorf("ReadAll returned %d signals, want %d", len(all), bufSize)
	}
}

func TestFeedbackPersistence(t *testing.T) {
	dir := t.TempDir()
	fc, err := NewFeedbackCollector(FeedbackConfig{
		BufferSize:    10,
		FlushInterval: time.Minute,
		StoragePath:   dir,
	})
	if err != nil {
		t.Fatalf("NewFeedbackCollector: %v", err)
	}

	for i := range 4 {
		if err := fc.Record(makeFeedbackSignal(string(rune('a'+i)), float64(i))); err != nil {
			t.Fatalf("Record(%d): %v", i, err)
		}
	}
	if _, err := fc.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := fc.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// New collector reads signals from disk.
	fc2, err := NewFeedbackCollector(FeedbackConfig{
		BufferSize:    10,
		FlushInterval: time.Minute,
		StoragePath:   dir,
	})
	if err != nil {
		t.Fatalf("NewFeedbackCollector(2): %v", err)
	}
	defer fc2.Close()

	all, err := fc2.ReadAll()
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(all) != 4 {
		t.Errorf("ReadAll returned %d signals, want 4", len(all))
	}
	if all[0].PredictionID != "a" {
		t.Errorf("first signal PredictionID = %q, want %q", all[0].PredictionID, "a")
	}
}

func TestStartStop(t *testing.T) {
	dir := t.TempDir()
	fc, err := NewFeedbackCollector(FeedbackConfig{
		BufferSize:    100,
		FlushInterval: 50 * time.Millisecond,
		StoragePath:   dir,
	})
	if err != nil {
		t.Fatalf("NewFeedbackCollector: %v", err)
	}

	if err := fc.Record(makeFeedbackSignal("x", 1.0)); err != nil {
		t.Fatalf("Record: %v", err)
	}

	ctx := context.Background()
	fc.Start(ctx)

	// Wait for at least one flush interval.
	time.Sleep(150 * time.Millisecond)

	fc.Stop()

	// Signal should have been flushed to disk.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	found := false
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".jsonl" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected JSONL file after Start/Stop, found none")
	}
}
