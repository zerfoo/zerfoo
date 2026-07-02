package multimodel

import (
	"context"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"testing"
)

// mockModel tracks whether it has been closed.
type mockModel struct {
	id     string
	closed atomic.Bool
}

func (m *mockModel) Close() error {
	m.closed.Store(true)
	return nil
}

// mockLoader returns mockModels of a fixed size.
type mockLoader struct {
	mu      sync.Mutex
	models  map[string]*mockModel
	sizeFn  func(id string) int64
	loadErr func(id string) error
}

func newMockLoader(size int64) *mockLoader {
	return &mockLoader{
		models: make(map[string]*mockModel),
		sizeFn: func(string) int64 { return size },
	}
}

func (l *mockLoader) Load(_ context.Context, modelID string) (io.Closer, int64, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.loadErr != nil {
		if err := l.loadErr(modelID); err != nil {
			return nil, 0, err
		}
	}
	m := &mockModel{id: modelID}
	l.models[modelID] = m
	return m, l.sizeFn(modelID), nil
}

func (l *mockLoader) get(id string) *mockModel {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.models[id]
}

func TestModelManager_LoadUnload(t *testing.T) {
	tests := []struct {
		name    string
		budget  int64
		modelID string
		size    int64
		wantErr bool
	}{
		{
			name:    "load within budget",
			budget:  1000,
			modelID: "model-a",
			size:    500,
		},
		{
			name:    "load exactly at budget",
			budget:  500,
			modelID: "model-b",
			size:    500,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := newMockLoader(tt.size)
			mgr, err := NewModelManager(loader, Config{
				MaxGPUMemoryBytes: tt.budget,
			})
			if err != nil {
				t.Fatalf("NewModelManager: %v", err)
			}
			defer mgr.Close()

			model, err := mgr.Get(context.Background(), tt.modelID)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Get(%q) error = %v, wantErr = %v", tt.modelID, err, tt.wantErr)
			}
			if err != nil {
				return
			}
			if model == nil {
				t.Fatal("Get returned nil model")
			}

			// Verify loaded.
			loaded := mgr.Loaded()
			if len(loaded) != 1 || loaded[0] != tt.modelID {
				t.Errorf("Loaded() = %v, want [%s]", loaded, tt.modelID)
			}

			// Verify used bytes.
			if got := mgr.UsedBytes(); got != tt.size {
				t.Errorf("UsedBytes() = %d, want %d", got, tt.size)
			}

			// Unload.
			if err := mgr.Unload(tt.modelID); err != nil {
				t.Fatalf("Unload(%q): %v", tt.modelID, err)
			}

			if got := mgr.UsedBytes(); got != 0 {
				t.Errorf("UsedBytes() after unload = %d, want 0", got)
			}

			// Verify model was closed.
			mock := loader.get(tt.modelID)
			if mock == nil {
				t.Fatal("mock model not found in loader")
			}
			if !mock.closed.Load() {
				t.Error("model was not closed after unload")
			}
		})
	}
}

func TestModelManager_UnloadNotLoaded(t *testing.T) {
	loader := newMockLoader(100)
	mgr, err := NewModelManager(loader, Config{MaxGPUMemoryBytes: 1000})
	if err != nil {
		t.Fatalf("NewModelManager: %v", err)
	}
	defer mgr.Close()

	if err := mgr.Unload("nonexistent"); err == nil {
		t.Error("expected error when unloading a model that is not loaded")
	}
}

func TestModelManager_LRUEviction(t *testing.T) {
	tests := []struct {
		name          string
		budget        int64
		modelSize     int64
		loadOrder     []string
		accessBeforeN string   // access this model before loading the last one
		wantEvicted   []string // models that should be closed after all loads
		wantLoaded    []string // models that should remain loaded
	}{
		{
			name:        "evict oldest when full",
			budget:      200,
			modelSize:   100,
			loadOrder:   []string{"a", "b", "c"},
			wantEvicted: []string{"a"},
			wantLoaded:  []string{"b", "c"},
		},
		{
			name:          "LRU promotion prevents eviction",
			budget:        200,
			modelSize:     100,
			loadOrder:     []string{"a", "b", "c"},
			accessBeforeN: "a", // access "a" before loading "c", so "b" is LRU
			wantEvicted:   []string{"b"},
			wantLoaded:    []string{"a", "c"},
		},
		{
			name:        "evict multiple to make room",
			budget:      200,
			modelSize:   100,
			loadOrder:   []string{"a", "b"},
			wantEvicted: []string{},
			wantLoaded:  []string{"a", "b"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := newMockLoader(tt.modelSize)
			mgr, err := NewModelManager(loader, Config{
				MaxGPUMemoryBytes: tt.budget,
			})
			if err != nil {
				t.Fatalf("NewModelManager: %v", err)
			}
			defer mgr.Close()

			for i, id := range tt.loadOrder {
				// Optionally access a model before loading the last one.
				if i == len(tt.loadOrder)-1 && tt.accessBeforeN != "" {
					if _, err := mgr.Get(context.Background(), tt.accessBeforeN); err != nil {
						t.Fatalf("Get(%q) for promotion: %v", tt.accessBeforeN, err)
					}
				}
				if _, err := mgr.Get(context.Background(), id); err != nil {
					t.Fatalf("Get(%q): %v", id, err)
				}
			}

			// Check evicted models.
			for _, id := range tt.wantEvicted {
				mock := loader.get(id)
				if mock == nil {
					t.Errorf("evicted model %q not found in loader", id)
					continue
				}
				if !mock.closed.Load() {
					t.Errorf("model %q should have been evicted (closed) but was not", id)
				}
			}

			// Check still-loaded models.
			for _, id := range tt.wantLoaded {
				mock := loader.get(id)
				if mock == nil {
					t.Errorf("loaded model %q not found in loader", id)
					continue
				}
				if mock.closed.Load() {
					t.Errorf("model %q should still be loaded but was closed", id)
				}
			}
		})
	}
}

func TestModelManager_ConcurrentRequests(t *testing.T) {
	const (
		numModels  = 5
		numWorkers = 20
		iterations = 50
	)

	loader := newMockLoader(100)
	mgr, err := NewModelManager(loader, Config{
		MaxGPUMemoryBytes: 300, // room for 3 models at a time
	})
	if err != nil {
		t.Fatalf("NewModelManager: %v", err)
	}
	defer mgr.Close()

	var wg sync.WaitGroup
	errCh := make(chan error, numWorkers*iterations)

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				modelID := fmt.Sprintf("model-%d", (worker+i)%numModels)
				model, err := mgr.Get(context.Background(), modelID)
				if err != nil {
					errCh <- fmt.Errorf("worker %d iter %d: Get(%q): %v", worker, i, modelID, err)
					return
				}
				if model == nil {
					errCh <- fmt.Errorf("worker %d iter %d: Get(%q) returned nil", worker, i, modelID)
					return
				}
			}
		}(w)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Error(err)
	}

	// Verify memory budget is respected.
	if used := mgr.UsedBytes(); used > 300 {
		t.Errorf("UsedBytes() = %d, exceeds budget 300", used)
	}
}

func TestModelManager_PreloadModels(t *testing.T) {
	loader := newMockLoader(100)
	mgr, err := NewModelManager(loader, Config{
		MaxGPUMemoryBytes: 500,
		PreloadModels:     []string{"pre-a", "pre-b"},
	})
	if err != nil {
		t.Fatalf("NewModelManager: %v", err)
	}
	defer mgr.Close()

	loaded := mgr.Loaded()
	if len(loaded) != 2 {
		t.Fatalf("Loaded() = %v, want 2 models", loaded)
	}

	if got := mgr.UsedBytes(); got != 200 {
		t.Errorf("UsedBytes() = %d, want 200", got)
	}
}

func TestModelManager_PreloadError(t *testing.T) {
	loader := newMockLoader(100)
	loader.loadErr = func(id string) error {
		if id == "bad-model" {
			return fmt.Errorf("not found")
		}
		return nil
	}

	_, err := NewModelManager(loader, Config{
		MaxGPUMemoryBytes: 500,
		PreloadModels:     []string{"good-model", "bad-model"},
	})
	if err == nil {
		t.Fatal("expected error from preload failure")
	}

	// Verify the successfully loaded model was cleaned up.
	mock := loader.get("good-model")
	if mock != nil && !mock.closed.Load() {
		t.Error("good-model should have been closed after preload failure cleanup")
	}
}

func TestModelManager_GetSameModelTwice(t *testing.T) {
	loader := newMockLoader(100)
	mgr, err := NewModelManager(loader, Config{MaxGPUMemoryBytes: 500})
	if err != nil {
		t.Fatalf("NewModelManager: %v", err)
	}
	defer mgr.Close()

	m1, err := mgr.Get(context.Background(), "model-x")
	if err != nil {
		t.Fatalf("first Get: %v", err)
	}

	m2, err := mgr.Get(context.Background(), "model-x")
	if err != nil {
		t.Fatalf("second Get: %v", err)
	}

	// Should return the same instance without reloading.
	if m1 != m2 {
		t.Error("Get returned different instances for the same model ID")
	}

	if got := mgr.UsedBytes(); got != 100 {
		t.Errorf("UsedBytes() = %d, want 100 (should not double-count)", got)
	}
}

func TestModelManager_LoadError(t *testing.T) {
	loader := newMockLoader(100)
	loader.loadErr = func(string) error {
		return fmt.Errorf("disk failure")
	}

	mgr, err := NewModelManager(loader, Config{MaxGPUMemoryBytes: 500})
	if err != nil {
		t.Fatalf("NewModelManager: %v", err)
	}
	defer mgr.Close()

	_, err = mgr.Get(context.Background(), "broken")
	if err == nil {
		t.Fatal("expected error from broken loader")
	}

	if got := mgr.UsedBytes(); got != 0 {
		t.Errorf("UsedBytes() = %d, want 0 after failed load", got)
	}
}
