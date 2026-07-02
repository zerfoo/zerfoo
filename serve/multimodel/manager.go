// Package multimodel provides a ModelManager that loads and unloads models
// on demand with LRU eviction when GPU memory budget is exceeded.
package multimodel

import (
	"container/list"
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
)

// ModelLoader loads a model by ID and returns its handle and estimated size in bytes.
type ModelLoader interface {
	Load(ctx context.Context, modelID string) (io.Closer, int64, error)
}

// Config controls the ModelManager behavior.
type Config struct {
	// MaxGPUMemoryBytes is the memory budget. When loading a new model would
	// exceed this limit, the least-recently-used model is evicted first.
	MaxGPUMemoryBytes int64

	// PreloadModels lists model IDs to load eagerly at creation time.
	PreloadModels []string
}

// entry tracks a loaded model and its position in the LRU list.
type entry struct {
	id    string
	model io.Closer
	size  int64
}

// ModelManager manages a set of loaded models with LRU GPU eviction.
// It is safe for concurrent use.
type ModelManager struct {
	mu     sync.Mutex
	loader ModelLoader
	config Config

	// models maps model ID to its LRU list element (value is *entry).
	models map[string]*list.Element

	// lru tracks access order; front is most-recently used.
	lru *list.List

	// usedBytes is the sum of sizes of all loaded models.
	usedBytes int64
}

// NewModelManager creates a ModelManager. If cfg.PreloadModels is non-empty,
// those models are loaded eagerly. An error is returned if any preload fails.
func NewModelManager(loader ModelLoader, cfg Config) (*ModelManager, error) {
	m := &ModelManager{
		loader: loader,
		config: cfg,
		models: make(map[string]*list.Element),
		lru:    list.New(),
	}

	for _, id := range cfg.PreloadModels {
		if _, err := m.Get(context.Background(), id); err != nil {
			// Clean up already-loaded models on failure.
			m.Close()
			return nil, fmt.Errorf("preload %q: %w", id, err)
		}
	}
	return m, nil
}

// Get returns the model for the given ID, loading it if necessary.
// If loading would exceed the memory budget, the least-recently-used
// model is evicted first. Get is safe for concurrent callers.
func (m *ModelManager) Get(ctx context.Context, modelID string) (io.Closer, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Fast path: model already loaded — promote to front of LRU.
	if elem, ok := m.models[modelID]; ok {
		m.lru.MoveToFront(elem)
		return elem.Value.(*entry).model, nil
	}

	// Load the model (unlock not needed; loader may be slow but we hold
	// the lock to keep eviction consistent — callers wanting parallelism
	// should use separate managers or a singleflight wrapper).
	model, size, err := m.loader.Load(ctx, modelID)
	if err != nil {
		return nil, fmt.Errorf("load model %q: %w", modelID, err)
	}

	// Evict LRU models until there is room.
	for m.usedBytes+size > m.config.MaxGPUMemoryBytes && m.lru.Len() > 0 {
		m.evictLRU()
	}

	e := &entry{id: modelID, model: model, size: size}
	elem := m.lru.PushFront(e)
	m.models[modelID] = elem
	m.usedBytes += size
	return model, nil
}

// Unload explicitly removes a model by ID, freeing its resources.
func (m *ModelManager) Unload(modelID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	elem, ok := m.models[modelID]
	if !ok {
		return fmt.Errorf("model %q not loaded", modelID)
	}
	return m.removeLocked(elem)
}

// Loaded returns the IDs of all currently loaded models.
func (m *ModelManager) Loaded() []string {
	m.mu.Lock()
	defer m.mu.Unlock()

	ids := make([]string, 0, len(m.models))
	for id := range m.models {
		ids = append(ids, id)
	}
	return ids
}

// UsedBytes returns the current estimated GPU memory usage.
func (m *ModelManager) UsedBytes() int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.usedBytes
}

// Close unloads all models and releases resources.
func (m *ModelManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var errs []error
	for m.lru.Len() > 0 {
		if err := m.evictLRU(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// evictLRU removes the least-recently-used model. Must be called with mu held.
func (m *ModelManager) evictLRU() error {
	back := m.lru.Back()
	if back == nil {
		return nil
	}
	return m.removeLocked(back)
}

// removeLocked removes the given element from LRU and models map.
// Must be called with mu held.
func (m *ModelManager) removeLocked(elem *list.Element) error {
	e := elem.Value.(*entry)
	m.lru.Remove(elem)
	delete(m.models, e.id)
	m.usedBytes -= e.size
	return e.model.Close()
}
