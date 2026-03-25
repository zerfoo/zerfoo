package cloud

import (
	"container/list"
	"errors"
	"sync"
	"time"
)

var (
	errZeroBudget     = errors.New("cloud: VRAM budget must be positive")
	errModelNotLoaded = errors.New("cloud: model not loaded")
	errModelTooLarge  = errors.New("cloud: model exceeds total VRAM budget")
)

// ModelInfo describes a loaded model tracked by the ResourceManager.
type ModelInfo struct {
	ModelID    string
	VRAMBytes uint64
	LoadedAt  time.Time
	LastUsed  time.Time
}

// ResourceManager tracks loaded models and their VRAM usage, evicting
// least-recently-used models when a new load would exceed the memory budget.
type ResourceManager struct {
	mu        sync.Mutex
	budget    uint64
	used      uint64
	models    map[string]*list.Element // model ID -> list element
	lru       *list.List               // front = most recently used, back = least recently used
	onEvict   func(modelID string)     // optional callback on eviction
}

// NewResourceManager creates a ResourceManager with the given VRAM budget in bytes.
func NewResourceManager(budgetBytes uint64) (*ResourceManager, error) {
	if budgetBytes == 0 {
		return nil, errZeroBudget
	}
	return &ResourceManager{
		budget: budgetBytes,
		models: make(map[string]*list.Element),
		lru:    list.New(),
	}, nil
}

// SetEvictCallback sets an optional function called when a model is evicted.
func (rm *ResourceManager) SetEvictCallback(fn func(modelID string)) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	rm.onEvict = fn
}

// Load registers a model with the given VRAM footprint. If loading would exceed
// the budget, LRU models are evicted until there is enough space. Returns an
// error if the model alone exceeds the entire budget.
func (rm *ResourceManager) Load(modelID string, vramBytes uint64) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	// If already loaded, treat as a touch.
	if elem, ok := rm.models[modelID]; ok {
		rm.lru.MoveToFront(elem)
		elem.Value.(*ModelInfo).LastUsed = time.Now()
		return nil
	}

	if vramBytes > rm.budget {
		return errModelTooLarge
	}

	// Evict LRU models until space is available.
	for rm.used+vramBytes > rm.budget {
		rm.evictLRU()
	}

	now := time.Now()
	info := &ModelInfo{
		ModelID:   modelID,
		VRAMBytes: vramBytes,
		LoadedAt:  now,
		LastUsed:  now,
	}
	elem := rm.lru.PushFront(info)
	rm.models[modelID] = elem
	rm.used += vramBytes
	return nil
}

// Touch updates the last-used time for a model, moving it to the front of the
// LRU list. Call this on each inference request.
func (rm *ResourceManager) Touch(modelID string) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	elem, ok := rm.models[modelID]
	if !ok {
		return errModelNotLoaded
	}
	rm.lru.MoveToFront(elem)
	elem.Value.(*ModelInfo).LastUsed = time.Now()
	return nil
}

// Evict explicitly removes a model from the manager.
func (rm *ResourceManager) Evict(modelID string) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	elem, ok := rm.models[modelID]
	if !ok {
		return errModelNotLoaded
	}
	rm.removeLocked(elem)
	return nil
}

// Stats returns the current memory usage statistics.
func (rm *ResourceManager) Stats() (used, budget uint64, loaded int) {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	return rm.used, rm.budget, rm.lru.Len()
}

// LoadedModels returns a snapshot of all currently loaded models.
func (rm *ResourceManager) LoadedModels() []ModelInfo {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	out := make([]ModelInfo, 0, rm.lru.Len())
	for e := rm.lru.Front(); e != nil; e = e.Next() {
		out = append(out, *e.Value.(*ModelInfo))
	}
	return out
}

// evictLRU removes the least recently used model. Caller must hold rm.mu.
func (rm *ResourceManager) evictLRU() {
	back := rm.lru.Back()
	if back == nil {
		return
	}
	rm.removeLocked(back)
}

// removeLocked removes an element from the LRU list and bookkeeping. Caller must hold rm.mu.
func (rm *ResourceManager) removeLocked(elem *list.Element) {
	info := rm.lru.Remove(elem).(*ModelInfo)
	delete(rm.models, info.ModelID)
	rm.used -= info.VRAMBytes
	if rm.onEvict != nil {
		rm.onEvict(info.ModelID)
	}
}
