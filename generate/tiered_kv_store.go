package generate

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// Tier identifies which storage tier a layer's KV data resides in.
type Tier int

const (
	// TierHot stores recent tokens uncompressed in memory (GPU or CPU).
	TierHot Tier = iota
	// TierWarm stores compressed tokens in CPU memory via CompressedKVCache.
	TierWarm
	// TierCold stores evicted tokens on disk.
	TierCold
)

// tieredLayerState tracks per-layer access counts and current tier.
type tieredLayerState struct {
	tier        Tier
	accessCount int
}

// TieredKVStore provides multi-tier KV cache storage with automatic
// promotion and demotion based on access patterns.
//
// Hot tier: recent tokens stored uncompressed (KVCache).
// Warm tier: compressed tokens in CPU memory (CompressedKVCache).
// Cold tier: evicted tokens serialized to disk.
//
// Tokens are initially stored in the hot tier. When a layer's access
// count drops below demoteThreshold, it is moved to the warm tier
// (compressed). If demoted again, data moves to the cold tier (disk).
// Accessing a cold or warm layer promotes it back toward the hot tier.
type TieredKVStore[T tensor.Numeric] struct {
	mu sync.RWMutex // protects all tier state for concurrent serve access

	hot  *KVCache[T]
	warm *CompressedKVCache[T]

	engine compute.Engine[T]

	layerStates []tieredLayerState
	numLayers   int
	maxSeqLen   int
	chunkSize   int

	// demoteThreshold: layers with accessCount below this on ManageTiers
	// are candidates for demotion.
	demoteThreshold int
	// promoteThreshold: layers with accessCount above this on ManageTiers
	// are candidates for promotion.
	promoteThreshold int

	// coldDir is the directory for cold-tier files.
	// isTempColdDir is true when coldDir was created by NewTieredKVStore via
	// os.MkdirTemp (i.e., cfg.ColdDir was empty). Close() only removes the
	// directory when this flag is true; user-provided directories are left intact.
	coldDir      string
	isTempColdDir bool
	coldMu       sync.Mutex

	// Async prefetch state.
	prefetchCh     chan prefetchRequest[T]
	prefetchCancel context.CancelFunc
	prefetchWg     sync.WaitGroup
	prefetchMu     sync.Mutex
	prefetched     map[int]*LayerKV[T] // layer → prefetched data
}

// prefetchRequest represents a request to prefetch a layer from a cold tier.
type prefetchRequest[T tensor.Numeric] struct {
	layer int
	store *TieredKVStore[T]
}

// TieredKVStoreConfig holds configuration for a TieredKVStore.
type TieredKVStoreConfig struct {
	NumLayers        int
	MaxSeqLen        int
	ChunkSize        int // compression chunk size for warm tier
	DemoteThreshold  int // access count below which layers are demoted
	PromoteThreshold int // access count above which layers are promoted
	ColdDir          string
}

// NewTieredKVStore creates a TieredKVStore. All layers start in the hot tier.
// If cfg.ColdDir is empty, a temporary directory is created.
func NewTieredKVStore[T tensor.Numeric](engine compute.Engine[T], cfg TieredKVStoreConfig) (*TieredKVStore[T], error) {
	if cfg.ChunkSize <= 0 {
		cfg.ChunkSize = 64
	}
	if cfg.DemoteThreshold <= 0 {
		cfg.DemoteThreshold = 2
	}
	if cfg.PromoteThreshold <= 0 {
		cfg.PromoteThreshold = 5
	}

	isTempColdDir := cfg.ColdDir == ""
	coldDir := cfg.ColdDir
	if isTempColdDir {
		var err error
		coldDir, err = os.MkdirTemp("", "tiered-kv-cold-*")
		if err != nil {
			return nil, fmt.Errorf("creating cold tier dir: %w", err)
		}
	}

	states := make([]tieredLayerState, cfg.NumLayers)
	for i := range states {
		states[i].tier = TierHot
	}

	ctx, cancel := context.WithCancel(context.Background())
	s := &TieredKVStore[T]{
		hot:              NewKVCache[T](cfg.NumLayers, cfg.MaxSeqLen),
		warm:             NewCompressedKVCache[T](engine, cfg.NumLayers, 1, 1, cfg.ChunkSize),
		engine:           engine,
		layerStates:      states,
		numLayers:        cfg.NumLayers,
		maxSeqLen:        cfg.MaxSeqLen,
		chunkSize:        cfg.ChunkSize,
		demoteThreshold:  cfg.DemoteThreshold,
		promoteThreshold: cfg.PromoteThreshold,
		coldDir:           coldDir,
		isTempColdDir:     isTempColdDir,
		prefetchCh:        make(chan prefetchRequest[T], cfg.NumLayers),
		prefetchCancel:   cancel,
		prefetched:       make(map[int]*LayerKV[T]),
	}
	s.prefetchWg.Add(1)
	go s.prefetchLoop(ctx)
	return s, nil
}

// NumLayers returns the number of layers in the store.
func (s *TieredKVStore[T]) NumLayers() int {
	return s.numLayers
}

// Tier returns the current storage tier for the given layer.
func (s *TieredKVStore[T]) Tier(layer int) Tier {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if layer < 0 || layer >= s.numLayers {
		return TierHot
	}
	return s.layerStates[layer].tier
}

// Update appends new key and value tensors for the given layer.
// Data is always written to the hot tier. If the layer was in a lower tier,
// it is promoted to hot first.
func (s *TieredKVStore[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if layer < 0 || layer >= s.numLayers {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, s.numLayers)
	}

	// Promote to hot if needed — new data always goes to hot tier.
	st := &s.layerStates[layer]
	if st.tier != TierHot {
		if err := s.promote(layer, TierHot); err != nil {
			return fmt.Errorf("promoting layer %d to hot: %w", layer, err)
		}
	}

	st.accessCount++
	return s.hot.Update(layer, newK, newV)
}

// Get returns the cached key-value pair for the given layer.
// The data is retrieved from whichever tier the layer currently resides in.
// Accessing a layer increments its access count.
func (s *TieredKVStore[T]) Get(layer int) (*LayerKV[T], bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if layer < 0 || layer >= s.numLayers {
		return nil, false
	}

	st := &s.layerStates[layer]
	st.accessCount++

	switch st.tier {
	case TierHot:
		return s.hot.Get(layer)
	case TierWarm:
		return s.warm.Get(layer)
	case TierCold:
		return s.getCold(layer)
	}
	return nil, false
}

// SeqLen returns the current sequence length from the hot tier.
func (s *TieredKVStore[T]) SeqLen() int {
	return s.hot.SeqLen()
}

// Reset clears all tiers, resets access counts, and drains prefetched data.
func (s *TieredKVStore[T]) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hot.Reset()
	s.warm.Reset()
	s.clearColdDir()
	s.prefetchMu.Lock()
	s.prefetched = make(map[int]*LayerKV[T])
	s.prefetchMu.Unlock()
	for i := range s.layerStates {
		s.layerStates[i].tier = TierHot
		s.layerStates[i].accessCount = 0
	}
}

// Truncate reduces the hot tier cache to the given sequence length.
func (s *TieredKVStore[T]) Truncate(newSeqLen int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hot.Truncate(newSeqLen)
}

// ManageTiers evaluates access patterns and moves layers between tiers.
// Layers with low access counts are demoted; layers with high access counts
// are promoted. Access counts are reset after management.
func (s *TieredKVStore[T]) ManageTiers() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.layerStates {
		st := &s.layerStates[i]
		switch {
		case st.accessCount < s.demoteThreshold && st.tier < TierCold:
			if err := s.demote(i); err != nil {
				return fmt.Errorf("demoting layer %d: %w", i, err)
			}
		case st.accessCount >= s.promoteThreshold && st.tier > TierHot:
			if err := s.promote(i, st.tier-1); err != nil {
				return fmt.Errorf("promoting layer %d: %w", i, err)
			}
		}
		st.accessCount = 0
	}
	return nil
}

// Demote moves a layer one tier down (hot→warm, warm→cold).
func (s *TieredKVStore[T]) Demote(layer int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if layer < 0 || layer >= s.numLayers {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, s.numLayers)
	}
	return s.demote(layer)
}

// Promote moves a layer one tier up (cold→warm, warm→hot).
func (s *TieredKVStore[T]) Promote(layer int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if layer < 0 || layer >= s.numLayers {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, s.numLayers)
	}
	st := &s.layerStates[layer]
	if st.tier == TierHot {
		return nil
	}
	return s.promote(layer, st.tier-1)
}

func (s *TieredKVStore[T]) demote(layer int) error {
	st := &s.layerStates[layer]

	switch st.tier {
	case TierHot:
		// Hot → Warm: get data from hot, update warm.
		lkv, ok := s.hot.Get(layer)
		if !ok {
			// No data to demote; just change tier.
			st.tier = TierWarm
			return nil
		}
		if err := s.warm.Update(layer, lkv.Key, lkv.Value); err != nil {
			return err
		}
		st.tier = TierWarm

	case TierWarm:
		// Warm → Cold: get data from warm, write to disk.
		lkv, ok := s.warm.Get(layer)
		if !ok {
			st.tier = TierCold
			return nil
		}
		if err := s.writeCold(layer, lkv); err != nil {
			return err
		}
		st.tier = TierCold

	case TierCold:
		// Already at coldest tier.
	}
	return nil
}

func (s *TieredKVStore[T]) promote(layer int, targetTier Tier) error {
	st := &s.layerStates[layer]

	for st.tier > targetTier {
		switch st.tier {
		case TierCold:
			// Cold → Warm: read from disk, load into warm.
			lkv, ok := s.getCold(layer)
			if ok {
				if err := s.warm.Update(layer, lkv.Key, lkv.Value); err != nil {
					return err
				}
			}
			s.removeColdFile(layer)
			st.tier = TierWarm

		case TierWarm:
			// Warm → Hot: get from warm, load into hot.
			lkv, ok := s.warm.Get(layer)
			if ok {
				if err := s.hot.Update(layer, lkv.Key, lkv.Value); err != nil {
					return err
				}
			}
			st.tier = TierHot
		}
	}
	return nil
}

// coldFilePath returns the file path for a cold-tier layer.
func (s *TieredKVStore[T]) coldFilePath(layer int) string {
	return filepath.Join(s.coldDir, fmt.Sprintf("layer_%d.bin", layer))
}

// writeCold serializes a LayerKV to disk as: [4 bytes batch][4 bytes seq][4 bytes dim][key float64s][val float64s].
func (s *TieredKVStore[T]) writeCold(layer int, lkv *LayerKV[T]) error {
	s.coldMu.Lock()
	defer s.coldMu.Unlock()

	shape := lkv.Key.Shape()
	kData := lkv.Key.Data()
	vData := lkv.Value.Data()

	f, err := os.Create(s.coldFilePath(layer))
	if err != nil {
		return err
	}
	defer f.Close()

	// Write shape: batch, seq, dim as uint32.
	header := [3]uint32{uint32(shape[0]), uint32(shape[1]), uint32(shape[2])}
	if err := binary.Write(f, binary.LittleEndian, header); err != nil {
		return err
	}

	// Write key and value data as float64 for lossless storage of any Numeric type.
	for _, v := range kData {
		if err := binary.Write(f, binary.LittleEndian, float64(v)); err != nil {
			return err
		}
	}
	for _, v := range vData {
		if err := binary.Write(f, binary.LittleEndian, float64(v)); err != nil {
			return err
		}
	}
	return nil
}

// getCold reads a LayerKV from disk.
func (s *TieredKVStore[T]) getCold(layer int) (*LayerKV[T], bool) {
	s.coldMu.Lock()
	defer s.coldMu.Unlock()

	f, err := os.Open(s.coldFilePath(layer))
	if err != nil {
		return nil, false
	}
	defer f.Close()

	var header [3]uint32
	if err := binary.Read(f, binary.LittleEndian, &header); err != nil {
		return nil, false
	}

	batch, seq, dim := int(header[0]), int(header[1]), int(header[2])
	size := batch * seq * dim

	kData := make([]T, size)
	vData := make([]T, size)

	for i := range size {
		var v float64
		if err := binary.Read(f, binary.LittleEndian, &v); err != nil {
			return nil, false
		}
		kData[i] = T(v)
	}
	for i := range size {
		var v float64
		if err := binary.Read(f, binary.LittleEndian, &v); err != nil {
			return nil, false
		}
		vData[i] = T(v)
	}

	shape := []int{batch, seq, dim}
	keyT, err := tensor.New(shape, kData)
	if err != nil {
		return nil, false
	}
	valT, err := tensor.New(shape, vData)
	if err != nil {
		return nil, false
	}

	return &LayerKV[T]{Key: keyT, Value: valT}, true
}

func (s *TieredKVStore[T]) removeColdFile(layer int) {
	s.coldMu.Lock()
	defer s.coldMu.Unlock()
	os.Remove(s.coldFilePath(layer))
}

func (s *TieredKVStore[T]) clearColdDir() {
	s.coldMu.Lock()
	defer s.coldMu.Unlock()
	entries, err := os.ReadDir(s.coldDir)
	if err != nil {
		return
	}
	for _, e := range entries {
		os.Remove(filepath.Join(s.coldDir, e.Name()))
	}
}

// PrefetchAsync queues the given layer positions for asynchronous prefetch
// from cold/warm tiers to the hot tier. Layers already in the hot tier or
// already prefetched are skipped. The caller continues execution immediately.
func (s *TieredKVStore[T]) PrefetchAsync(positions []int) {
	for _, layer := range positions {
		if layer < 0 || layer >= s.numLayers {
			continue
		}
		st := &s.layerStates[layer]
		if st.tier == TierHot {
			continue
		}
		s.prefetchMu.Lock()
		_, already := s.prefetched[layer]
		s.prefetchMu.Unlock()
		if already {
			continue
		}
		// Non-blocking send; drop if channel is full.
		select {
		case s.prefetchCh <- prefetchRequest[T]{layer: layer, store: s}:
		default:
		}
	}
}

// GetPrefetched returns a previously prefetched LayerKV for the given layer
// and removes it from the prefetch cache. Returns nil, false if no prefetched
// data is available.
func (s *TieredKVStore[T]) GetPrefetched(layer int) (*LayerKV[T], bool) {
	s.prefetchMu.Lock()
	defer s.prefetchMu.Unlock()
	lkv, ok := s.prefetched[layer]
	if ok {
		delete(s.prefetched, layer)
	}
	return lkv, ok
}

// prefetchLoop runs in a background goroutine, processing prefetch requests.
func (s *TieredKVStore[T]) prefetchLoop(ctx context.Context) {
	defer s.prefetchWg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case req, ok := <-s.prefetchCh:
			if !ok {
				return
			}
			st := &req.store.layerStates[req.layer]
			if st.tier == TierHot {
				continue
			}
			// Fetch the data from the current tier.
			var lkv *LayerKV[T]
			switch st.tier {
			case TierWarm:
				data, ok := req.store.warm.Get(req.layer)
				if ok {
					lkv = data
				}
			case TierCold:
				data, ok := req.store.getCold(req.layer)
				if ok {
					lkv = data
				}
			}
			if lkv != nil {
				req.store.prefetchMu.Lock()
				req.store.prefetched[req.layer] = lkv
				req.store.prefetchMu.Unlock()
			}
		}
	}
}

// Close stops the prefetch goroutine and cleans up cold-tier files.
// When the cold directory was created by NewTieredKVStore (cfg.ColdDir was
// empty), Close removes it. When the caller supplied their own ColdDir,
// Close leaves the directory untouched so the caller can reuse it across
// generation calls.
func (s *TieredKVStore[T]) Close() error {
	s.prefetchCancel()
	s.prefetchWg.Wait()
	if s.isTempColdDir {
		s.clearColdDir()
		return os.Remove(s.coldDir)
	}
	return nil
}

// AccessCount returns the current access count for a layer.
func (s *TieredKVStore[T]) AccessCount(layer int) int {
	if layer < 0 || layer >= s.numLayers {
		return 0
	}
	return s.layerStates[layer].accessCount
}

