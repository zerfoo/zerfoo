package inference

import (
	"context"
	"sync"
	"sync/atomic"
)

// TransferFunc is called to begin an async CPU-to-GPU transfer of expert
// weights. The implementation should copy the expert's weight data into a GPU
// staging buffer. It receives the expert ID and returns when the transfer has
// been initiated (not necessarily completed).
type TransferFunc func(ctx context.Context, expertID int) error

// PrefetchStats tracks prediction hit/miss rates for monitoring.
type PrefetchStats struct {
	Hits   atomic.Int64
	Misses atomic.Int64
}

// HitRate returns the fraction of predictions that were correct. Returns 0 if
// no predictions have been evaluated.
func (s *PrefetchStats) HitRate() float64 {
	h := s.Hits.Load()
	m := s.Misses.Load()
	total := h + m
	if total == 0 {
		return 0
	}
	return float64(h) / float64(total)
}

// Total returns the total number of predictions evaluated.
func (s *PrefetchStats) Total() int64 {
	return s.Hits.Load() + s.Misses.Load()
}

// Reset zeroes the counters.
func (s *PrefetchStats) Reset() {
	s.Hits.Store(0)
	s.Misses.Store(0)
}

// ExpertPrefetcher predicts which experts will be needed in the next layer
// based on routing history and initiates async CPU-to-GPU weight transfers.
//
// The prediction heuristic exploits expert stickiness: tokens tend to route to
// the same experts across consecutive layers. The prefetcher records routing
// decisions per layer and predicts that the next layer will use the same set
// of experts as the current layer.
type ExpertPrefetcher struct {
	mu sync.Mutex

	// history stores the set of routed expert IDs for each layer, keyed by
	// layer index. Only the most recent routing decision per layer is kept.
	history map[int][]int

	// pending tracks expert IDs whose transfers are in flight. This prevents
	// duplicate transfers for the same expert within a single prefetch round.
	pending map[int]struct{}

	// deviceMap indicates which experts are on CPU vs GPU. Only CPU-resident
	// experts need prefetching.
	deviceMap *MoEDeviceMap

	// transfer initiates an async CPU-to-GPU transfer for a single expert.
	transfer TransferFunc

	Stats PrefetchStats
}

// NewExpertPrefetcher creates a prefetcher that uses the given device map
// to determine which experts are CPU-resident and the transfer function to
// initiate async weight uploads.
func NewExpertPrefetcher(deviceMap *MoEDeviceMap, transfer TransferFunc) *ExpertPrefetcher {
	return &ExpertPrefetcher{
		history:   make(map[int][]int),
		pending:   make(map[int]struct{}),
		deviceMap: deviceMap,
		transfer:  transfer,
	}
}

// RecordAndPrefetch records the routing decision for the given layer and
// initiates prefetch transfers for predicted next-layer experts.
//
// The prediction is simple: experts routed in layer L are predicted to also be
// routed in layer L+1 (sticky routing). Only CPU-resident experts trigger a
// transfer; GPU-resident experts are already available.
//
// Returns the list of expert IDs for which prefetch was initiated.
func (p *ExpertPrefetcher) RecordAndPrefetch(ctx context.Context, layer int, expertIDs []int) []int {
	p.mu.Lock()

	// Store current routing decision.
	ids := make([]int, len(expertIDs))
	copy(ids, expertIDs)
	p.history[layer] = ids

	// Predict next layer will use the same experts.
	predicted := p.predictNextLayer(layer)

	// Filter to CPU-resident experts that aren't already pending.
	var toPrefetch []int
	for _, eid := range predicted {
		if p.deviceMap.DeviceForExpert(eid) != CPU {
			continue
		}
		if _, ok := p.pending[eid]; ok {
			continue
		}
		p.pending[eid] = struct{}{}
		toPrefetch = append(toPrefetch, eid)
	}

	p.mu.Unlock()

	// Initiate transfers outside the lock.
	if p.transfer != nil {
		for _, eid := range toPrefetch {
			// Best-effort: ignore individual transfer errors since this is
			// speculative prefetch. The actual forward pass will handle misses.
			_ = p.transfer(ctx, eid)
		}
	}

	return toPrefetch
}

// CheckPrediction evaluates how well the prefetch prediction matched the
// actual routing decision for the given layer. Call this when the real routing
// for a layer becomes known.
//
// For each expert in actualIDs: if it was predicted (present in the previous
// layer's routing set), count a hit; otherwise count a miss.
func (p *ExpertPrefetcher) CheckPrediction(layer int, actualIDs []int) {
	p.mu.Lock()
	// The prediction for this layer was based on layer-1's routing.
	predicted := p.history[layer-1]

	// Clear pending set for evaluated experts.
	for _, eid := range actualIDs {
		delete(p.pending, eid)
	}
	p.mu.Unlock()

	predSet := make(map[int]struct{}, len(predicted))
	for _, eid := range predicted {
		predSet[eid] = struct{}{}
	}

	for _, eid := range actualIDs {
		// Only count CPU-resident experts since GPU experts don't need prefetch.
		if p.deviceMap.DeviceForExpert(eid) != CPU {
			continue
		}
		if _, ok := predSet[eid]; ok {
			p.Stats.Hits.Add(1)
		} else {
			p.Stats.Misses.Add(1)
		}
	}
}

// predictNextLayer returns the predicted expert IDs for the next layer.
// Uses sticky routing: predicts the same experts as the current layer.
func (p *ExpertPrefetcher) predictNextLayer(layer int) []int {
	// Deduplicate the current layer's routing decisions.
	seen := make(map[int]struct{})
	var predicted []int
	for _, eid := range p.history[layer] {
		if _, ok := seen[eid]; !ok {
			seen[eid] = struct{}{}
			predicted = append(predicted, eid)
		}
	}
	return predicted
}

// ClearHistory removes all stored routing history. Useful between sequences.
func (p *ExpertPrefetcher) ClearHistory() {
	p.mu.Lock()
	p.history = make(map[int][]int)
	p.pending = make(map[int]struct{})
	p.mu.Unlock()
}
