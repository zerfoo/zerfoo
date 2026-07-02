package registry

import (
	"errors"
	"hash/fnv"
	"sync"
	"sync/atomic"
)

// ABConfig configures champion-challenger A/B routing.
type ABConfig struct {
	ChampionID      string
	ChallengerID    string
	ChallengerWeight float64 // 0.0–1.0
}

// ABStats holds request counters for each variant.
type ABStats struct {
	ChampionRequests   int64
	ChallengerRequests int64
}

// ABRouter routes requests between a champion and challenger model using
// deterministic hashing for consistent assignment.
type ABRouter struct {
	championID   string
	challengerID string
	// weight is stored as an int64 representing ChallengerWeight * 1000,
	// enabling lock-free atomic updates.
	weight atomic.Int64

	mu               sync.Mutex
	championCounter  int64
	challengerCounter int64
}

// NewABRouter creates an ABRouter with the given configuration.
func NewABRouter(cfg ABConfig) *ABRouter {
	r := &ABRouter{
		championID:   cfg.ChampionID,
		challengerID: cfg.ChallengerID,
	}
	r.weight.Store(int64(cfg.ChallengerWeight * 1000))
	return r
}

// Route returns the model ID to use for the given request. The assignment is
// deterministic: the same requestID always maps to the same model.
func (r *ABRouter) Route(requestID string) string {
	h := fnv.New64a()
	h.Write([]byte(requestID))
	bucket := int64(h.Sum64() % 1000)

	w := r.weight.Load()
	if bucket < w {
		r.mu.Lock()
		r.challengerCounter++
		r.mu.Unlock()
		return r.challengerID
	}
	r.mu.Lock()
	r.championCounter++
	r.mu.Unlock()
	return r.championID
}

// UpdateWeights atomically updates the challenger traffic weight.
// It returns an error if weight is not in [0, 1].
func (r *ABRouter) UpdateWeights(challengerWeight float64) error {
	if challengerWeight < 0 || challengerWeight > 1 {
		return errors.New("registry: challenger weight must be between 0.0 and 1.0")
	}
	r.weight.Store(int64(challengerWeight * 1000))
	return nil
}

// Stats returns the current request counters.
func (r *ABRouter) Stats() ABStats {
	r.mu.Lock()
	defer r.mu.Unlock()
	return ABStats{
		ChampionRequests:   r.championCounter,
		ChallengerRequests: r.challengerCounter,
	}
}
