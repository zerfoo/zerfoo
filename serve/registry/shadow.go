package registry

import (
	"context"
	"log/slog"
	"math/rand/v2"
	"sync"
	"sync/atomic"
	"time"
)

// ShadowConfig configures shadow mode inference where a challenger model
// runs alongside the champion model on a sample of requests.
type ShadowConfig struct {
	ChampionID   string
	ChallengerID string
	SampleRate   float64 // 0.0–1.0: fraction of requests that also run the challenger.
}

// ShadowResult holds the output of a shadow inference run.
type ShadowResult struct {
	ChampionOutput   []float32
	ChallengerOutput []float32
	LatencyDelta     time.Duration // challenger_latency - champion_latency
	Sampled          bool
}

// ShadowMetrics tracks aggregate shadow inference counters.
type ShadowMetrics struct {
	TotalRequests    int64
	SampledRequests  int64
	ChallengerErrors int64
}

// ShadowRunner executes shadow mode inference.
type ShadowRunner struct {
	cfg  ShadowConfig
	rand func() float64 // injectable for testing

	mu               sync.Mutex
	totalRequests    atomic.Int64
	sampledRequests  atomic.Int64
	challengerErrors atomic.Int64
}

// NewShadowRunner creates a ShadowRunner with the given configuration.
func NewShadowRunner(cfg ShadowConfig) *ShadowRunner {
	return &ShadowRunner{
		cfg:  cfg,
		rand: rand.Float64,
	}
}

// RunShadow executes the champion model and optionally the challenger model
// concurrently. The champion result is always returned. If the challenger
// errors, the error is logged but the champion result is still returned.
func (s *ShadowRunner) RunShadow(
	ctx context.Context,
	input []float32,
	inferFn func(modelID string, input []float32) ([]float32, error),
) (*ShadowResult, error) {
	s.totalRequests.Add(1)

	sampled := s.rand() < s.cfg.SampleRate

	// Always run the champion.
	championStart := time.Now()
	championOutput, err := inferFn(s.cfg.ChampionID, input)
	championLatency := time.Since(championStart)
	if err != nil {
		return nil, err
	}

	result := &ShadowResult{
		ChampionOutput: championOutput,
		Sampled:        sampled,
	}

	if !sampled {
		return result, nil
	}

	s.sampledRequests.Add(1)

	// Run challenger concurrently — we already have the champion result,
	// so we just need to measure challenger latency.
	type challengerResult struct {
		output  []float32
		latency time.Duration
		err     error
	}
	ch := make(chan challengerResult, 1)
	go func() {
		start := time.Now()
		out, cerr := inferFn(s.cfg.ChallengerID, input)
		ch <- challengerResult{output: out, latency: time.Since(start), err: cerr}
	}()

	select {
	case <-ctx.Done():
		return result, nil
	case cr := <-ch:
		if cr.err != nil {
			s.challengerErrors.Add(1)
			slog.Warn("shadow challenger error",
				"challenger_id", s.cfg.ChallengerID,
				"error", cr.err,
			)
			return result, nil
		}
		result.ChallengerOutput = cr.output
		result.LatencyDelta = cr.latency - championLatency
	}

	return result, nil
}

// Metrics returns aggregate shadow inference counters.
func (s *ShadowRunner) Metrics() ShadowMetrics {
	return ShadowMetrics{
		TotalRequests:    s.totalRequests.Load(),
		SampledRequests:  s.sampledRequests.Load(),
		ChallengerErrors: s.challengerErrors.Load(),
	}
}
