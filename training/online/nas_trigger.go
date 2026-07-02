package online

import (
	"context"
	"fmt"
	"sync"

	"github.com/zerfoo/zerfoo/training/nas"
)

// AutoNASConfig holds configuration for the automated NAS trigger.
type AutoNASConfig struct {
	// ImprovementThreshold is the minimum relative Sharpe ratio improvement
	// required to propose the new architecture (e.g. 0.05 means 5%).
	ImprovementThreshold float64

	// SearchConfig is the NAS search configuration passed to RunSignalNAS.
	SearchConfig nas.SignalSearchConfig

	// Validators are the safety validators (ADR-052) that must approve
	// the architecture replacement before it is accepted.
	Validators []Validator
}

// NASProposal represents a proposed architecture replacement discovered by
// automated NAS after a drift event.
type NASProposal struct {
	// Alert is the drift alert that triggered the NAS search.
	Alert DriftAlert

	// SearchOutput is the full NAS search output.
	SearchOutput *nas.SignalSearchOutput

	// CurrentSharpe is the Sharpe ratio of the current model at trigger time.
	CurrentSharpe float64

	// DiscoveredSharpe is the Sharpe-like metric of the discovered architecture.
	DiscoveredSharpe float64

	// Improvement is the relative improvement (discovered - current) / |current|.
	Improvement float64

	// Accepted indicates whether the proposal passed safety validation.
	Accepted bool

	// RejectionReason is non-empty if the proposal was rejected by a validator.
	RejectionReason string
}

// AutoNASTrigger listens for DriftAlert events and runs NAS search to discover
// improved architectures. When the discovered architecture's Sharpe ratio
// exceeds the current model's by at least ImprovementThreshold, it proposes
// the replacement through the safety validation pipeline.
type AutoNASTrigger struct {
	mu        sync.Mutex
	cfg       AutoNASConfig
	proposals []NASProposal
}

// NewAutoNASTrigger creates a new AutoNASTrigger with the given configuration.
func NewAutoNASTrigger(cfg AutoNASConfig) *AutoNASTrigger {
	if cfg.ImprovementThreshold <= 0 {
		cfg.ImprovementThreshold = 0.05
	}
	return &AutoNASTrigger{cfg: cfg}
}

// OnDriftAlert processes a drift alert by running NAS search and proposing
// an architecture replacement if the discovered architecture is sufficiently
// better than the current model. The currentSharpe argument is the Sharpe
// ratio of the currently deployed model.
func (t *AutoNASTrigger) OnDriftAlert(
	ctx context.Context,
	alert DriftAlert,
	currentSharpe float64,
	data nas.SignalDataProvider,
) (*NASProposal, error) {
	output, err := nas.RunSignalNAS(ctx, t.cfg.SearchConfig, data)
	if err != nil {
		return nil, fmt.Errorf("autonas: search failed: %w", err)
	}

	discoveredSharpe := output.Best.Metric

	var improvement float64
	if currentSharpe != 0 {
		improvement = (discoveredSharpe - currentSharpe) / abs(currentSharpe)
	} else if discoveredSharpe > 0 {
		// Current Sharpe is zero; any positive discovered Sharpe is infinite
		// improvement, but we cap it to signal a clear improvement.
		improvement = 1.0
	}

	proposal := NASProposal{
		Alert:            alert,
		SearchOutput:     output,
		CurrentSharpe:    currentSharpe,
		DiscoveredSharpe: discoveredSharpe,
		Improvement:      improvement,
	}

	if improvement < t.cfg.ImprovementThreshold {
		proposal.Accepted = false
		proposal.RejectionReason = fmt.Sprintf(
			"insufficient improvement: %.4f < threshold %.4f",
			improvement, t.cfg.ImprovementThreshold,
		)
		t.recordProposal(proposal)
		return &proposal, nil
	}

	// Run through safety validators (ADR-052).
	before := ModelSnapshot{Loss: 1.0 / (1.0 + currentSharpe)}
	after := ModelSnapshot{Loss: 1.0 / (1.0 + discoveredSharpe)}
	for _, v := range t.cfg.Validators {
		result := v.Validate(before, after)
		if !result.Pass {
			proposal.Accepted = false
			proposal.RejectionReason = result.Reason
			t.recordProposal(proposal)
			return &proposal, nil
		}
	}

	proposal.Accepted = true
	t.recordProposal(proposal)
	return &proposal, nil
}

// Proposals returns all recorded proposals.
func (t *AutoNASTrigger) Proposals() []NASProposal {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]NASProposal, len(t.proposals))
	copy(out, t.proposals)
	return out
}

func (t *AutoNASTrigger) recordProposal(p NASProposal) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.proposals = append(t.proposals, p)
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
