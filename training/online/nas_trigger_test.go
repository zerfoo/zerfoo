package online

import (
	"context"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/training/nas"
)

// syntheticDataProvider returns deterministic synthetic data for NAS search.
type syntheticDataProvider struct {
	inputFeatures int
	patchLen      int
	horizonLen    int
}

func (s *syntheticDataProvider) TrainBatch() (input, target []float32, shape []int, err error) {
	n := s.inputFeatures * s.patchLen
	shape = []int{1, n}
	input = make([]float32, n)
	target = make([]float32, n)
	for i := range n {
		input[i] = float32(i) * 0.01
		target[i] = float32(i) * 0.01
	}
	return input, target, shape, nil
}

func (s *syntheticDataProvider) ValBatch() (input, target []float32, shape []int, err error) {
	return s.TrainBatch()
}

func TestAutoNASTrigger(t *testing.T) {
	tests := []struct {
		name              string
		currentSharpe     float64
		improvementThresh float64
		validators        []Validator
		wantAccepted      bool
		wantHasProposal   bool
	}{
		{
			name:              "drift triggers NAS and discovers improvement",
			currentSharpe:     0.3,
			improvementThresh: 0.05,
			validators:        nil,
			wantAccepted:      true,
			wantHasProposal:   true,
		},
		{
			name:              "improvement below threshold is rejected",
			currentSharpe:     0.95,
			improvementThresh: 0.50,
			validators:        nil,
			wantAccepted:      false,
			wantHasProposal:   true,
		},
		{
			name:              "validator rejects proposal",
			currentSharpe:     0.3,
			improvementThresh: 0.05,
			validators: []Validator{
				&rejectAllValidator{},
			},
			wantAccepted:    false,
			wantHasProposal: true,
		},
		{
			name:              "zero current sharpe with positive discovered",
			currentSharpe:     0.0,
			improvementThresh: 0.05,
			validators:        nil,
			wantAccepted:      true,
			wantHasProposal:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			searchCfg := nas.SignalSearchConfig{
				NumTrials:     1,
				SearchSteps:   2,
				WeightLR:      0.01,
				AlphaLR:       0.01,
				InputFeatures: 2,
				PatchLen:      4,
				HorizonLen:    2,
				HiddenDim:     8,
				NumLayers:     1,
				Seed:          42,
			}

			trigger := NewAutoNASTrigger(AutoNASConfig{
				ImprovementThreshold: tt.improvementThresh,
				SearchConfig:         searchCfg,
				Validators:           tt.validators,
			})

			alert := DriftAlert{
				Timestamp:     time.Date(2026, 3, 17, 0, 0, 0, 0, time.UTC),
				CurrentSharpe: tt.currentSharpe,
				MeanSharpe:    1.5,
				Threshold:     1.0,
				WindowSize:    30,
			}

			data := &syntheticDataProvider{
				inputFeatures: 2,
				patchLen:      4,
				horizonLen:    2,
			}

			ctx := context.Background()
			proposal, err := trigger.OnDriftAlert(ctx, alert, tt.currentSharpe, data)
			if err != nil {
				t.Fatalf("OnDriftAlert() error: %v", err)
			}

			if tt.wantHasProposal && proposal == nil {
				t.Fatal("expected proposal, got nil")
			}
			if !tt.wantHasProposal && proposal != nil {
				t.Fatalf("expected no proposal, got %+v", proposal)
			}
			if proposal == nil {
				return
			}

			if proposal.Accepted != tt.wantAccepted {
				t.Errorf("proposal.Accepted = %v, want %v (reason: %s)",
					proposal.Accepted, tt.wantAccepted, proposal.RejectionReason)
			}

			if proposal.SearchOutput == nil {
				t.Error("proposal.SearchOutput is nil")
			}
			if proposal.DiscoveredSharpe <= 0 {
				t.Errorf("proposal.DiscoveredSharpe = %f, want > 0", proposal.DiscoveredSharpe)
			}

			// Verify proposal was recorded.
			proposals := trigger.Proposals()
			if len(proposals) != 1 {
				t.Errorf("Proposals() returned %d, want 1", len(proposals))
			}
		})
	}
}

func TestAutoNASTrigger_EndToEnd(t *testing.T) {
	// Simulate the full pipeline: stable P&L -> degradation -> drift alert ->
	// NAS search -> improvement detection.
	dd := NewDriftDetector(DriftConfig{WindowSize: 30})
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	searchCfg := nas.SignalSearchConfig{
		NumTrials:     1,
		SearchSteps:   2,
		WeightLR:      0.01,
		AlphaLR:       0.01,
		InputFeatures: 2,
		PatchLen:      4,
		HorizonLen:    2,
		HiddenDim:     8,
		NumLayers:     1,
		Seed:          42,
	}

	trigger := NewAutoNASTrigger(AutoNASConfig{
		ImprovementThreshold: 0.05,
		SearchConfig:         searchCfg,
		Validators: []Validator{
			NewLossDeltaValidator(5.0),
		},
	})

	data := &syntheticDataProvider{
		inputFeatures: 2,
		patchLen:      4,
		horizonLen:    2,
	}

	day := 0

	// Phase 1: 90 days of steadily increasing P&L.
	for ; day < 90; day++ {
		pnl := 100.0 + float64(day)*1.0
		dd.Observe(base.Add(time.Duration(day)*24*time.Hour), pnl)
	}

	// Phase 2: inject degradation.
	var gotAlert *DriftAlert
	for ; day < 150; day++ {
		pnl := -50.0 + float64(day%5)*10.0
		alert := dd.Observe(base.Add(time.Duration(day)*24*time.Hour), pnl)
		if alert != nil {
			gotAlert = alert
			break
		}
	}

	if gotAlert == nil {
		t.Fatal("expected drift alert after degradation, got none")
	}

	// Use the degraded Sharpe as the current model's Sharpe.
	currentSharpe := gotAlert.CurrentSharpe

	ctx := context.Background()
	proposal, err := trigger.OnDriftAlert(ctx, *gotAlert, currentSharpe, data)
	if err != nil {
		t.Fatalf("OnDriftAlert() error: %v", err)
	}

	if proposal == nil {
		t.Fatal("expected proposal, got nil")
	}

	// The NAS search should discover an architecture with a positive
	// Sharpe-like metric, which is better than the degraded current Sharpe.
	if proposal.DiscoveredSharpe <= 0 {
		t.Errorf("discovered Sharpe = %f, want > 0", proposal.DiscoveredSharpe)
	}

	if proposal.Improvement <= 0 {
		t.Errorf("improvement = %f, want > 0", proposal.Improvement)
	}

	if !proposal.Accepted {
		t.Errorf("expected proposal to be accepted, rejected with: %s",
			proposal.RejectionReason)
	}

	// Verify the alert propagated correctly.
	if proposal.Alert.WindowSize != 30 {
		t.Errorf("proposal.Alert.WindowSize = %d, want 30", proposal.Alert.WindowSize)
	}
}

// rejectAllValidator always rejects for testing.
type rejectAllValidator struct{}

func (r *rejectAllValidator) Validate(_, _ ModelSnapshot) ValidationResult {
	return ValidationResult{
		Pass:   false,
		Reason: "rejected by safety validator",
	}
}
