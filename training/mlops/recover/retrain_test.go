package recover

import (
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/training/mlops/monitor"
)

// mockDetector returns true after a set number of observations.
type mockDetector struct {
	count     int
	threshold int
}

func (m *mockDetector) Observe(float64) bool {
	m.count++
	return m.count >= m.threshold
}

func TestAutoRetrain_Pipeline(t *testing.T) {
	tests := []struct {
		name        string
		rollbackErr error
		retrainErr  error
		validateErr error
		redeployErr error
		wantPhase   Phase
		wantErr     bool
	}{
		{
			name:    "full pipeline succeeds",
			wantErr: false,
		},
		{
			name:        "rollback failure stops pipeline",
			rollbackErr: errors.New("snapshot not found"),
			wantPhase:   PhaseRollback,
			wantErr:     true,
		},
		{
			name:       "retrain failure stops pipeline",
			retrainErr: errors.New("out of memory"),
			wantPhase:  PhaseRetrain,
			wantErr:    true,
		},
		{
			name:        "validate failure stops pipeline",
			validateErr: errors.New("accuracy below threshold"),
			wantPhase:   PhaseValidate,
			wantErr:     true,
		},
		{
			name:        "redeploy failure stops pipeline",
			redeployErr: errors.New("server unavailable"),
			wantPhase:   PhaseRedeploy,
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var phases []Phase

			cfg := AutoRetrainConfig{
				RollbackFn: func() error {
					phases = append(phases, PhaseRollback)
					return tt.rollbackErr
				},
				RetrainFn: func() error {
					phases = append(phases, PhaseRetrain)
					return tt.retrainErr
				},
				ValidateFn: func() error {
					phases = append(phases, PhaseValidate)
					return tt.validateErr
				},
				RedeployFn: func() error {
					phases = append(phases, PhaseRedeploy)
					return tt.redeployErr
				},
			}

			ar, err := NewAutoRetrain(cfg)
			if err != nil {
				t.Fatalf("NewAutoRetrain: %v", err)
			}

			err = ar.RunOnDrift()

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				var pe *PipelineError
				if !errors.As(err, &pe) {
					t.Fatalf("expected PipelineError, got %T: %v", err, err)
				}
				if pe.Phase != tt.wantPhase {
					t.Fatalf("expected phase %v, got %v", tt.wantPhase, pe.Phase)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				// Verify all phases ran in order.
				want := []Phase{PhaseRollback, PhaseRetrain, PhaseValidate, PhaseRedeploy}
				if len(phases) != len(want) {
					t.Fatalf("expected %d phases, got %d: %v", len(want), len(phases), phases)
				}
				for i, p := range phases {
					if p != want[i] {
						t.Fatalf("phase %d: expected %v, got %v", i, want[i], p)
					}
				}
			}
		})
	}
}

func TestAutoRetrain_Run_DetectsThenRecovers(t *testing.T) {
	detector := &mockDetector{threshold: 5}

	callCount := 0
	retrained := false

	cfg := AutoRetrainConfig{
		RetrainFn: func() error {
			retrained = true
			return nil
		},
	}

	ar, err := NewAutoRetrain(cfg)
	if err != nil {
		t.Fatalf("NewAutoRetrain: %v", err)
	}

	// Stream 10 values; drift triggers at observation 5.
	err = ar.Run(detector, func() (float64, bool) {
		callCount++
		if callCount > 10 {
			return 0, false
		}
		return float64(callCount), true
	})

	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !retrained {
		t.Fatal("expected retrain to be called after drift detection")
	}
}

func TestAutoRetrain_NilRetrainFn(t *testing.T) {
	_, err := NewAutoRetrain(AutoRetrainConfig{})
	if err == nil {
		t.Fatal("expected error for nil RetrainFn")
	}
}

func TestAutoRetrain_NilOptionalCallbacks(t *testing.T) {
	// Pipeline should succeed with only RetrainFn set.
	retrained := false
	cfg := AutoRetrainConfig{
		RetrainFn: func() error {
			retrained = true
			return nil
		},
	}

	ar, err := NewAutoRetrain(cfg)
	if err != nil {
		t.Fatalf("NewAutoRetrain: %v", err)
	}

	if err := ar.RunOnDrift(); err != nil {
		t.Fatalf("RunOnDrift: %v", err)
	}
	if !retrained {
		t.Fatal("expected retrain to be called")
	}
}

func TestAutoRetrain_WithPageHinkley(t *testing.T) {
	ph := monitor.NewPageHinkley(monitor.PageHinkleyConfig{
		Delta:  0.005,
		Lambda: 5,
	})

	retrained := false
	cfg := AutoRetrainConfig{
		RetrainFn: func() error {
			retrained = true
			return nil
		},
	}

	ar, err := NewAutoRetrain(cfg)
	if err != nil {
		t.Fatalf("NewAutoRetrain: %v", err)
	}

	i := 0
	err = ar.Run(ph, func() (float64, bool) {
		i++
		if i > 300 {
			return 0, false
		}
		// First 100: stable at 1.0, then shift to 10.0.
		if i <= 100 {
			return 1.0, true
		}
		return 10.0, true
	})

	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !retrained {
		t.Fatal("expected retrain to trigger after drift with PageHinkley")
	}
}

func TestAutoRetrain_WithADWIN(t *testing.T) {
	ad := monitor.NewADWIN(monitor.ADWINConfig{Confidence: 0.01})

	retrained := false
	cfg := AutoRetrainConfig{
		RetrainFn: func() error {
			retrained = true
			return nil
		},
	}

	ar, err := NewAutoRetrain(cfg)
	if err != nil {
		t.Fatalf("NewAutoRetrain: %v", err)
	}

	i := 0
	err = ar.Run(ad, func() (float64, bool) {
		i++
		if i > 300 {
			return 0, false
		}
		if i <= 100 {
			return 0.0, true
		}
		return 10.0, true
	})

	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !retrained {
		t.Fatal("expected retrain to trigger after drift with ADWIN")
	}
}

func TestPipelineError_Unwrap(t *testing.T) {
	inner := errors.New("inner error")
	pe := &PipelineError{Phase: PhaseRetrain, Err: inner}

	if !errors.Is(pe, inner) {
		t.Fatal("expected Unwrap to return inner error")
	}

	got := pe.Error()
	want := "recover: retrain phase failed: inner error"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestPhase_String(t *testing.T) {
	tests := []struct {
		phase Phase
		want  string
	}{
		{PhaseDetect, "detect"},
		{PhaseRollback, "rollback"},
		{PhaseRetrain, "retrain"},
		{PhaseValidate, "validate"},
		{PhaseRedeploy, "redeploy"},
		{Phase(99), "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.phase.String(); got != tt.want {
				t.Fatalf("expected %q, got %q", tt.want, got)
			}
		})
	}
}
