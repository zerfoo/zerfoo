package scheduler

import (
	"math"
	"testing"
)

func TestReduceOnPlateau(t *testing.T) {
	tests := []struct {
		name       string
		mode       string
		metrics    []float64
		patience   int
		factor     float64
		initialLR  float32
		minLR      float64
		wantLRLess bool
	}{
		{
			name:       "reduce after patience with min mode",
			mode:       "min",
			metrics:    []float64{1.0, 0.9, 0.8, 0.8, 0.8, 0.8}, // plateau at index 2
			patience:   3,
			factor:     0.1,
			initialLR:  0.01,
			minLR:      0.0,
			wantLRLess: true,
		},
		{
			name:       "no reduce when improving",
			mode:       "min",
			metrics:    []float64{1.0, 0.9, 0.8, 0.7, 0.6},
			patience:   3,
			factor:     0.1,
			initialLR:  0.01,
			minLR:      0.0,
			wantLRLess: false,
		},
		{
			name:       "max mode reduce after plateau",
			mode:       "max",
			metrics:    []float64{0.5, 0.6, 0.7, 0.7, 0.7, 0.7},
			patience:   3,
			factor:     0.5,
			initialLR:  0.01,
			minLR:      0.0,
			wantLRLess: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := NewReduceOnPlateau(ReduceOnPlateauConfig[float32]{
				InitialLR: tc.initialLR,
				Factor:    tc.factor,
				Patience:  tc.patience,
				MinLR:     tc.minLR,
				Mode:      tc.mode,
			})

			for i, m := range tc.metrics {
				s.Step(i, m)
			}

			got := float64(s.GetLR())
			initial := float64(tc.initialLR)

			if tc.wantLRLess && got >= initial {
				t.Errorf("expected LR < %v after plateau, got %v", initial, got)
			}
			if !tc.wantLRLess && got != initial {
				t.Errorf("expected LR = %v (no plateau), got %v", initial, got)
			}
		})
	}
}

func TestReduceOnPlateau_MinLR(t *testing.T) {
	s := NewReduceOnPlateau(ReduceOnPlateauConfig[float32]{
		InitialLR: 0.01,
		Factor:    0.1,
		Patience:  1,
		MinLR:     0.001,
		Mode:      "min",
	})

	// Feed a constant metric many times to trigger multiple reductions.
	for i := 0; i < 20; i++ {
		s.Step(i, 5.0)
	}

	got := float64(s.GetLR())
	if got < 0.001-1e-9 {
		t.Errorf("LR %v went below MinLR 0.001", got)
	}
}

func TestCosineAnnealing(t *testing.T) {
	tMax := 10
	etaMax := float32(0.01)
	etaMin := 0.0001

	s := NewCosineAnnealing(CosineAnnealingConfig[float32]{
		EtaMax: etaMax,
		EtaMin: etaMin,
		TMax:   tMax,
	})

	// At epoch 0, LR should be etaMax (cosine(0) = 1).
	s.Step(0, 0)
	lr0 := float64(s.GetLR())
	if math.Abs(lr0-float64(etaMax)) > 1e-5 {
		t.Errorf("epoch 0: expected LR ~%v, got %v", etaMax, lr0)
	}

	// At epoch tMax, LR should be etaMin (cosine(pi) = -1).
	s.Step(tMax, 0)
	lrEnd := float64(s.GetLR())
	if math.Abs(lrEnd-etaMin) > 1e-5 {
		t.Errorf("epoch %d: expected LR ~%v, got %v", tMax, etaMin, lrEnd)
	}

	// At epoch tMax/2, LR should be midpoint.
	s.Step(tMax/2, 0)
	lrMid := float64(s.GetLR())
	expectedMid := etaMin + 0.5*(float64(etaMax)-etaMin)*(1+math.Cos(math.Pi*0.5))
	if math.Abs(lrMid-expectedMid) > 1e-5 {
		t.Errorf("epoch %d: expected LR ~%v, got %v", tMax/2, expectedMid, lrMid)
	}
}

func TestCosineAnnealing_WarmRestarts(t *testing.T) {
	tMax := 5
	etaMax := float32(0.01)
	etaMin := 0.0001

	s := NewCosineAnnealing(CosineAnnealingConfig[float32]{
		EtaMax:       etaMax,
		EtaMin:       etaMin,
		TMax:         tMax,
		WarmRestarts: true,
	})

	// At epoch 0 and epoch tMax (which resets to 0), LR should be etaMax.
	s.Step(0, 0)
	lr0 := float64(s.GetLR())

	s.Step(tMax, 0)
	lrRestart := float64(s.GetLR())

	if math.Abs(lr0-lrRestart) > 1e-5 {
		t.Errorf("warm restart: epoch 0 LR (%v) != epoch %d LR (%v)", lr0, tMax, lrRestart)
	}

	// LR at epoch 2*tMax should also reset.
	s.Step(2*tMax, 0)
	lrRestart2 := float64(s.GetLR())
	if math.Abs(lr0-lrRestart2) > 1e-5 {
		t.Errorf("warm restart: epoch 0 LR (%v) != epoch %d LR (%v)", lr0, 2*tMax, lrRestart2)
	}
}

func TestScheduler_IntegrationWithSetLR(t *testing.T) {
	// Verify the scheduler output can be used as an optimizer LR value.
	s := NewCosineAnnealing(CosineAnnealingConfig[float32]{
		EtaMax: 0.01,
		EtaMin: 0.0001,
		TMax:   10,
	})

	s.Step(5, 0)
	lr := s.GetLR()

	// Simulate what an optimizer.SetLR call does: accept a T value.
	var optimizerLR float32
	optimizerLR = lr

	if optimizerLR <= 0 {
		t.Errorf("expected positive LR from scheduler, got %v", optimizerLR)
	}
	if optimizerLR >= 0.01 {
		t.Errorf("expected LR < initial after half-cycle, got %v", optimizerLR)
	}
}
