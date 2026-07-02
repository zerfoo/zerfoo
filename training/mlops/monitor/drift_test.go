package monitor

import (
	"testing"
)

func TestDriftDetector_DetectsShift(t *testing.T) {
	t.Run("PageHinkley", func(t *testing.T) {
		tests := []struct {
			name      string
			config    PageHinkleyConfig
			stable    []float64
			shifted   []float64
			wantDrift bool
		}{
			{
				name:   "detects upward mean shift",
				config: PageHinkleyConfig{Delta: 0.005, Lambda: 5},
				stable: func() []float64 {
					vals := make([]float64, 100)
					for i := range vals {
						vals[i] = 1.0
					}
					return vals
				}(),
				shifted: func() []float64 {
					vals := make([]float64, 100)
					for i := range vals {
						vals[i] = 5.0
					}
					return vals
				}(),
				wantDrift: true,
			},
			{
				name:   "no drift on stable data",
				config: PageHinkleyConfig{Delta: 0.005, Lambda: 50},
				stable: func() []float64 {
					vals := make([]float64, 200)
					for i := range vals {
						vals[i] = 1.0
					}
					return vals
				}(),
				shifted:   nil,
				wantDrift: false,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				ph := NewPageHinkley(tt.config)
				driftDetected := false

				for _, v := range tt.stable {
					if ph.Observe(v) {
						driftDetected = true
						break
					}
				}
				if !tt.wantDrift && driftDetected {
					t.Fatal("unexpected drift during stable phase")
				}
				if tt.wantDrift && driftDetected {
					// Drift was detected during stable phase, which
					// shouldn't happen for a proper shift test.
					t.Fatal("drift detected too early, during stable phase")
				}

				for _, v := range tt.shifted {
					if ph.Observe(v) {
						driftDetected = true
						break
					}
				}

				if tt.wantDrift && !driftDetected {
					t.Fatal("expected drift to be detected after mean shift")
				}
				if !tt.wantDrift && driftDetected {
					t.Fatal("unexpected drift detected on stable data")
				}
			})
		}
	})

	t.Run("ADWIN", func(t *testing.T) {
		tests := []struct {
			name      string
			config    ADWINConfig
			stable    []float64
			shifted   []float64
			wantDrift bool
		}{
			{
				name:   "detects distributional shift",
				config: ADWINConfig{Confidence: 0.01},
				stable: func() []float64 {
					vals := make([]float64, 100)
					for i := range vals {
						vals[i] = 0.0
					}
					return vals
				}(),
				shifted: func() []float64 {
					vals := make([]float64, 100)
					for i := range vals {
						vals[i] = 10.0
					}
					return vals
				}(),
				wantDrift: true,
			},
			{
				name:   "no drift on constant data",
				config: ADWINConfig{Confidence: 0.002},
				stable: func() []float64 {
					vals := make([]float64, 200)
					for i := range vals {
						vals[i] = 5.0
					}
					return vals
				}(),
				shifted:   nil,
				wantDrift: false,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				ad := NewADWIN(tt.config)
				driftDetected := false

				for _, v := range tt.stable {
					if ad.Observe(v) {
						driftDetected = true
						break
					}
				}
				if driftDetected {
					t.Fatal("unexpected drift during stable phase")
				}

				for _, v := range tt.shifted {
					if ad.Observe(v) {
						driftDetected = true
						break
					}
				}

				if tt.wantDrift && !driftDetected {
					t.Fatal("expected drift to be detected after distributional shift")
				}
				if !tt.wantDrift && driftDetected {
					t.Fatal("unexpected drift detected on stable data")
				}
			})
		}
	})
}

func TestDriftDetector_Interface(t *testing.T) {
	// Verify both types satisfy the DriftDetector interface.
	var _ DriftDetector = (*PageHinkley)(nil)
	var _ DriftDetector = (*ADWIN)(nil)
}

func TestPageHinkley_Reset(t *testing.T) {
	ph := NewPageHinkley(PageHinkleyConfig{Delta: 0.005, Lambda: 5})

	// Feed values that would cause drift.
	for i := 0; i < 100; i++ {
		ph.Observe(1.0)
	}
	for i := 0; i < 100; i++ {
		if ph.Observe(10.0) {
			break
		}
	}

	// After reset, should not detect drift on stable data.
	ph.Reset()
	driftDetected := false
	for i := 0; i < 50; i++ {
		if ph.Observe(1.0) {
			driftDetected = true
			break
		}
	}
	if driftDetected {
		t.Fatal("drift detected after reset on stable data")
	}
}

func TestADWIN_Reset(t *testing.T) {
	ad := NewADWIN(ADWINConfig{Confidence: 0.01})

	// Feed values that would cause drift.
	for i := 0; i < 50; i++ {
		ad.Observe(0.0)
	}
	for i := 0; i < 50; i++ {
		if ad.Observe(10.0) {
			break
		}
	}

	// After reset, should not detect drift on stable data.
	ad.Reset()
	driftDetected := false
	for i := 0; i < 50; i++ {
		if ad.Observe(5.0) {
			driftDetected = true
			break
		}
	}
	if driftDetected {
		t.Fatal("drift detected after reset on stable data")
	}
}

func TestPageHinkley_DefaultConfig(t *testing.T) {
	ph := NewPageHinkley(PageHinkleyConfig{})
	if ph.delta != 0.005 {
		t.Fatalf("expected default delta 0.005, got %f", ph.delta)
	}
	if ph.lambda != 50 {
		t.Fatalf("expected default lambda 50, got %f", ph.lambda)
	}
}

func TestADWIN_DefaultConfig(t *testing.T) {
	ad := NewADWIN(ADWINConfig{})
	if ad.confidence != 0.002 {
		t.Fatalf("expected default confidence 0.002, got %f", ad.confidence)
	}
}
