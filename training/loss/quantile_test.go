package loss

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestQuantileLoss(t *testing.T) {
	tests := []struct {
		name      string
		preds     []float32
		predShape []int
		targets   []float32
		targShape []int
		quantiles []float32
		want      float32
		tol       float32
	}{
		{
			name:      "median q=0.5 symmetric",
			preds:     []float32{1.0, 2.0}, // 2 samples, 1 quantile each
			predShape: []int{2, 1},
			targets:   []float32{2.0, 1.0},
			targShape: []int{2},
			quantiles: []float32{0.5},
			// sample 0: err=2-1=1>=0, loss=0.5*1=0.5
			// sample 1: err=1-2=-1<0, loss=(0.5-1)*(-1)=0.5
			// mean = (0.5+0.5)/2 = 0.5
			want: 0.5,
			tol:  1e-6,
		},
		{
			name:      "q=0.1 asymmetric",
			preds:     []float32{3.0, 1.0},
			predShape: []int{2, 1},
			targets:   []float32{2.0, 2.0},
			targShape: []int{2},
			quantiles: []float32{0.1},
			// sample 0: err=2-3=-1<0, loss=(0.1-1)*(-1)=0.9
			// sample 1: err=2-1=1>=0, loss=0.1*1=0.1
			// mean = (0.9+0.1)/2 = 0.5
			want: 0.5,
			tol:  1e-6,
		},
		{
			name:      "q=0.9 asymmetric",
			preds:     []float32{3.0, 1.0},
			predShape: []int{2, 1},
			targets:   []float32{2.0, 2.0},
			targShape: []int{2},
			quantiles: []float32{0.9},
			// sample 0: err=2-3=-1<0, loss=(0.9-1)*(-1)=0.1
			// sample 1: err=2-1=1>=0, loss=0.9*1=0.9
			// mean = (0.1+0.9)/2 = 0.5
			want: 0.5,
			tol:  1e-6,
		},
		{
			name:      "multiple quantiles",
			preds:     []float32{2.0, 2.0, 2.0}, // 1 sample, 3 quantiles
			predShape: []int{1, 3},
			targets:   []float32{3.0},
			targShape: []int{1},
			quantiles: []float32{0.1, 0.5, 0.9},
			// err=3-2=1>=0 for all quantiles
			// losses: 0.1*1=0.1, 0.5*1=0.5, 0.9*1=0.9
			// mean = (0.1+0.5+0.9)/3 = 0.5
			want: 0.5,
			tol:  1e-6,
		},
		{
			name:      "zero loss when preds equal targets",
			preds:     []float32{5.0, 5.0, 5.0},
			predShape: []int{1, 3},
			targets:   []float32{5.0},
			targShape: []int{1},
			quantiles: []float32{0.1, 0.5, 0.9},
			want:      0.0,
			tol:       1e-6,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			preds, err := tensor.New[float32](tc.predShape, tc.preds)
			if err != nil {
				t.Fatalf("failed to create preds tensor: %v", err)
			}
			targets, err := tensor.New[float32](tc.targShape, tc.targets)
			if err != nil {
				t.Fatalf("failed to create targets tensor: %v", err)
			}

			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
			got, err := QuantileLoss(engine, preds, targets, tc.quantiles)
			if err != nil {
				t.Fatalf("QuantileLoss returned error: %v", err)
			}

			if diff := float32(math.Abs(float64(got - tc.want))); diff > tc.tol {
				t.Errorf("QuantileLoss = %v, want %v (diff %v)", got, tc.want, diff)
			}
		})
	}
}

func TestQuantileLoss_Float64(t *testing.T) {
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	preds, err := tensor.New[float64]([]int{2, 1}, []float64{1.0, 2.0})
	if err != nil {
		t.Fatalf("failed to create preds: %v", err)
	}
	targets, err := tensor.New[float64]([]int{2}, []float64{2.0, 1.0})
	if err != nil {
		t.Fatalf("failed to create targets: %v", err)
	}
	got, err := QuantileLoss(engine, preds, targets, []float32{0.5})
	if err != nil {
		t.Fatalf("QuantileLoss float64 returned error: %v", err)
	}
	if diff := math.Abs(float64(got) - 0.5); diff > 1e-6 {
		t.Errorf("QuantileLoss float64 = %v, want 0.5", got)
	}
}

func TestSharpeLoss(t *testing.T) {
	t.Run("uniform portfolio known Sharpe", func(t *testing.T) {
		// 4 time steps, 2 assets. Equal weights (uniform after softmax).
		// returns: asset0=[0.01, 0.02, -0.01, 0.03], asset1=[0.02, -0.01, 0.03, 0.01]
		// With uniform weights (0.5, 0.5):
		// portfolio returns: [0.015, 0.005, 0.01, 0.02]
		// mean = 0.0125, std = sqrt(((0.0025^2 + 0.0075^2 + 0.0025^2 + 0.0075^2)/4))
		// var = (6.25e-6 + 56.25e-6 + 6.25e-6 + 56.25e-6)/4 = 125e-6/4 = 31.25e-6
		// std = sqrt(31.25e-6) = 0.00559017
		// Sharpe = 0.0125 / 0.00559017 = 2.2360679...

		weights, err := tensor.New[float32]([]int{4, 2}, []float32{
			0, 0, // equal logits -> softmax -> 0.5, 0.5
			0, 0,
			0, 0,
			0, 0,
		})
		if err != nil {
			t.Fatalf("failed to create weights: %v", err)
		}

		returns_, err := tensor.New[float32]([]int{4, 2}, []float32{
			0.01, 0.02,
			0.02, -0.01,
			-0.01, 0.03,
			0.03, 0.01,
		})
		if err != nil {
			t.Fatalf("failed to create returns: %v", err)
		}

		got, err := SharpeLoss[float32](weights, returns_)
		if err != nil {
			t.Fatalf("SharpeLoss returned error: %v", err)
		}

		// Expected Sharpe ~ 2.236, so loss ~ -2.236
		expectedSharpe := 0.0125 / math.Sqrt(31.25e-6)
		expectedLoss := float32(-expectedSharpe)

		if diff := float32(math.Abs(float64(got - expectedLoss))); diff > 0.01 {
			t.Errorf("SharpeLoss = %v, want ~%v (diff %v)", got, expectedLoss, diff)
		}
	})

	t.Run("positive Sharpe gives negative loss", func(t *testing.T) {
		// Portfolio with consistently positive returns should give negative loss.
		weights, err := tensor.New[float32]([]int{3, 2}, []float32{
			0, 0,
			0, 0,
			0, 0,
		})
		if err != nil {
			t.Fatalf("failed to create weights: %v", err)
		}

		returns_, err := tensor.New[float32]([]int{3, 2}, []float32{
			0.05, 0.03,
			0.04, 0.06,
			0.03, 0.05,
		})
		if err != nil {
			t.Fatalf("failed to create returns: %v", err)
		}

		got, err := SharpeLoss[float32](weights, returns_)
		if err != nil {
			t.Fatalf("SharpeLoss returned error: %v", err)
		}

		if got >= 0 {
			t.Errorf("SharpeLoss should be negative for positive Sharpe, got %v", got)
		}
	})

	t.Run("zero variance returns zero loss", func(t *testing.T) {
		// All returns identical -> std=0 -> loss=0
		weights, err := tensor.New[float32]([]int{3, 2}, []float32{
			0, 0,
			0, 0,
			0, 0,
		})
		if err != nil {
			t.Fatalf("failed to create weights: %v", err)
		}

		returns_, err := tensor.New[float32]([]int{3, 2}, []float32{
			0.01, 0.01,
			0.01, 0.01,
			0.01, 0.01,
		})
		if err != nil {
			t.Fatalf("failed to create returns: %v", err)
		}

		got, err := SharpeLoss[float32](weights, returns_)
		if err != nil {
			t.Fatalf("SharpeLoss returned error: %v", err)
		}

		if got != 0 {
			t.Errorf("SharpeLoss should be 0 for zero-variance returns, got %v", got)
		}
	})
}
