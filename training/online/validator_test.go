package online

import (
	"math"
	"testing"
)

func TestLossDeltaValidator(t *testing.T) {
	v := NewLossDeltaValidator(0.1)

	t.Run("pass when loss improves", func(t *testing.T) {
		before := ModelSnapshot{Loss: 1.0}
		after := ModelSnapshot{Loss: 0.8}
		result := v.Validate(before, after)
		if !result.Pass {
			t.Fatalf("expected pass, got fail: %s", result.Reason)
		}
	})

	t.Run("pass when loss increases within threshold", func(t *testing.T) {
		before := ModelSnapshot{Loss: 1.0}
		after := ModelSnapshot{Loss: 1.05}
		result := v.Validate(before, after)
		if !result.Pass {
			t.Fatalf("expected pass, got fail: %s", result.Reason)
		}
	})

	t.Run("fail when loss increases beyond threshold", func(t *testing.T) {
		before := ModelSnapshot{Loss: 1.0}
		after := ModelSnapshot{Loss: 1.2}
		result := v.Validate(before, after)
		if result.Pass {
			t.Fatal("expected fail, got pass")
		}
		if result.Reason == "" {
			t.Fatal("expected failure reason")
		}
	})
}

func TestWeightNormValidator(t *testing.T) {
	v := NewWeightNormValidator(10.0)

	t.Run("pass normal weights", func(t *testing.T) {
		after := ModelSnapshot{
			Weights: map[string][]float32{
				"layer1": {1.0, 2.0, 3.0},
				"layer2": {0.5, 0.5, 0.5},
			},
		}
		result := v.Validate(ModelSnapshot{}, after)
		if !result.Pass {
			t.Fatalf("expected pass, got fail: %s", result.Reason)
		}
	})

	t.Run("fail huge weights", func(t *testing.T) {
		after := ModelSnapshot{
			Weights: map[string][]float32{
				"layer1": {1.0, 2.0, 3.0},
				"layer2": {100.0, 100.0, 100.0},
			},
		}
		// L2 norm of layer2 = sqrt(30000) ≈ 173.2
		result := v.Validate(ModelSnapshot{}, after)
		if result.Pass {
			t.Fatal("expected fail, got pass")
		}
		if result.Reason == "" {
			t.Fatal("expected failure reason")
		}
	})

	t.Run("pass empty weights", func(t *testing.T) {
		after := ModelSnapshot{
			Weights: map[string][]float32{},
		}
		result := v.Validate(ModelSnapshot{}, after)
		if !result.Pass {
			t.Fatalf("expected pass, got fail: %s", result.Reason)
		}
	})
}

func TestCompositeValidator(t *testing.T) {
	t.Run("all pass", func(t *testing.T) {
		cv := NewCompositeValidator(
			NewLossDeltaValidator(0.1),
			NewWeightNormValidator(10.0),
		)
		before := ModelSnapshot{Loss: 1.0}
		after := ModelSnapshot{
			Loss: 0.9,
			Weights: map[string][]float32{
				"layer1": {1.0, 2.0, 3.0},
			},
		}
		result := cv.Validate(before, after)
		if !result.Pass {
			t.Fatalf("expected pass, got fail: %s", result.Reason)
		}
	})

	t.Run("one fails with reason", func(t *testing.T) {
		cv := NewCompositeValidator(
			NewLossDeltaValidator(0.1),
			NewWeightNormValidator(5.0),
		)
		before := ModelSnapshot{Loss: 1.0}
		after := ModelSnapshot{
			Loss: 0.9,
			Weights: map[string][]float32{
				"layer1": {10.0, 10.0, 10.0},
			},
		}
		// L2 norm of layer1 = sqrt(300) ≈ 17.3 > 5.0
		result := cv.Validate(before, after)
		if result.Pass {
			t.Fatal("expected fail, got pass")
		}
		if result.Reason == "" {
			t.Fatal("expected failure reason")
		}
	})

	t.Run("first validator fails", func(t *testing.T) {
		cv := NewCompositeValidator(
			NewLossDeltaValidator(0.01),
			NewWeightNormValidator(100.0),
		)
		before := ModelSnapshot{Loss: 1.0}
		after := ModelSnapshot{
			Loss: 1.5,
			Weights: map[string][]float32{
				"layer1": {1.0, 1.0},
			},
		}
		result := cv.Validate(before, after)
		if result.Pass {
			t.Fatal("expected fail, got pass")
		}
		if result.Reason != "loss increased beyond threshold" {
			t.Fatalf("unexpected reason: %s", result.Reason)
		}
	})
}

func TestL2Norm(t *testing.T) {
	// [3, 4] → sqrt(9+16) = 5
	got := l2Norm([]float32{3.0, 4.0})
	if math.Abs(got-5.0) > 1e-6 {
		t.Fatalf("expected 5.0, got %f", got)
	}
}
