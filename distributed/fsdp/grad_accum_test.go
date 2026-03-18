package fsdp

import (
	"math"
	"testing"
)

func TestGradAccum(t *testing.T) {
	const (
		worldSize    = 2
		paramSize    = 8
		stepsPerSync = 4
	)

	model := newMockModel(map[string]int{"w": paramSize})
	sm := NewShardedModule[float32](model, 0, worldSize, nil)
	ga := NewGradAccum[float32](sm, stepsPerSync)

	// Accumulate 4 gradient steps with known values.
	// Step i contributes gradient value (i+1) for every element.
	for step := 0; step < stepsPerSync; step++ {
		grads := map[string][]float32{
			"w": make([]float32, paramSize/worldSize),
		}
		for i := range grads["w"] {
			grads["w"][i] = float32(step + 1)
		}
		ready := ga.Accumulate(grads)
		if step < stepsPerSync-1 && ready {
			t.Fatalf("Accumulate returned true at step %d, want false", step)
		}
		if step == stepsPerSync-1 && !ready {
			t.Fatal("Accumulate did not return true at final step")
		}
	}

	// Sync and verify averaged gradient = (1+2+3+4)/4 = 2.5
	result := ga.Sync()
	wGrad, ok := result["w"]
	if !ok {
		t.Fatal("Sync result missing gradient for 'w'")
	}

	expectedAvg := float32(2.5) // (1+2+3+4)/4
	for i, v := range wGrad {
		if math.Abs(float64(v-expectedAvg)) > 1e-5 {
			t.Errorf("averaged grad[%d] = %v, want %v", i, v, expectedAvg)
		}
	}

	// After Sync, accumulator should be cleared (Reset was called internally).
	if ga.current != 0 {
		t.Errorf("current step after Sync = %d, want 0", ga.current)
	}
	if len(ga.accum) != 0 {
		t.Errorf("accum map after Sync has %d entries, want 0", len(ga.accum))
	}
}

func TestGradAccumReset(t *testing.T) {
	model := newMockModel(map[string]int{"w": 4})
	sm := NewShardedModule[float32](model, 0, 1, nil)
	ga := NewGradAccum[float32](sm, 3)

	// Accumulate one step, then reset.
	ga.Accumulate(map[string][]float32{"w": {1, 2, 3, 4}})
	if ga.current != 1 {
		t.Fatalf("current = %d, want 1", ga.current)
	}

	ga.Reset()

	if ga.current != 0 {
		t.Errorf("current after Reset = %d, want 0", ga.current)
	}
	if len(ga.accum) != 0 {
		t.Errorf("accum has %d entries after Reset, want 0", len(ga.accum))
	}
}

func TestGradAccumMultipleParams(t *testing.T) {
	model := newMockModel(map[string]int{"a": 4, "b": 4})
	sm := NewShardedModule[float32](model, 0, 1, nil)
	ga := NewGradAccum[float32](sm, 2)

	ga.Accumulate(map[string][]float32{
		"a": {1, 1, 1, 1},
		"b": {2, 2, 2, 2},
	})
	ga.Accumulate(map[string][]float32{
		"a": {3, 3, 3, 3},
		"b": {4, 4, 4, 4},
	})

	result := ga.Sync()

	// "a": (1+3)/2 = 2.0
	for i, v := range result["a"] {
		if math.Abs(float64(v-2.0)) > 1e-5 {
			t.Errorf("a[%d] = %v, want 2.0", i, v)
		}
	}
	// "b": (2+4)/2 = 3.0
	for i, v := range result["b"] {
		if math.Abs(float64(v-3.0)) > 1e-5 {
			t.Errorf("b[%d] = %v, want 3.0", i, v)
		}
	}
}

func TestGradAccumStepsPerSyncClampedToOne(t *testing.T) {
	model := newMockModel(map[string]int{"w": 4})
	sm := NewShardedModule[float32](model, 0, 1, nil)
	ga := NewGradAccum[float32](sm, 0) // should be clamped to 1

	ready := ga.Accumulate(map[string][]float32{"w": {1, 2, 3, 4}})
	if !ready {
		t.Error("with stepsPerSync=1 (clamped from 0), Accumulate should return true on first call")
	}
}
