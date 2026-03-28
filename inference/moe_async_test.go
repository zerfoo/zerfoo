package inference

import (
	"context"
	"math"
	"testing"
)

// makeExpertWork creates an ExpertWork item with deterministic data for testing.
// Weight[i] = float32(expertID*1000 + i) * 0.001, Input[i] = float32(i+1) * 0.01.
func makeExpertWork(expertID, numTokens, outDim, inDim int) ExpertWork {
	m, n, k := numTokens, outDim, inDim
	weight := make([]float32, n*k)
	for i := range weight {
		weight[i] = float32(expertID*1000+i) * 0.001
	}
	input := make([]float32, m*k)
	for i := range input {
		input[i] = float32(i+1) * 0.01
	}
	output := make([]float32, m*n)
	return ExpertWork{
		ExpertID: expertID,
		Weight:   weight,
		Input:    input,
		Output:   output,
		M:        m, N: n, K: k,
	}
}

func TestAsyncDispatchMatchesSerial(t *testing.T) {
	tests := []struct {
		name      string
		experts   int
		numTokens int
		outDim    int
		inDim     int
	}{
		{"single_expert_single_token", 1, 1, 4, 8},
		{"single_expert_multi_token", 1, 4, 4, 8},
		{"multi_expert_single_token", 4, 1, 4, 8},
		{"multi_expert_multi_token", 4, 3, 16, 32},
		{"large_dims", 2, 2, 64, 128},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Build serial work items.
			serialItems := make([]ExpertWork, tc.experts)
			for i := range serialItems {
				serialItems[i] = makeExpertWork(i, tc.numTokens, tc.outDim, tc.inDim)
			}

			// Build async work items with identical inputs.
			asyncItems := make([]ExpertWork, tc.experts)
			for i := range asyncItems {
				asyncItems[i] = makeExpertWork(i, tc.numTokens, tc.outDim, tc.inDim)
			}

			// Run serial.
			if err := SerialExpertDispatch(serialItems); err != nil {
				t.Fatalf("serial dispatch: %v", err)
			}

			// Run async.
			d := NewAsyncExpertDispatcher(2)
			defer d.Shutdown()
			d.Dispatch(context.Background(), asyncItems)
			if err := d.Wait(); err != nil {
				t.Fatalf("async dispatch: %v", err)
			}

			// Compare outputs.
			for e := 0; e < tc.experts; e++ {
				for i := range serialItems[e].Output {
					s := serialItems[e].Output[i]
					a := asyncItems[e].Output[i]
					if diff := math.Abs(float64(s - a)); diff > 1e-5 {
						t.Errorf("expert %d output[%d]: serial=%v async=%v diff=%v",
							e, i, s, a, diff)
					}
				}
			}
		})
	}
}

func TestAsyncDispatchEmptyItems(t *testing.T) {
	d := NewAsyncExpertDispatcher(2)
	defer d.Shutdown()
	d.Dispatch(context.Background(), nil)
	if err := d.Wait(); err != nil {
		t.Fatalf("unexpected error on empty dispatch: %v", err)
	}
}

func TestAsyncDispatchCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	items := []ExpertWork{makeExpertWork(0, 1, 4, 8)}

	d := NewAsyncExpertDispatcher(2)
	defer d.Shutdown()
	d.Dispatch(ctx, items)
	err := d.Wait()
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}

func TestAsyncDispatchInvalidDimensions(t *testing.T) {
	items := []ExpertWork{{
		ExpertID: 0,
		Weight:   []float32{1, 2},
		Input:    []float32{1},
		Output:   []float32{0},
		M:        0, N: 1, K: 1,
	}}

	d := NewAsyncExpertDispatcher(2)
	defer d.Shutdown()
	d.Dispatch(context.Background(), items)
	err := d.Wait()
	if err == nil {
		t.Fatal("expected error from invalid dimensions, got nil")
	}
}

func TestSerialDispatchInvalidDimensions(t *testing.T) {
	items := []ExpertWork{{
		ExpertID: 0,
		Weight:   []float32{1},
		Input:    []float32{1},
		Output:   []float32{0},
		M:        -1, N: 1, K: 1,
	}}
	err := SerialExpertDispatch(items)
	if err == nil {
		t.Fatal("expected error from invalid dimensions, got nil")
	}
}

func TestTransposeF32(t *testing.T) {
	// 2x3 matrix:
	// [1 2 3]
	// [4 5 6]
	src := []float32{1, 2, 3, 4, 5, 6}
	// Expected 3x2 transpose:
	// [1 4]
	// [2 5]
	// [3 6]
	want := []float32{1, 4, 2, 5, 3, 6}
	got := transposeF32(2, 3, src)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("transposeF32[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestMultipleDispatchCycles(t *testing.T) {
	d := NewAsyncExpertDispatcher(2)
	defer d.Shutdown()

	for cycle := 0; cycle < 3; cycle++ {
		serialItems := []ExpertWork{makeExpertWork(cycle, 2, 4, 8)}
		asyncItems := []ExpertWork{makeExpertWork(cycle, 2, 4, 8)}

		if err := SerialExpertDispatch(serialItems); err != nil {
			t.Fatalf("cycle %d serial: %v", cycle, err)
		}

		d.Dispatch(context.Background(), asyncItems)
		if err := d.Wait(); err != nil {
			t.Fatalf("cycle %d async: %v", cycle, err)
		}

		for i := range serialItems[0].Output {
			s := serialItems[0].Output[i]
			a := asyncItems[0].Output[i]
			if diff := math.Abs(float64(s - a)); diff > 1e-5 {
				t.Errorf("cycle %d output[%d]: serial=%v async=%v", cycle, i, s, a)
			}
		}
	}
}
