package core

import (
	"context"
	"math"
	"sync/atomic"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// fusedSwiGLUEngine wraps a CPUEngine and implements FusedSwiGLUProvider.
// It records whether GPUFusedSwiGLU was called so tests can verify dispatch.
type fusedSwiGLUEngine struct {
	compute.Engine[float32]
	calls atomic.Int64
}

func (e *fusedSwiGLUEngine) GPUFusedSwiGLU(w1, w3 *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	e.calls.Add(1)
	// Compute SwiGLU on CPU: output[i] = w1[i] * sigmoid(w1[i]) * w3[i]
	d1 := w1.Data()
	d3 := w3.Data()
	out := make([]float32, len(d1))
	for i := range d1 {
		sig := float32(1.0 / (1.0 + math.Exp(-float64(d1[i]))))
		out[i] = d1[i] * sig * d3[i]
	}
	return tensor.New(w1.Shape(), out)
}

func TestFFN_FusedSwiGLU_Dispatch(t *testing.T) {
	tests := []struct {
		name      string
		inputDim  int
		hiddenDim int
		outputDim int
		batchSize int
	}{
		{"small", 4, 8, 4, 1},
		{"medium", 8, 16, 8, 2},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			base := compute.NewCPUEngine[float32](numeric.Float32Ops{})
			engine := &fusedSwiGLUEngine{Engine: base}

			ffn, err := NewFFN[float32]("test", engine, &numeric.Float32Ops{},
				tc.inputDim, tc.hiddenDim, tc.outputDim,
				WithFFNNoBias[float32](),
			)
			if err != nil {
				t.Fatalf("NewFFN: %v", err)
			}

			input, err := tensor.New[float32]([]int{tc.batchSize, tc.inputDim}, nil)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			before := engine.calls.Load()
			_, err = ffn.Forward(context.Background(), input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			after := engine.calls.Load()
			if after <= before {
				t.Errorf("GPUFusedSwiGLU was not dispatched: calls before=%d after=%d", before, after)
			}
		})
	}
}

func TestFFN_FusedSwiGLU_DispatchViaProxy(t *testing.T) {
	base := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	fused := &fusedSwiGLUEngine{Engine: base}
	proxy := compute.NewEngineProxy[float32](fused)

	ffn, err := NewFFN[float32]("test", proxy, &numeric.Float32Ops{}, 4, 8, 4,
		WithFFNNoBias[float32](),
	)
	if err != nil {
		t.Fatalf("NewFFN: %v", err)
	}

	input, err := tensor.New[float32]([]int{1, 4}, nil)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	before := fused.calls.Load()
	_, err = ffn.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	after := fused.calls.Load()
	if after <= before {
		t.Errorf("GPUFusedSwiGLU was not dispatched through EngineProxy: calls before=%d after=%d", before, after)
	}
}

func TestFFN_FusedSwiGLU_FallbackWhenNotAvailable(t *testing.T) {
	// Plain CPUEngine does not implement FusedSwiGLUProvider.
	// FFN should still work via the unfused path.
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	ffn, err := NewFFN[float32]("test", engine, &numeric.Float32Ops{}, 4, 8, 4,
		WithFFNNoBias[float32](),
	)
	if err != nil {
		t.Fatalf("NewFFN: %v", err)
	}

	input, err := tensor.New[float32]([]int{1, 4}, nil)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	out, err := ffn.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if out == nil {
		t.Fatal("output is nil")
	}
}
