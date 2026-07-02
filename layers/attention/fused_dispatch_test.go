package attention

import (
	"context"
	"math"
	"sync/atomic"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// fusedScaledSoftmaxEngine wraps a CPUEngine and implements FusedScaledSoftmaxProvider.
// It records whether GPUScaledSoftmax was called so tests can verify dispatch.
type fusedScaledSoftmaxEngine struct {
	compute.Engine[float32]
	calls atomic.Int64
}

func (e *fusedScaledSoftmaxEngine) GPUScaledSoftmax(input *tensor.TensorNumeric[float32], scale float32, axis int) (*tensor.TensorNumeric[float32], error) {
	e.calls.Add(1)
	// Compute scaled softmax on CPU: softmax(input * scale)
	data := input.Data()
	shape := input.Shape()
	out := make([]float32, len(data))
	copy(out, data)

	// Scale
	for i := range out {
		out[i] *= scale
	}

	// Softmax along last axis
	lastDim := shape[len(shape)-1]
	batches := len(out) / lastDim

	for b := 0; b < batches; b++ {
		off := b * lastDim
		row := out[off : off+lastDim]

		maxVal := row[0]
		for _, v := range row[1:] {
			if v > maxVal {
				maxVal = v
			}
		}
		sum := float32(0)
		for i := range row {
			row[i] = float32(math.Exp(float64(row[i] - maxVal)))
			sum += row[i]
		}
		for i := range row {
			row[i] /= sum
		}
	}

	return tensor.New(shape, out)
}

func TestSDPA_FusedScaledSoftmax_Dispatch(t *testing.T) {
	tests := []struct {
		name    string
		batch   int
		seqLen  int
		headDim int
	}{
		{"small", 1, 1, 4},
		{"medium", 2, 1, 8},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			base := compute.NewCPUEngine[float32](numeric.Float32Ops{})
			engine := &fusedScaledSoftmaxEngine{Engine: base}

			sdpa := NewScaledDotProductAttention[float32](engine, tc.headDim)

			q, err := tensor.New[float32]([]int{tc.batch, tc.seqLen, tc.headDim}, nil)
			if err != nil {
				t.Fatalf("tensor.New Q: %v", err)
			}
			k, err := tensor.New[float32]([]int{tc.batch, tc.seqLen, tc.headDim}, nil)
			if err != nil {
				t.Fatalf("tensor.New K: %v", err)
			}
			v, err := tensor.New[float32]([]int{tc.batch, tc.seqLen, tc.headDim}, nil)
			if err != nil {
				t.Fatalf("tensor.New V: %v", err)
			}

			before := engine.calls.Load()
			_, err = sdpa.Forward(context.Background(), q, k, v, nil)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			after := engine.calls.Load()
			if after <= before {
				t.Errorf("GPUScaledSoftmax was not dispatched: calls before=%d after=%d", before, after)
			}
		})
	}
}

func TestSDPA_FusedScaledSoftmax_DispatchViaProxy(t *testing.T) {
	base := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	fused := &fusedScaledSoftmaxEngine{Engine: base}
	proxy := compute.NewEngineProxy[float32](fused)

	sdpa := NewScaledDotProductAttention[float32](proxy, 4)

	q, _ := tensor.New[float32]([]int{1, 1, 4}, nil)
	k, _ := tensor.New[float32]([]int{1, 1, 4}, nil)
	v, _ := tensor.New[float32]([]int{1, 1, 4}, nil)

	before := fused.calls.Load()
	_, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	after := fused.calls.Load()
	if after <= before {
		t.Errorf("GPUScaledSoftmax was not dispatched through EngineProxy: calls before=%d after=%d", before, after)
	}
}

func TestSDPA_FusedScaledSoftmax_SkippedWithMask(t *testing.T) {
	base := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	engine := &fusedScaledSoftmaxEngine{Engine: base}

	sdpa := NewScaledDotProductAttention[float32](engine, 4)

	q, _ := tensor.New[float32]([]int{1, 2, 4}, nil)
	k, _ := tensor.New[float32]([]int{1, 2, 4}, nil)
	v, _ := tensor.New[float32]([]int{1, 2, 4}, nil)
	mask, _ := tensor.New[float32]([]int{1, 1, 2, 2}, nil)

	before := engine.calls.Load()
	_, err := sdpa.Forward(context.Background(), q, k, v, mask)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	after := engine.calls.Load()
	if after != before {
		t.Errorf("GPUScaledSoftmax should not be dispatched when mask is present: calls before=%d after=%d", before, after)
	}
}

func TestSDPA_FusedScaledSoftmax_FallbackWhenNotAvailable(t *testing.T) {
	// Plain CPUEngine does not implement FusedScaledSoftmaxProvider.
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	sdpa := NewScaledDotProductAttention[float32](engine, 4)

	q, _ := tensor.New[float32]([]int{1, 1, 4}, nil)
	k, _ := tensor.New[float32]([]int{1, 1, 4}, nil)
	v, _ := tensor.New[float32]([]int{1, 1, 4}, nil)

	out, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if out == nil {
		t.Fatal("output is nil")
	}
}
