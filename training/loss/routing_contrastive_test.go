package loss

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func TestRoutingContrastive_Forward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	tests := []struct {
		name     string
		shape    []int
		data     []float32
		scale    float64
		wantLoss float64
		tol      float64
	}{
		{
			name:  "identical heads yield loss=scale",
			shape: []int{1, 2, 3},
			// Two identical heads: cosine similarity = 1.0
			data:     []float32{1, 2, 3, 1, 2, 3},
			scale:    0.01,
			wantLoss: 0.01, // scale * 1.0
			tol:      1e-6,
		},
		{
			name:  "orthogonal heads yield loss=0",
			shape: []int{1, 2, 2},
			// head0=[1,0], head1=[0,1] => cos=0
			data:     []float32{1, 0, 0, 1},
			scale:    0.01,
			wantLoss: 0.0,
			tol:      1e-6,
		},
		{
			name:  "opposite heads yield negative loss",
			shape: []int{1, 2, 2},
			// head0=[1,0], head1=[-1,0] => cos=-1
			data:     []float32{1, 0, -1, 0},
			scale:    0.01,
			wantLoss: -0.01,
			tol:      1e-6,
		},
		{
			name:  "three heads averaged",
			shape: []int{1, 3, 2},
			// head0=[1,0], head1=[0,1], head2=[1,1]
			// cos(0,1)=0, cos(0,2)=1/sqrt(2), cos(1,2)=1/sqrt(2)
			// mean = (0 + 1/sqrt(2) + 1/sqrt(2)) / 3 = 2/(3*sqrt(2))
			data:     []float32{1, 0, 0, 1, 1, 1},
			scale:    1.0,
			wantLoss: 2.0 / (3.0 * math.Sqrt(2)),
			tol:      1e-5,
		},
		{
			name:  "batch of 2",
			shape: []int{2, 2, 2},
			// batch0: head0=[1,0], head1=[0,1] => cos=0
			// batch1: head0=[1,1], head1=[1,1] => cos=1
			// mean over batch = (0 + 1) / 2 = 0.5
			data:     []float32{1, 0, 0, 1, 1, 1, 1, 1},
			scale:    0.1,
			wantLoss: 0.05, // 0.1 * 0.5
			tol:      1e-6,
		},
		{
			name:     "single head yields zero",
			shape:    []int{1, 1, 4},
			data:     []float32{1, 2, 3, 4},
			scale:    0.01,
			wantLoss: 0.0,
			tol:      1e-6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rc := NewRoutingContrastive[float32](engine, ops, tt.scale)
			scores, err := tensor.New[float32](tt.shape, tt.data)
			if err != nil {
				t.Fatalf("failed to create scores tensor: %v", err)
			}
			result, err := rc.Forward(ctx, scores)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			got := float64(result.Data()[0])
			if math.Abs(got-tt.wantLoss) > tt.tol {
				t.Errorf("loss = %v, want %v (tol %v)", got, tt.wantLoss, tt.tol)
			}
		})
	}
}

func TestRoutingContrastive_Forward_Errors(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()
	rc := NewRoutingContrastive[float32](engine, ops, 0.01)

	t.Run("no inputs", func(t *testing.T) {
		_, err := rc.Forward(ctx)
		if err == nil {
			t.Fatal("expected error for no inputs")
		}
	})

	t.Run("wrong dimensionality", func(t *testing.T) {
		scores, _ := tensor.New[float32]([]int{4, 4}, []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		})
		_, err := rc.Forward(ctx, scores)
		if err == nil {
			t.Fatal("expected error for 2D input")
		}
	})
}

func TestRoutingContrastive_Backward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	t.Run("orthogonal heads have zero gradient", func(t *testing.T) {
		rc := NewRoutingContrastive[float32](engine, ops, 0.01)
		// head0=[1,0], head1=[0,1] => cos=0
		scores, _ := tensor.New[float32]([]int{1, 2, 2}, []float32{1, 0, 0, 1})
		_, err := rc.Forward(ctx, scores)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})
		grads, err := rc.Backward(ctx, types.FullBackprop, dOut)
		if err != nil {
			t.Fatalf("Backward failed: %v", err)
		}
		if len(grads) != 1 {
			t.Fatalf("expected 1 gradient tensor, got %d", len(grads))
		}
		// For orthogonal vectors: d(cos)/d(a_s) = b_s/(|a|*|b|)
		// head0=[1,0], head1=[0,1], |h0|=1, |h1|=1
		// grad for h0[0]: coeff * (0/(1*1) - 0*1/1) = 0
		// grad for h0[1]: coeff * (1/(1*1) - 0*0/1) = coeff
		grad := grads[0].Data()
		coeff := float32(0.01) // scale * 1/(numPairs*batch) = 0.01 * 1/(1*1)
		expected := []float32{0, coeff, coeff, 0}
		for i, v := range grad {
			if math.Abs(float64(v-expected[i])) > 1e-6 {
				t.Errorf("grad[%d] = %v, want %v", i, v, expected[i])
			}
		}
	})

	t.Run("single head has zero gradient", func(t *testing.T) {
		rc := NewRoutingContrastive[float32](engine, ops, 0.01)
		scores, _ := tensor.New[float32]([]int{1, 1, 3}, []float32{1, 2, 3})
		_, err := rc.Forward(ctx, scores)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})
		grads, err := rc.Backward(ctx, types.FullBackprop, dOut)
		if err != nil {
			t.Fatalf("Backward failed: %v", err)
		}
		for i, v := range grads[0].Data() {
			if v != 0 {
				t.Errorf("grad[%d] = %v, want 0", i, v)
			}
		}
	})

	t.Run("numerical gradient check", func(t *testing.T) {
		rc := NewRoutingContrastive[float32](engine, ops, 0.5)
		data := []float32{0.3, 0.7, 0.1, 0.9, 0.4, 0.6, 0.2, 0.8, 0.5, 0.5, 0.3, 0.7}
		scores, _ := tensor.New[float32]([]int{1, 3, 4}, data)

		_, err := rc.Forward(ctx, scores)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})
		grads, err := rc.Backward(ctx, types.FullBackprop, dOut)
		if err != nil {
			t.Fatalf("Backward failed: %v", err)
		}
		analyticGrad := grads[0].Data()

		// Numerical gradient via central differences.
		eps := float32(1e-4)
		for idx := range data {
			dataPl := make([]float32, len(data))
			dataMn := make([]float32, len(data))
			copy(dataPl, data)
			copy(dataMn, data)
			dataPl[idx] += eps
			dataMn[idx] -= eps

			rcP := NewRoutingContrastive[float32](engine, ops, 0.5)
			sp, _ := tensor.New[float32]([]int{1, 3, 4}, dataPl)
			lp, _ := rcP.Forward(ctx, sp)

			rcM := NewRoutingContrastive[float32](engine, ops, 0.5)
			sm, _ := tensor.New[float32]([]int{1, 3, 4}, dataMn)
			lm, _ := rcM.Forward(ctx, sm)

			numGrad := (lp.Data()[0] - lm.Data()[0]) / (2 * eps)
			diff := math.Abs(float64(analyticGrad[idx] - numGrad))
			if diff > 1e-3 {
				t.Errorf("idx %d: analytic=%v, numerical=%v, diff=%v", idx, analyticGrad[idx], numGrad, diff)
			}
		}
	})
}

func TestRoutingContrastive_Interface(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	rc := NewRoutingContrastive[float32](engine, ops, 0.01)

	if rc.OpType() != "RoutingContrastive" {
		t.Errorf("OpType = %q, want %q", rc.OpType(), "RoutingContrastive")
	}

	if got := rc.OutputShape(); len(got) != 1 || got[0] != 1 {
		t.Errorf("OutputShape = %v, want [1]", got)
	}

	attrs := rc.Attributes()
	if attrs["scale"] != float32(0.01) {
		t.Errorf("Attributes[scale] = %v, want 0.01", attrs["scale"])
	}

	if rc.Parameters() != nil {
		t.Error("Parameters should be nil")
	}
}
