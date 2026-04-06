package optimizer

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestAdamW8bit_Quantize(t *testing.T) {
	tests := []struct {
		name      string
		input     []float32
		maxRelErr float64
	}{
		{
			name:      "uniform values",
			input:     []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8},
			maxRelErr: 0.01,
		},
		{
			name:      "zeros",
			input:     make([]float32, 8),
			maxRelErr: 0,
		},
		{
			name:      "similar magnitude",
			input:     []float32{1.0, 1.1, -1.2, 0.9},
			maxRelErr: 0.01,
		},
		{
			name:      "block boundary",
			input:     makeSimilarRange(300), // crosses one block boundary, similar magnitudes
			maxRelErr: 0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := quantizeToInt8(tt.input)
			got := dequantizeFromInt8(q)

			if len(got) != len(tt.input) {
				t.Fatalf("length mismatch: got %d, want %d", len(got), len(tt.input))
			}

			for i, want := range tt.input {
				if want == 0 {
					if got[i] != 0 {
						t.Errorf("index %d: got %f, want 0", i, got[i])
					}
					continue
				}
				relErr := math.Abs(float64(got[i]-want)) / math.Abs(float64(want))
				if relErr > tt.maxRelErr {
					t.Errorf("index %d: relative error %f > %f (got %f, want %f)",
						i, relErr, tt.maxRelErr, got[i], want)
				}
			}
		})
	}
}

func TestAdamW8bit_MemoryReduction(t *testing.T) {
	// 1024 elements: FP32 = 4096 bytes, INT8 = 1024 + 4*ceil(1024/256)*4 = 1024+16 = 1040 bytes
	n := 1024
	src := makeRange(n)

	q := quantizeToInt8(src)
	fp32Bytes := n * 4
	int8Bytes := q.memoryBytes()

	ratio := float64(fp32Bytes) / float64(int8Bytes)
	t.Logf("FP32: %d bytes, INT8: %d bytes, ratio: %.2fx", fp32Bytes, int8Bytes, ratio)

	if ratio < 3.5 {
		t.Errorf("memory reduction ratio %.2f < 3.5x (expected ~4x)", ratio)
	}
}

func TestAdamW8bit_Convergence(t *testing.T) {
	// Minimize f(x,y) = (x-3)^2 + (y+2)^2
	// Optimal: x=3, y=-2, loss=0
	ctx := context.Background()

	// Run FP32 AdamW baseline.
	fp32Loss := runQuadraticAdamW(ctx, t)

	// Run 8-bit AdamW.
	int8Loss := runQuadraticAdamW8bit(ctx, t)

	t.Logf("FP32 final loss: %e", fp32Loss)
	t.Logf("INT8 final loss: %e", int8Loss)

	// Both should converge near zero.
	if fp32Loss > 1e-4 {
		t.Fatalf("FP32 AdamW did not converge: loss=%e", fp32Loss)
	}

	if int8Loss > 1e-4 {
		t.Fatalf("INT8 AdamW did not converge: loss=%e", int8Loss)
	}

	// INT8 loss within 2% of FP32 loss (use absolute comparison when both near zero).
	diff := math.Abs(float64(int8Loss - fp32Loss))
	threshold := math.Max(0.02*float64(fp32Loss), 1e-6)
	if diff > threshold {
		t.Errorf("INT8 loss %e not within 2%% of FP32 loss %e (diff=%e, threshold=%e)",
			int8Loss, fp32Loss, diff, threshold)
	}
}

func TestAdamW8bit_NoGradient(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	opt := NewAdamW8bit[float32](engine, 0.01, 0.9, 0.999, 1e-8, 0.0)

	value, err := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	if err != nil {
		t.Fatal(err)
	}

	param, err := graph.NewParameter("p", value, tensor.New[float32])
	if err != nil {
		t.Fatal(err)
	}
	// No gradient set.

	err = opt.Step(context.Background(), []*graph.Parameter[float32]{param})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := param.Value.Data()
	if got[0] != 1.0 || got[1] != 2.0 {
		t.Errorf("params should not change without gradient: got %v", got)
	}
}

func TestAdamW8bit_WeightDecay(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	opt := NewAdamW8bit[float32](engine, 0.01, 0.9, 0.999, 1e-8, 0.1)

	value, err := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	if err != nil {
		t.Fatal(err)
	}

	gradient, err := tensor.New[float32]([]int{2}, []float32{0.0, 0.0})
	if err != nil {
		t.Fatal(err)
	}

	param, err := graph.NewParameter("p", value, tensor.New[float32])
	if err != nil {
		t.Fatal(err)
	}
	param.Gradient = gradient

	orig := make([]float32, 2)
	copy(orig, param.Value.Data())

	if err := opt.Step(ctx, []*graph.Parameter[float32]{param}); err != nil {
		t.Fatal(err)
	}

	got := param.Value.Data()
	for i, o := range orig {
		if got[i] >= o {
			t.Errorf("index %d: value %f should decrease due to weight decay (was %f)", i, got[i], o)
		}
	}
}

func TestAdamW8bit_GradientCleared(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	opt := NewAdamW8bit[float32](engine, 0.01, 0.9, 0.999, 1e-8, 0.0)

	value, err := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	if err != nil {
		t.Fatal(err)
	}

	gradient, err := tensor.New[float32]([]int{2}, []float32{0.5, 0.5})
	if err != nil {
		t.Fatal(err)
	}

	param, err := graph.NewParameter("p", value, tensor.New[float32])
	if err != nil {
		t.Fatal(err)
	}
	param.Gradient = gradient

	if err := opt.Step(ctx, []*graph.Parameter[float32]{param}); err != nil {
		t.Fatal(err)
	}

	for i, v := range param.Gradient.Data() {
		if v != 0 {
			t.Errorf("gradient[%d] = %f, want 0", i, v)
		}
	}
}

// --- helpers ---

func makeRange(n int) []float32 {
	out := make([]float32, n)
	for i := range n {
		out[i] = float32(i-n/2) * 0.01
	}
	return out
}

func makeSimilarRange(n int) []float32 {
	out := make([]float32, n)
	for i := range n {
		out[i] = 1.0 + float32(i)*0.001
	}
	return out
}

// runQuadraticAdamW runs FP32 AdamW on f(x,y) = (x-3)^2 + (y+2)^2 for 1000 steps.
func runQuadraticAdamW(ctx context.Context, t *testing.T) float32 {
	t.Helper()

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	opt := NewAdamW[float32](engine, 0.01, 0.9, 0.999, 1e-8, 0.0)

	value, err := tensor.New[float32]([]int{2}, []float32{0.0, 0.0})
	if err != nil {
		t.Fatal(err)
	}

	param, err := graph.NewParameter("xy", value, tensor.New[float32])
	if err != nil {
		t.Fatal(err)
	}

	params := []*graph.Parameter[float32]{param}

	for range 1000 {
		d := param.Value.Data()
		x, y := float64(d[0]), float64(d[1])

		// grad f = (2(x-3), 2(y+2))
		gx := float32(2 * (x - 3))
		gy := float32(2 * (y + 2))

		gradient, err := tensor.New[float32]([]int{2}, []float32{gx, gy})
		if err != nil {
			t.Fatal(err)
		}
		param.Gradient = gradient

		if err := opt.Step(ctx, params); err != nil {
			t.Fatal(err)
		}
	}

	d := param.Value.Data()
	x, y := float64(d[0]), float64(d[1])
	return float32((x-3)*(x-3) + (y+2)*(y+2))
}

// runQuadraticAdamW8bit runs INT8 AdamW on the same quadratic for 1000 steps.
func runQuadraticAdamW8bit(ctx context.Context, t *testing.T) float32 {
	t.Helper()

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	opt := NewAdamW8bit[float32](engine, 0.01, 0.9, 0.999, 1e-8, 0.0)

	value, err := tensor.New[float32]([]int{2}, []float32{0.0, 0.0})
	if err != nil {
		t.Fatal(err)
	}

	param, err := graph.NewParameter("xy", value, tensor.New[float32])
	if err != nil {
		t.Fatal(err)
	}

	params := []*graph.Parameter[float32]{param}

	for range 1000 {
		d := param.Value.Data()
		x, y := float64(d[0]), float64(d[1])

		gx := float32(2 * (x - 3))
		gy := float32(2 * (y + 2))

		gradient, err := tensor.New[float32]([]int{2}, []float32{gx, gy})
		if err != nil {
			t.Fatal(err)
		}
		param.Gradient = gradient

		if err := opt.Step(ctx, params); err != nil {
			t.Fatal(err)
		}
	}

	d := param.Value.Data()
	x, y := float64(d[0]), float64(d[1])
	return float32((x-3)*(x-3) + (y+2)*(y+2))
}
