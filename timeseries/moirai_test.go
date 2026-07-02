package timeseries

import (
	"context"
	"math"
	"testing"

	its "github.com/zerfoo/zerfoo/inference/timeseries"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// newMoiraiGraph builds a Moirai graph with synthetic weights for testing.
func newMoiraiGraph(t *testing.T, numVars int) (*its.MoiraiConfig, func(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)) {
	t.Helper()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &its.MoiraiConfig{
		NumLayers:         2,
		HiddenDim:         32,
		NumHeads:          4,
		InputDim:          16,
		NumFreqEmbeddings: 32,
		Horizon:           8,
		NumVars:           numVars,
	}

	g, err := its.BuildMoirai[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildMoirai(numVars=%d): %v", numVars, err)
	}

	forward := func(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return g.Forward(ctx, input)
	}
	return cfg, forward
}

// makeSyntheticInput creates a [batch, numVars, inputDim] tensor with deterministic values.
func makeSyntheticInput(t *testing.T, batch, numVars, inputDim, seed int) *tensor.TensorNumeric[float32] {
	t.Helper()
	n := batch * numVars * inputDim
	data := make([]float32, n)
	for i := range data {
		// Deterministic pseudo-random-looking values based on seed.
		data[i] = float32(math.Sin(float64(i+seed*1000)*0.1)) * 0.5
	}
	in, err := tensor.New[float32]([]int{batch, numVars, inputDim}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}
	return in
}

func TestMoiraiParityOutputShape(t *testing.T) {
	for _, numVars := range []int{1, 5, 20} {
		t.Run("variates_"+itoa(numVars), func(t *testing.T) {
			cfg, forward := newMoiraiGraph(t, numVars)
			ctx := context.Background()

			batch := 10
			input := makeSyntheticInput(t, batch, numVars, cfg.InputDim, 42)

			output, err := forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			got := output.Shape()
			want := []int{batch, cfg.Horizon, numVars}
			if len(got) != 3 {
				t.Fatalf("output rank: got %d, want 3", len(got))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("shape[%d]: got %d, want %d", i, got[i], want[i])
				}
			}
		})
	}
}

func TestMoiraiParityDeterministic(t *testing.T) {
	for _, numVars := range []int{1, 5, 20} {
		t.Run("variates_"+itoa(numVars), func(t *testing.T) {
			cfg, forward := newMoiraiGraph(t, numVars)
			ctx := context.Background()

			batch := 10
			input := makeSyntheticInput(t, batch, numVars, cfg.InputDim, 7)

			out1, err := forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward run 1: %v", err)
			}

			out2, err := forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward run 2: %v", err)
			}

			d1 := out1.Data()
			d2 := out2.Data()
			if len(d1) != len(d2) {
				t.Fatalf("output length mismatch: %d vs %d", len(d1), len(d2))
			}
			for i := range d1 {
				if d1[i] != d2[i] {
					t.Fatalf("non-deterministic at index %d: %v vs %v", i, d1[i], d2[i])
				}
			}
		})
	}
}

func TestMoiraiParityNonDegenerate(t *testing.T) {
	for _, numVars := range []int{1, 5, 20} {
		t.Run("variates_"+itoa(numVars), func(t *testing.T) {
			cfg, forward := newMoiraiGraph(t, numVars)
			ctx := context.Background()

			batch := 10
			input := makeSyntheticInput(t, batch, numVars, cfg.InputDim, 13)

			output, err := forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			data := output.Data()

			// Check no NaN or Inf.
			for i, v := range data {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("output[%d] = %v, want finite", i, v)
				}
			}

			// Check not all zeros.
			allZero := true
			for _, v := range data {
				if v != 0 {
					allZero = false
					break
				}
			}
			if allZero {
				t.Fatal("output is all zeros")
			}

			// Check not all identical values.
			allSame := true
			first := data[0]
			for _, v := range data[1:] {
				if v != first {
					allSame = false
					break
				}
			}
			if allSame {
				t.Fatal("output is all identical values -- degenerate")
			}
		})
	}
}

func TestMoiraiParityDifferentInputsDifferentOutputs(t *testing.T) {
	for _, numVars := range []int{1, 5, 20} {
		t.Run("variates_"+itoa(numVars), func(t *testing.T) {
			cfg, forward := newMoiraiGraph(t, numVars)
			ctx := context.Background()

			batch := 10

			inputA := makeSyntheticInput(t, batch, numVars, cfg.InputDim, 100)
			inputB := makeSyntheticInput(t, batch, numVars, cfg.InputDim, 200)

			outA, err := forward(ctx, inputA)
			if err != nil {
				t.Fatalf("Forward A: %v", err)
			}
			outB, err := forward(ctx, inputB)
			if err != nil {
				t.Fatalf("Forward B: %v", err)
			}

			dA := outA.Data()
			dB := outB.Data()

			allSame := true
			for i := range dA {
				if dA[i] != dB[i] {
					allSame = false
					break
				}
			}
			if allSame {
				t.Fatal("different inputs produced identical outputs")
			}
		})
	}
}

