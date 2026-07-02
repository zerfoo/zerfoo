package training

import (
	"context"
	"math"
	"testing"

	core "github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGradcheckLossSeed is the regression gate for issue #872.
//
// The single shared gradient call site (computeGradientsTensorCommon) must
// seed loss.Backward with d(loss)/d(loss) = 1, NOT the scalar loss value L.
// Loss layers end Backward with grad = localGrad * dOut, so seeding dOut with
// L scales every parameter gradient by L: the analytic gradient becomes
// L * dL/dparam (the gradient of (1/2)L^2) instead of dL/dparam.
//
// This finite-difference check builds a tiny Linear -> CrossEntropyLoss graph,
// reads the analytic per-parameter gradient produced by the strategy, then
// estimates dL/dparam by central differences. With the correct 1.0 seed the
// two agree to fd tolerance; under the old L-seed they differ by a factor of
// ~L, which this test detects and reports.
//
// It runs for both CrossEntropyLoss and MSE (both multiply by dOut), and for
// each it asserts the analytic/numeric ratio is ~1, NOT ~L. On origin/main the
// ratio is ~L for every parameter; with the fix it is ~1.
func TestGradcheckLossSeed(t *testing.T) {
	const (
		eps = 1e-4
		tol = 2e-3 // central-difference accuracy on float64 for this graph
	)

	tests := []struct {
		name    string
		newLoss func(compute.Engine[float64]) graph.Node[float64]
		// targets fed to the loss; for CE these are class indices, for MSE
		// they are regression targets matched to the output shape.
		targets []float64
		tShape  []int
	}{
		{
			name: "cross_entropy",
			newLoss: func(e compute.Engine[float64]) graph.Node[float64] {
				return loss.NewCrossEntropyLoss[float64](e)
			},
			targets: []float64{2, 0}, // class indices for 2 samples
			tShape:  []int{2},
		},
		{
			name: "mse",
			newLoss: func(e compute.Engine[float64]) graph.Node[float64] {
				return loss.NewMSE[float64](e, numeric.Float64Ops{})
			},
			targets: []float64{0.1, -0.2, 0.3, 0.4, -0.5, 0.6}, // 2x3 regression targets
			tShape:  []int{2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ops := numeric.Float64Ops{}
			engine := compute.NewCPUEngine[float64](ops)

			// Tiny graph: Input[2,4] -> Linear(4->3) -> logits[2,3].
			const (
				batch = 2
				in    = 4
				out   = 3
			)
			lin, err := core.NewLinear[float64]("lin", engine, ops, in, out)
			if err != nil {
				t.Fatalf("NewLinear: %v", err)
			}
			// Deterministic, well-conditioned weights.
			w := lin.Parameters()[0]
			wd := w.Value.Data()
			for i := range wd {
				wd[i] = 0.05 * float64(i+1)
			}

			b := graph.NewBuilder[float64](engine)
			inNode := b.Input([]int{batch, in})
			b.AddNode(lin, inNode)
			g, err := b.Build(lin)
			if err != nil {
				t.Fatalf("Build: %v", err)
			}

			lossNode := tt.newLoss(engine)

			inputData := make([]float64, batch*in)
			for i := range inputData {
				inputData[i] = 0.1*float64(i) - 0.3
			}
			input, err := tensor.New[float64]([]int{batch, in}, inputData)
			if err != nil {
				t.Fatalf("input tensor: %v", err)
			}
			targets, err := tensor.New[float64](tt.tShape, tt.targets)
			if err != nil {
				t.Fatalf("targets tensor: %v", err)
			}
			batchData := Batch[float64]{
				Inputs:  map[graph.Node[float64]]*tensor.TensorNumeric[float64]{inNode: input},
				Targets: targets,
			}

			ctx := context.Background()

			// scalarLoss recomputes the loss value for the current weights.
			scalarLoss := func() float64 {
				output, ferr := g.Forward(ctx, input)
				if ferr != nil {
					t.Fatalf("forward: %v", ferr)
				}
				lt, lerr := lossNode.Forward(ctx, output, targets)
				if lerr != nil {
					t.Fatalf("loss forward: %v", lerr)
				}
				v := lt.Data()[0]
				g.ClearMemo()
				return v
			}

			// Analytic gradient from the strategy (the code under test).
			strategy := NewDefaultBackpropStrategy[float64]()
			w.ClearGradient()
			lossVal, err := strategy.ComputeGradients(ctx, g, lossNode, batchData)
			if err != nil {
				t.Fatalf("ComputeGradients: %v", err)
			}
			analytic := append([]float64(nil), w.Gradient.Data()...)

			if lossVal <= 0 || math.Abs(lossVal-1) < 1e-6 {
				// The whole point is L != 1, so that an L-seed is distinguishable
				// from a 1-seed. Guard the fixture against an accidental L≈1.
				t.Logf("loss value L = %g (L must differ from 1 for this test to discriminate)", lossVal)
			}

			// Numeric gradient by central finite differences.
			maxRatioErr := 0.0
			worstIdx := -1
			for i := range wd {
				orig := wd[i]

				wd[i] = orig + eps
				lp := scalarLoss()

				wd[i] = orig - eps
				lm := scalarLoss()

				wd[i] = orig

				numGrad := (lp - lm) / (2 * eps)
				diff := math.Abs(analytic[i] - numGrad)
				denom := math.Max(1e-8, math.Abs(numGrad))
				if rel := diff / denom; rel > maxRatioErr {
					maxRatioErr = rel
					worstIdx = i
				}

				if diff > tol+tol*math.Abs(numGrad) {
					t.Errorf("param[%d]: analytic=%.8f numeric=%.8f diff=%.3e (ratio analytic/numeric=%.4f, L=%.4f) -- a ratio ~L means loss.Backward was seeded with the loss value, not 1.0 (#872)",
						i, analytic[i], numGrad, diff, analytic[i]/numGrad, lossVal)
				}
			}
			t.Logf("%s: L=%.6f, max relative grad error=%.3e at param[%d] (tol=%.1e)", tt.name, lossVal, maxRatioErr, worstIdx, tol)
		})
	}
}
