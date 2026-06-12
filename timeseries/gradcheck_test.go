package timeseries

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/gradcheck"
	"github.com/zerfoo/ztensor/types"
)

// This file registers the timeseries model backward passes with ztensor's
// shared gradcheck harness (testing/gradcheck, plan T1.1/T1.6, UC-GH-001),
// replacing the bespoke central finite-difference checkers that previously
// lived in timemixer_backward_test.go, patchtst_backward_test.go,
// itransformer_backward_test.go, and itransformer_test.go.
//
// Tolerance mapping from the retired bespoke checkers: they used eps=1e-5 and
// failed on relErr = |a-n|/max(|a|,|n|) > 1e-3, skipping near-zero pairs
// (both gradients below ~1e-6, where relative error is meaningless). The
// gradcheck criterion |a-n| > Atol + Rtol*max(|a|,|n|) reproduces this with
// Rtol=1e-3 and Atol=2e-6 (the worst-case |a-n| the old deadband admitted).
//
// The iTransformer entry checks the float64 CPU backward (m.backward); the
// engine-batched backward remains covered by
// TestITransformer_BackwardBatchEngine_ParityWithCPU, which pins it to the
// CPU path -- the same decomposition the retired PatchTST checker documented.

// flatParamNode adapts a zerfoo timeseries model exposing flat-parameter
// access ([]*float64) to ztensor's graph.Node[float64] so gradcheck can
// differentiate it. The node takes no graph inputs: deterministic input
// windows are baked into the closures, and the single "theta" parameter
// carries every trainable scalar of the model (gradcheck perturbs theta and
// compares the model's analytic backward against central differences of the
// flattened prediction).
type flatParamNode struct {
	opType   string
	outShape []int
	theta    *graph.Parameter[float64]
	sync     func(vals []float64)
	forward  func() ([]float64, error)
	backward func(upstream []float64) ([]float64, error)
}

func newFlatParamNode(
	opType string,
	outShape []int,
	flat []*float64,
	forward func() ([]float64, error),
	backward func(upstream []float64) ([]float64, error),
) (graph.Node[float64], error) {
	vals := make([]float64, len(flat))
	for i, p := range flat {
		vals[i] = *p
	}
	valT, err := tensor.New[float64]([]int{len(flat)}, vals)
	if err != nil {
		return nil, err
	}
	theta, err := graph.NewParameter("theta", valT, tensor.New[float64])
	if err != nil {
		return nil, err
	}
	return &flatParamNode{
		opType:   opType,
		outShape: outShape,
		theta:    theta,
		sync: func(vals []float64) {
			for i := range flat {
				*flat[i] = vals[i]
			}
		},
		forward:  forward,
		backward: backward,
	}, nil
}

func (n *flatParamNode) OpType() string                     { return n.opType }
func (n *flatParamNode) Attributes() map[string]interface{} { return nil }
func (n *flatParamNode) OutputShape() []int                 { return n.outShape }
func (n *flatParamNode) Parameters() []*graph.Parameter[float64] {
	return []*graph.Parameter[float64]{n.theta}
}

func (n *flatParamNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float64]) (*tensor.TensorNumeric[float64], error) {
	n.sync(n.theta.Value.Data())
	out, err := n.forward()
	if err != nil {
		return nil, err
	}
	return tensor.New[float64](n.outShape, out)
}

func (n *flatParamNode) Backward(_ context.Context, _ types.BackwardMode, upstream *tensor.TensorNumeric[float64], _ ...*tensor.TensorNumeric[float64]) ([]*tensor.TensorNumeric[float64], error) {
	n.sync(n.theta.Value.Data())
	grads, err := n.backward(upstream.Data())
	if err != nil {
		return nil, err
	}
	copy(n.theta.Gradient.Data(), grads)
	return nil, nil
}

// makeTimeMixerNode wraps a TimeMixer: output is the flattened prediction
// (predict over Forward's multi-scale output), backward maps the upstream
// dL/dpred onto per-scale trend gradients exactly as predict averages them,
// then runs the model's analytic backward.
func makeTimeMixerNode(cfg TimeMixerConfig) (graph.Node[float64], error) {
	m := NewTimeMixer(cfg, WithTimeMixerRNG(rand.New(rand.NewPCG(2026, 407))))

	rng := rand.New(rand.NewPCG(2026, 407))
	input := make([][]float64, cfg.NumFeatures)
	for f := range input {
		input[f] = make([]float64, cfg.InputLen)
		for i := range input[f] {
			input[f][i] = rng.NormFloat64() * 0.5
		}
	}

	forward := func() ([]float64, error) {
		out, err := m.Forward(input)
		if err != nil {
			return nil, err
		}
		pred := m.predict(&out.MultiScaleOutput)
		flat := make([]float64, 0, cfg.NumFeatures*cfg.OutputLen)
		for f := range pred {
			flat = append(flat, pred[f]...)
		}
		return flat, nil
	}

	backward := func(upstream []float64) ([]float64, error) {
		_, cache := m.forwardWithCache(input)

		numScales := cfg.NumScales
		dScales := make([]scaleDecomposition, numScales)
		for s := 0; s < numScales; s++ {
			dScales[s] = scaleDecomposition{
				trend:    make([][]float64, cfg.NumFeatures),
				seasonal: make([][]float64, cfg.NumFeatures),
			}
			for f := 0; f < cfg.NumFeatures; f++ {
				dScales[s].trend[f] = make([]float64, cfg.InputLen)
				dScales[s].seasonal[f] = make([]float64, cfg.InputLen)
			}
		}
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.OutputLen; i++ {
				srcIdx := cfg.InputLen - cfg.OutputLen + i
				if srcIdx < 0 {
					srcIdx = 0
				}
				dPred := upstream[f*cfg.OutputLen+i]
				for s := 0; s < numScales; s++ {
					dScales[s].trend[f][srcIdx] += dPred / float64(numScales)
				}
			}
		}

		grads := newTimeMixerGrads(m)
		m.backward(dScales, cache, &grads)
		return grads.collectGrads(m), nil
	}

	return newFlatParamNode(
		"TimeMixer",
		[]int{cfg.NumFeatures, cfg.OutputLen},
		m.FlatParams(),
		forward,
		backward,
	)
}

// makePatchTSTNode wraps PatchTST's pure-float64 CPU path (forwardF64 /
// backwardF64): output is the concatenated per-sample prediction over a
// deterministic batch of windows, backward accumulates per-sample analytic
// gradients. The engine backward is pinned to this CPU path by the existing
// parity tests.
func makePatchTSTNode(cfg PatchTSTConfig, batch, channels int) (graph.Node[float64], error) {
	m, err := NewPatchTST(cfg, nil, nil)
	if err != nil {
		return nil, err
	}
	params := m.extractParamsF64()

	windows := make([][][]float64, batch)
	for s := range windows {
		windows[s] = make([][]float64, channels)
		for c := range windows[s] {
			windows[s][c] = make([]float64, cfg.InputLength)
			for i := range windows[s][c] {
				windows[s][c][i] = float64(s*100+c*10+i+1) * 0.01
			}
		}
	}

	outDim := cfg.OutputDim
	forward := func() ([]float64, error) {
		flat := make([]float64, 0, batch*outDim)
		for s := range windows {
			flat = append(flat, m.forwardF64(windows[s], params)...)
		}
		return flat, nil
	}

	backward := func(upstream []float64) ([]float64, error) {
		acc := make([]float64, params.paramCount())
		for s := range windows {
			_, cache := m.forwardF64WithCache(windows[s], params)
			sGrads := m.backwardF64(upstream[s*outDim:(s+1)*outDim], params, cache)
			for i := range acc {
				acc[i] += sGrads[i]
			}
		}
		return acc, nil
	}

	return newFlatParamNode(
		"PatchTST",
		[]int{batch, outDim},
		params.flatParams(),
		forward,
		backward,
	)
}

// makeITransformerNode wraps the iTransformer float64 CPU path (forward /
// forwardWithCache / backward) over a deterministic batch of windows. The
// engine-batched backward is pinned to this CPU path by
// TestITransformer_BackwardBatchEngine_ParityWithCPU.
func makeITransformerNode(cfg ITransformerConfig, batch int) (graph.Node[float64], error) {
	m, err := NewITransformer(cfg, nil, nil)
	if err != nil {
		return nil, err
	}

	rng := rand.New(rand.NewPCG(99, 0))
	windows := make([][][]float64, batch)
	for s := range windows {
		windows[s] = make([][]float64, cfg.Channels)
		for c := range windows[s] {
			windows[s][c] = make([]float64, cfg.InputLen)
			for i := range windows[s][c] {
				windows[s][c][i] = rng.NormFloat64() * 0.5
			}
		}
	}

	outLen := cfg.OutputLen
	forward := func() ([]float64, error) {
		flat := make([]float64, 0, batch*cfg.Channels*outLen)
		for s := range windows {
			pred := m.forward(windows[s])
			for c := range pred {
				flat = append(flat, pred[c]...)
			}
		}
		return flat, nil
	}

	backward := func(upstream []float64) ([]float64, error) {
		acc := newITransformerGrads(cfg)
		for s := range windows {
			_, cache := m.forwardWithCache(windows[s])
			dOutput := make([][]float64, cfg.Channels)
			for c := 0; c < cfg.Channels; c++ {
				dOutput[c] = make([]float64, outLen)
				for o := 0; o < outLen; o++ {
					dOutput[c][o] = upstream[(s*cfg.Channels+c)*outLen+o]
				}
			}
			m.backward(dOutput, cache, &acc)
		}
		return acc.collectGrads(cfg), nil
	}

	return newFlatParamNode(
		"ITransformer",
		[]int{batch, cfg.Channels, outLen},
		m.FlatParams(),
		forward,
		backward,
	)
}

// timeseriesGradcheckOps is the zerfoo-side OpInfo registry for the
// timeseries model backward passes. Entries follow the registration pattern
// established by ztensor's gradcheck.Registry() (T1.1); they take no graph
// inputs because the differentiated quantities are the model parameters.
func timeseriesGradcheckOps() []gradcheck.OpInfo {
	tol := &gradcheck.Tolerance{Atol: 2e-6, Rtol: 1e-3}

	makeNode := func(make func() (graph.Node[float64], error)) func(compute.Engine[float64]) (graph.Node[float64], error) {
		return func(compute.Engine[float64]) (graph.Node[float64], error) { return make() }
	}

	return []gradcheck.OpInfo{
		{
			Name: "TimeMixer/backward", Seed: 1, Eps: 1e-5, Tol: tol,
			Make: makeNode(func() (graph.Node[float64], error) {
				return makeTimeMixerNode(TimeMixerConfig{
					InputLen: 8, OutputLen: 4, NumFeatures: 2,
					NumScales: 2, HiddenSize: 4, NumLayers: 1,
				})
			}),
		},
		{
			// HiddenSize=16: smaller hidden sizes regularly saturate the
			// mixing MLP's hidden ReLU neurons, and finite differences can
			// "wake" dead neurons that the analytic gradient correctly
			// reports as zero. See issue #351 for the full investigation.
			Name: "TimeMixer/backward-multilayer", Seed: 2, Eps: 1e-5, Tol: tol,
			Make: makeNode(func() (graph.Node[float64], error) {
				return makeTimeMixerNode(TimeMixerConfig{
					InputLen: 8, OutputLen: 4, NumFeatures: 2,
					NumScales: 3, HiddenSize: 16, NumLayers: 2,
				})
			}),
		},
		{
			Name: "PatchTST/backward-b1c1l1", Seed: 3, Eps: 1e-5, Tol: tol,
			Make: makeNode(func() (graph.Node[float64], error) {
				return makePatchTSTNode(PatchTSTConfig{
					InputLength: 8, PatchLength: 4, Stride: 4,
					DModel: 4, NHeads: 2, NLayers: 1, OutputDim: 2,
				}, 1, 1)
			}),
		},
		{
			Name: "PatchTST/backward-b3c1l1", Seed: 4, Eps: 1e-5, Tol: tol,
			Make: makeNode(func() (graph.Node[float64], error) {
				return makePatchTSTNode(PatchTSTConfig{
					InputLength: 8, PatchLength: 4, Stride: 4,
					DModel: 4, NHeads: 2, NLayers: 1, OutputDim: 2,
				}, 3, 1)
			}),
		},
		{
			Name: "PatchTST/backward-b2c2l2", Seed: 5, Eps: 1e-5, Tol: tol,
			Make: makeNode(func() (graph.Node[float64], error) {
				return makePatchTSTNode(PatchTSTConfig{
					InputLength: 8, PatchLength: 4, Stride: 4,
					DModel: 4, NHeads: 2, NLayers: 2, OutputDim: 2,
				}, 2, 2)
			}),
		},
		{
			Name: "ITransformer/backward", Seed: 6, Eps: 1e-5, Tol: tol,
			Make: makeNode(func() (graph.Node[float64], error) {
				return makeITransformerNode(ITransformerConfig{
					Channels: 2, InputLen: 4, OutputLen: 2,
					DModel: 4, DFF: 8, NHeads: 2, NLayers: 1,
				}, 3)
			}),
		},
	}
}

// TestTimeseriesBackward_Gradcheck verifies every registered timeseries
// backward pass against float64 central finite differences via ztensor's
// shared gradcheck harness.
func TestTimeseriesBackward_Gradcheck(t *testing.T) {
	ctx := context.Background()
	for _, op := range timeseriesGradcheckOps() {
		t.Run(op.Name, func(t *testing.T) {
			report, err := op.Run(ctx)
			if err != nil {
				t.Fatalf("gradcheck: %v", err)
			}
			if !report.OK() {
				t.Errorf("%s", report)
			}
			t.Logf("%s", report)
		})
	}
}
