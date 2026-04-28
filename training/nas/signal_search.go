package nas

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// SignalSearchConfig holds configuration for a NAS search over time-series
// signal model architectures.
type SignalSearchConfig struct {
	// NumTrials is the number of DARTS search trials to run. Each trial
	// randomly initializes the architecture parameters and runs bilevel
	// optimization for SearchSteps steps.
	NumTrials int
	// SearchSteps is the number of bilevel optimization steps per trial.
	SearchSteps int
	// WeightLR is the inner-loop learning rate for network weights.
	WeightLR float64
	// AlphaLR is the outer-loop learning rate for architecture parameters.
	AlphaLR float64
	// MaxParams is the maximum parameter budget for the discretized architecture.
	// Zero means no limit.
	MaxParams int64

	// PatchTST-like architecture dimensions.
	// InputFeatures is the number of input features per time step.
	InputFeatures int
	// PatchLen is the number of time steps per patch.
	PatchLen int
	// HorizonLen is the forecast horizon length.
	HorizonLen int
	// HiddenDim is the hidden dimension of the model.
	HiddenDim int
	// NumLayers is the number of stacked cells.
	NumLayers int

	// SearchSpace defines the DARTS search space. If nil, a default space
	// suitable for PatchTST-like architectures is used.
	SearchSpace *SearchSpace

	// Seed for reproducibility. Zero means non-deterministic.
	Seed uint64
}

// SignalSearchResult holds the result of a NAS search trial.
type SignalSearchResult struct {
	// Trial is the 0-based trial index.
	Trial int
	// Arch is the discretized architecture discovered in this trial.
	Arch *DiscretizedArch
	// Metric is the evaluation metric value (lower is better for loss,
	// higher is better for Sharpe ratio depending on usage).
	Metric float64
	// FinalLoss is the validation loss at the end of the search.
	FinalLoss float64
}

// SignalSearchOutput holds the complete output of RunSignalNAS.
type SignalSearchOutput struct {
	// Best is the result with the lowest validation loss across all trials.
	Best SignalSearchResult
	// AllResults contains results from every trial.
	AllResults []SignalSearchResult
	// ExportConfig is the GGUF export configuration derived from the search config.
	ExportConfig ExportConfig
}

// SignalDataProvider supplies training and validation data for the NAS search.
// Implementations can load from disk or generate synthetic data for testing.
type SignalDataProvider interface {
	// TrainBatch returns a (input, target) pair for the training split.
	TrainBatch() (input, target []float32, shape []int, err error)
	// ValBatch returns a (input, target) pair for the validation split.
	ValBatch() (input, target []float32, shape []int, err error)
}

// DefaultSignalSearchSpace returns the default DARTS search space for
// PatchTST-like signal models: 4 nodes with pooling, skip, and zero ops.
func DefaultSignalSearchSpace() *SearchSpace {
	return NewSearchSpaceWithOps(4, []OpType{
		OpAvgPool3x3,
		OpMaxPool3x3,
		OpSkipConnect,
		OpZero,
	})
}

// RunSignalNAS runs the full NAS search pipeline for time-series signal models.
// It performs multiple DARTS trials, discretizes the best architecture, and
// returns a result ready for GGUF export.
func RunSignalNAS(ctx context.Context, cfg SignalSearchConfig, data SignalDataProvider) (*SignalSearchOutput, error) {
	if err := validateSignalConfig(cfg); err != nil {
		return nil, err
	}

	space := cfg.SearchSpace
	if space == nil {
		space = DefaultSignalSearchSpace()
	}

	var rng *rand.Rand
	if cfg.Seed != 0 {
		rng = rand.New(rand.NewPCG(cfg.Seed, 0))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	results := make([]SignalSearchResult, 0, cfg.NumTrials)
	bestIdx := -1
	bestLoss := math.Inf(1)

	for trial := range cfg.NumTrials {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		result, err := runSignalTrial(ctx, engine, ops, space, cfg, data, rng, trial)
		if err != nil {
			return nil, fmt.Errorf("nas: trial %d: %w", trial, err)
		}

		results = append(results, result)
		if result.FinalLoss < bestLoss {
			bestLoss = result.FinalLoss
			bestIdx = len(results) - 1
		}
	}

	if bestIdx < 0 {
		return nil, errors.New("nas: no trials completed")
	}

	exportCfg := ExportConfig{
		ModelName:     "nas-signal",
		HiddenDim:     cfg.HiddenDim,
		NumLayers:     cfg.NumLayers,
		InputFeatures: cfg.InputFeatures,
		PatchLen:      cfg.PatchLen,
		HorizonLen:    cfg.HorizonLen,
	}

	return &SignalSearchOutput{
		Best:         results[bestIdx],
		AllResults:   results,
		ExportConfig: exportCfg,
	}, nil
}

func validateSignalConfig(cfg SignalSearchConfig) error {
	if cfg.NumTrials < 1 {
		return errors.New("nas: NumTrials must be at least 1")
	}
	if cfg.SearchSteps < 1 {
		return errors.New("nas: SearchSteps must be at least 1")
	}
	if cfg.WeightLR <= 0 {
		return errors.New("nas: WeightLR must be positive")
	}
	if cfg.AlphaLR <= 0 {
		return errors.New("nas: AlphaLR must be positive")
	}
	if cfg.InputFeatures < 1 {
		return errors.New("nas: InputFeatures must be at least 1")
	}
	if cfg.PatchLen < 1 {
		return errors.New("nas: PatchLen must be at least 1")
	}
	if cfg.HorizonLen < 1 {
		return errors.New("nas: HorizonLen must be at least 1")
	}
	if cfg.HiddenDim < 1 {
		return errors.New("nas: HiddenDim must be at least 1")
	}
	if cfg.NumLayers < 1 {
		return errors.New("nas: NumLayers must be at least 1")
	}
	return nil
}

// runSignalTrial executes a single DARTS search trial. It creates candidate
// operations from the search space, runs bilevel optimization, and discretizes
// the resulting architecture.
func runSignalTrial(
	ctx context.Context,
	engine compute.Engine[float32],
	ops numeric.Float32Ops,
	space *SearchSpace,
	cfg SignalSearchConfig,
	data SignalDataProvider,
	rng *rand.Rand,
	trial int,
) (SignalSearchResult, error) {
	numEdges := space.numEdges()
	numOps := len(space.Ops)

	// Build candidate operations per edge. Each edge gets its own DARTSLayer
	// with one candidate per op type. We flatten all edges into a single
	// search by creating one DARTSLayer with numEdges * numOps candidates,
	// but for simplicity we use a single DARTSLayer with numOps candidates
	// (simulating a single-edge search that is representative of the cell).
	candidates := make([]graph.Node[float32], numOps)
	for i := range numOps {
		// Each candidate is a simple scaling op that simulates the cost
		// profile of the operation type. We use (i+1)*0.5 as the scale to
		// differentiate candidates during optimization.
		scale := float32(i+1) * 0.5
		candidates[i] = &signalCandidate{
			opType: space.Ops[i],
			scale:  scale,
			engine: engine,
		}
	}

	layer, err := NewDARTSLayer[float32](engine, ops, candidates)
	if err != nil {
		return SignalSearchResult{}, err
	}

	// Randomly perturb alpha initialization for diversity across trials.
	alphaData := layer.Parameters()[0].Value.Data()
	for i := range alphaData {
		alphaData[i] = float32(rng.Float64()*0.2 - 0.1)
	}

	dartsCfg := DARTSOptimizerConfig[float32]{
		WeightLR: float32(cfg.WeightLR),
		AlphaLR:  float32(cfg.AlphaLR),
	}
	optimizer, err := NewDARTSOptimizer[float32](engine, ops, layer, dartsCfg)
	if err != nil {
		return SignalSearchResult{}, err
	}

	var finalLoss float32
	for step := range cfg.SearchSteps {
		select {
		case <-ctx.Done():
			return SignalSearchResult{}, ctx.Err()
		default:
		}

		trainData, trainTarget, trainShape, err := data.TrainBatch()
		if err != nil {
			return SignalSearchResult{}, fmt.Errorf("train batch at step %d: %w", step, err)
		}
		valData, valTarget, valShape, err := data.ValBatch()
		if err != nil {
			return SignalSearchResult{}, fmt.Errorf("val batch at step %d: %w", step, err)
		}

		trainInput, err := tensor.New[float32](trainShape, trainData)
		if err != nil {
			return SignalSearchResult{}, err
		}
		trainTgt, err := tensor.New[float32](trainShape, trainTarget)
		if err != nil {
			return SignalSearchResult{}, err
		}
		valInput, err := tensor.New[float32](valShape, valData)
		if err != nil {
			return SignalSearchResult{}, err
		}
		valTgt, err := tensor.New[float32](valShape, valTarget)
		if err != nil {
			return SignalSearchResult{}, err
		}

		if err := optimizer.Step(ctx, trainInput, trainTgt, valInput, valTgt); err != nil {
			return SignalSearchResult{}, fmt.Errorf("step %d: %w", step, err)
		}

		// Track final loss.
		if step == cfg.SearchSteps-1 {
			pred, err := layer.Forward(ctx, valInput)
			if err != nil {
				return SignalSearchResult{}, err
			}
			loss, _, err := signalMSELoss(ops, pred, valTgt)
			if err != nil {
				return SignalSearchResult{}, err
			}
			finalLoss = loss
		}
	}

	// Expand single-edge alpha to full cell alpha by replicating the learned
	// preference across all edges.
	singleAlpha := layer.Parameters()[0].Value.Data()
	fullAlpha := make([]float32, numEdges*numOps)
	for e := range numEdges {
		copy(fullAlpha[e*numOps:(e+1)*numOps], singleAlpha)
	}

	arch, err := Discretize[float32](fullAlpha, space, cfg.MaxParams)
	if err != nil {
		return SignalSearchResult{}, fmt.Errorf("discretize: %w", err)
	}

	// Compute a simple Sharpe-like metric from the loss trajectory.
	// For simulation, we use 1/(1+loss) so lower loss gives higher metric.
	metric := 1.0 / (1.0 + float64(finalLoss))

	return SignalSearchResult{
		Trial:     trial,
		Arch:      arch,
		Metric:    metric,
		FinalLoss: float64(finalLoss),
	}, nil
}

// signalCandidate is a simple candidate operation for signal NAS search.
// It scales the input by a fixed factor, simulating different operation costs.
type signalCandidate struct {
	opType OpType
	scale  float32
	engine compute.Engine[float32]
}

func (c *signalCandidate) OpType() string                          { return string(c.opType) }
func (c *signalCandidate) Attributes() map[string]interface{}      { return nil }
func (c *signalCandidate) Parameters() []*graph.Parameter[float32] { return nil }
func (c *signalCandidate) OutputShape() []int                      { return nil }

func (c *signalCandidate) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return c.engine.MulScalar(ctx, inputs[0], c.scale)
}

func (c *signalCandidate) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	grad, err := c.engine.MulScalar(ctx, dOut, c.scale)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[float32]{grad}, nil
}

// signalMSELoss computes MSE loss for signal model evaluation.
func signalMSELoss(ops numeric.Float32Ops, pred, target *tensor.TensorNumeric[float32]) (float32, *tensor.TensorNumeric[float32], error) {
	pData := pred.Data()
	tData := target.Data()
	n := len(pData)
	nT := ops.FromFloat64(float64(n))
	two := ops.FromFloat64(2.0)

	var loss float32
	gradData := make([]float32, n)
	for i := range pData {
		diff := ops.Sub(pData[i], tData[i])
		loss = ops.Add(loss, ops.Mul(diff, diff))
		gradData[i] = ops.Div(ops.Mul(two, diff), nT)
	}
	loss = ops.Div(loss, nT)

	grad, err := tensor.New[float32](pred.Shape(), gradData)
	if err != nil {
		return 0, nil, err
	}
	return loss, grad, nil
}

// SharpeRatio computes the Sharpe ratio from a series of returns.
// Returns 0 if there are fewer than 2 values or if the standard deviation is 0.
func SharpeRatio(returns []float64) float64 {
	n := len(returns)
	if n < 2 {
		return 0
	}

	var sum float64
	for _, r := range returns {
		sum += r
	}
	mean := sum / float64(n)

	var sumSq float64
	for _, r := range returns {
		d := r - mean
		sumSq += d * d
	}
	stddev := math.Sqrt(sumSq / float64(n-1))
	if stddev < 1e-12 {
		return 0
	}
	return mean / stddev
}
