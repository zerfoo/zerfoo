package meta

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// cpuEngine is a package-level CPU engine used for linear forward passes.
var cpuEngine = compute.NewCPUEngine[float64](numeric.Float64Ops{})

// Task provides train/test data splits for a single meta-learning task.
type Task interface {
	// TrainData returns the support set (inputs and targets) for adaptation.
	TrainData() (inputs [][]float64, targets []float64)
	// TestData returns the query set (inputs and targets) for evaluation.
	TestData() (inputs [][]float64, targets []float64)
}

// MAMLConfig holds hyperparameters for MAML meta-learning.
type MAMLConfig struct {
	// InnerLR is the learning rate for task-specific adaptation (inner loop).
	InnerLR float64
	// OuterLR is the learning rate for meta-parameter updates (outer loop).
	OuterLR float64
	// InnerSteps is the number of gradient steps in the inner loop.
	InnerSteps int
	// NTasksPerBatch is the number of tasks sampled per meta-update.
	NTasksPerBatch int
	// MetaEpochs is the number of outer-loop iterations.
	MetaEpochs int
	// HiddenDims specifies the hidden layer sizes for the internal MLP.
	HiddenDims []int
	// Seed, when non-nil, seeds the random number generator for reproducible
	// weight initialization and task sampling. If nil, a non-deterministic
	// source is used.
	Seed *uint64
}

// AdaptedModel represents a model adapted to a specific task.
type AdaptedModel struct {
	weights [][]float64 // layer weight matrices, shape [out, in] (flattened row-major)
	biases  [][]float64 // layer bias vectors
	dims    []int       // layer dimensions: [input, hidden..., output]
}

// Predict runs inference on the adapted model for a single input.
func (a *AdaptedModel) Predict(input []float64) (float64, error) {
	if len(input) != a.dims[0] {
		return 0, fmt.Errorf("meta: predict: expected %d features, got %d", a.dims[0], len(input))
	}
	x := input
	for i := 0; i < len(a.weights); i++ {
		out, err := linearFwd(x, a.weights[i], a.biases[i], a.dims[i], a.dims[i+1])
		if err != nil {
			return 0, fmt.Errorf("meta: predict: layer %d: %w", i, err)
		}
		x = out
		if i < len(a.weights)-1 {
			x = relu(x)
		}
	}
	return x[0], nil
}

// MAML implements Model-Agnostic Meta-Learning.
type MAML struct {
	config  MAMLConfig
	weights [][]float64 // meta-parameters: layer weight matrices, shape [out, in]
	biases  [][]float64 // meta-parameters: layer bias vectors
	dims    []int       // layer dimensions
	rng     *rand.Rand  // random source for init and sampling
}

// NewMAML creates a new MAML instance with randomly initialized meta-parameters.
func NewMAML(config MAMLConfig) (*MAML, error) {
	if config.InnerLR <= 0 {
		return nil, fmt.Errorf("meta: InnerLR must be positive, got %f", config.InnerLR)
	}
	if config.OuterLR <= 0 {
		return nil, fmt.Errorf("meta: OuterLR must be positive, got %f", config.OuterLR)
	}
	if config.InnerSteps <= 0 {
		return nil, fmt.Errorf("meta: InnerSteps must be positive, got %d", config.InnerSteps)
	}
	if config.NTasksPerBatch <= 0 {
		return nil, fmt.Errorf("meta: NTasksPerBatch must be positive, got %d", config.NTasksPerBatch)
	}
	if config.MetaEpochs <= 0 {
		return nil, fmt.Errorf("meta: MetaEpochs must be positive, got %d", config.MetaEpochs)
	}
	if len(config.HiddenDims) == 0 {
		return nil, fmt.Errorf("meta: HiddenDims must have at least one element")
	}
	for i, h := range config.HiddenDims {
		if h <= 0 {
			return nil, fmt.Errorf("meta: HiddenDims[%d] must be positive, got %d", i, h)
		}
	}

	var rng *rand.Rand
	if config.Seed != nil {
		rng = rand.New(rand.NewPCG(*config.Seed, 0))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	return &MAML{config: config, rng: rng}, nil
}

// MetaTrain runs the MAML meta-training loop across the given tasks.
// It initializes the network dimensions from the first task's data and
// runs MetaEpochs outer-loop iterations, each sampling NTasksPerBatch tasks.
func (m *MAML) MetaTrain(tasks []Task, config MAMLConfig) error {
	if len(tasks) == 0 {
		return fmt.Errorf("meta: MetaTrain: no tasks provided")
	}

	// Determine input/output dimensions from the first task.
	inputs, targets := tasks[0].TrainData()
	if len(inputs) == 0 {
		return fmt.Errorf("meta: MetaTrain: first task has no training data")
	}
	inputDim := len(inputs[0])
	_ = targets // output dim is always 1 for regression

	// Build dims: [inputDim, hidden..., 1]
	m.dims = make([]int, 0, len(config.HiddenDims)+2)
	m.dims = append(m.dims, inputDim)
	m.dims = append(m.dims, config.HiddenDims...)
	m.dims = append(m.dims, 1)

	// Initialize meta-parameters with He initialization.
	m.initWeights()

	// Outer loop: meta-training.
	for epoch := 0; epoch < config.MetaEpochs; epoch++ {
		// Sample tasks for this meta-batch.
		sampled := m.sampleTasks(tasks, config.NTasksPerBatch)

		// Accumulate meta-gradients across tasks.
		metaGradW := zeroLike(m.weights)
		metaGradB := zeroLike(m.biases)

		for _, task := range sampled {
			// Clone meta-parameters for inner-loop adaptation.
			w := cloneParams(m.weights)
			b := cloneParams(m.biases)

			// Inner loop: adapt to this task.
			trainIn, trainTgt := task.TrainData()
			for step := 0; step < config.InnerSteps; step++ {
				gw, gb := computeGradients(w, b, m.dims, trainIn, trainTgt)
				for i := range w {
					for j := range w[i] {
						w[i][j] -= config.InnerLR * gw[i][j]
					}
					for j := range b[i] {
						b[i][j] -= config.InnerLR * gb[i][j]
					}
				}
			}

			// Evaluate adapted parameters on the test set.
			testIn, testTgt := task.TestData()
			gw, gb := computeGradients(w, b, m.dims, testIn, testTgt)

			// Accumulate meta-gradients (first-order MAML approximation).
			for i := range metaGradW {
				for j := range metaGradW[i] {
					metaGradW[i][j] += gw[i][j]
				}
				for j := range metaGradB[i] {
					metaGradB[i][j] += gb[i][j]
				}
			}
		}

		// Meta-update: apply averaged meta-gradients.
		scale := 1.0 / float64(len(sampled))
		for i := range m.weights {
			for j := range m.weights[i] {
				m.weights[i][j] -= config.OuterLR * scale * metaGradW[i][j]
			}
			for j := range m.biases[i] {
				m.biases[i][j] -= config.OuterLR * scale * metaGradB[i][j]
			}
		}
	}

	return nil
}

// Adapt takes the current meta-parameters and adapts them to a new task
// using the specified number of inner-loop gradient steps.
func (m *MAML) Adapt(task Task, steps int) *AdaptedModel {
	w := cloneParams(m.weights)
	b := cloneParams(m.biases)

	trainIn, trainTgt := task.TrainData()
	for step := 0; step < steps; step++ {
		gw, gb := computeGradients(w, b, m.dims, trainIn, trainTgt)
		for i := range w {
			for j := range w[i] {
				w[i][j] -= m.config.InnerLR * gw[i][j]
			}
			for j := range b[i] {
				b[i][j] -= m.config.InnerLR * gb[i][j]
			}
		}
	}

	return &AdaptedModel{
		weights: w,
		biases:  b,
		dims:    m.dims,
	}
}

// MetaLoss computes the average loss across tasks after inner-loop adaptation.
func (m *MAML) MetaLoss(tasks []Task) float64 {
	if len(tasks) == 0 {
		return 0
	}
	var total float64
	for _, task := range tasks {
		adapted := m.Adapt(task, m.config.InnerSteps)
		testIn, testTgt := task.TestData()
		total += mseLoss(adapted.weights, adapted.biases, m.dims, testIn, testTgt)
	}
	return total / float64(len(tasks))
}

// initWeights initializes all layer weights using He (Kaiming) initialization
// and zero biases. Weights are stored in [out, in] layout to match
// functional.Linear convention. Random values are generated in [in, out] order
// then transposed to preserve deterministic seeded behavior.
func (m *MAML) initWeights() {
	nLayers := len(m.dims) - 1
	m.weights = make([][]float64, nLayers)
	m.biases = make([][]float64, nLayers)

	for i := 0; i < nLayers; i++ {
		in, out := m.dims[i], m.dims[i+1]
		scale := math.Sqrt(2.0 / float64(in))
		// Generate in [in, out] order for seed compatibility, then transpose.
		tmp := make([]float64, in*out)
		for j := range tmp {
			tmp[j] = m.rng.NormFloat64() * scale
		}
		w := make([]float64, out*in)
		for k := 0; k < in; k++ {
			for j := 0; j < out; j++ {
				w[j*in+k] = tmp[k*out+j]
			}
		}
		m.weights[i] = w
		m.biases[i] = make([]float64, out)
	}
}

// linearFwd computes y = x @ W^T + b via functional.Linear.
// W is stored row-major with shape [out, in].
func linearFwd(x []float64, w []float64, b []float64, in, out int) ([]float64, error) {
	ctx := context.Background()
	xT, err := tensor.New[float64]([]int{1, in}, x)
	if err != nil {
		return nil, err
	}
	wT, err := tensor.New[float64]([]int{out, in}, w)
	if err != nil {
		return nil, err
	}
	bT, err := tensor.New[float64]([]int{out}, b)
	if err != nil {
		return nil, err
	}
	result, err := functional.Linear(ctx, cpuEngine, xT, wT, bT)
	if err != nil {
		return nil, err
	}
	return result.Data(), nil
}

// relu applies ReLU activation element-wise.
func relu(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

// forward runs the full MLP forward pass for a single sample, returning
// all pre-activation and post-activation intermediate values for backprop.
func forward(w, b [][]float64, dims []int, input []float64) (preActs, postActs [][]float64, output float64) {
	nLayers := len(w)
	preActs = make([][]float64, nLayers)
	postActs = make([][]float64, nLayers)

	x := input
	for i := 0; i < nLayers; i++ {
		pre, err := linearFwd(x, w[i], b[i], dims[i], dims[i+1])
		if err != nil {
			panic(fmt.Sprintf("meta: forward: layer %d: %v", i, err))
		}
		preActs[i] = pre
		if i < nLayers-1 {
			post := relu(pre)
			postActs[i] = post
			x = post
		} else {
			postActs[i] = pre
			x = pre
		}
	}
	return preActs, postActs, x[0]
}

// mseLoss computes mean squared error over a dataset.
func mseLoss(w, b [][]float64, dims []int, inputs [][]float64, targets []float64) float64 {
	n := len(inputs)
	if n == 0 {
		return 0
	}
	var total float64
	for i := 0; i < n; i++ {
		_, _, pred := forward(w, b, dims, inputs[i])
		diff := pred - targets[i]
		total += diff * diff
	}
	return total / float64(n)
}

// computeGradients computes the average gradients of MSE loss w.r.t. weights
// and biases over the given dataset using backpropagation.
func computeGradients(w, b [][]float64, dims []int, inputs [][]float64, targets []float64) ([][]float64, [][]float64) {
	nLayers := len(w)
	gw := zeroLike(w)
	gb := zeroLike(b)

	n := len(inputs)
	if n == 0 {
		return gw, gb
	}

	for s := 0; s < n; s++ {
		preActs, postActs, pred := forward(w, b, dims, inputs[s])

		// dL/dpred = 2*(pred - target) / n
		dOut := 2 * (pred - targets[s]) / float64(n)

		// Backprop through layers in reverse.
		dX := []float64{dOut}
		for i := nLayers - 1; i >= 0; i-- {
			in := dims[i]
			out := dims[i+1]

			// Input to this layer.
			var layerInput []float64
			if i > 0 {
				layerInput = postActs[i-1]
			} else {
				layerInput = inputs[s]
			}

			// Apply ReLU gradient for hidden layers.
			if i < nLayers-1 {
				for j := 0; j < out; j++ {
					if preActs[i][j] <= 0 {
						dX[j] = 0
					}
				}
			}

			// dW[i] += dX * layerInput^T (weight shape is [out, in])
			for j := 0; j < out; j++ {
				for k := 0; k < in; k++ {
					gw[i][j*in+k] += layerInput[k] * dX[j]
				}
			}

			// dB[i] += dX
			for j := 0; j < out; j++ {
				gb[i][j] += dX[j]
			}

			// Propagate gradient to previous layer: dX_prev = W^T @ dX
			// With weight shape [out, in], W^T[k,j] = W[j,k] = w[i][j*in+k].
			if i > 0 {
				dPrev := make([]float64, in)
				for k := 0; k < in; k++ {
					var sum float64
					for j := 0; j < out; j++ {
						sum += dX[j] * w[i][j*in+k]
					}
					dPrev[k] = sum
				}
				dX = dPrev
			}
		}
	}

	return gw, gb
}

// cloneParams creates a deep copy of parameter slices.
func cloneParams(params [][]float64) [][]float64 {
	out := make([][]float64, len(params))
	for i, p := range params {
		c := make([]float64, len(p))
		copy(c, p)
		out[i] = c
	}
	return out
}

// zeroLike creates zero-valued slices with the same shape as the input.
func zeroLike(params [][]float64) [][]float64 {
	out := make([][]float64, len(params))
	for i, p := range params {
		out[i] = make([]float64, len(p))
	}
	return out
}

// sampleTasks randomly samples n tasks (with replacement) from the pool.
func (m *MAML) sampleTasks(tasks []Task, n int) []Task {
	sampled := make([]Task, n)
	for i := 0; i < n; i++ {
		sampled[i] = tasks[m.rng.IntN(len(tasks))]
	}
	return sampled
}
