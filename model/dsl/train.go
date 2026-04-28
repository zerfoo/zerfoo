// Package modeldsl provides training support for DSL-defined models.
package dsl

import (
	"errors"
	"fmt"
	"math"
)

// trainableLayer extends execLayer with backward pass and parameter access.
type trainableLayer interface {
	execLayer
	// backward computes gradients with respect to inputs and updates internal parameter gradients.
	// It returns the gradient with respect to the layer's input.
	backward(gradOutput []float64) ([]float64, error)
	// params returns all trainable parameters as flattened slices.
	// Each returned Param holds the values and accumulated gradients.
	params() []*Param
}

// Param holds a trainable parameter vector and its accumulated gradient.
type Param struct {
	Data []float64
	Grad []float64
}

// TrainConfig configures a training run.
type TrainConfig struct {
	Epochs       int
	LearningRate float64
}

// TrainResult holds the outcome of training.
type TrainResult struct {
	FinalLoss float64
	BestLoss  float64
	BestEpoch int
	EpochLoss []float64
}

// Sample is a single training example.
type Sample struct {
	Input  []float64
	Target []float64
}

// Train runs a training loop over the given samples using mean squared error loss.
// It performs gradient descent using the specified learning rate.
func (m *Model) Train(config TrainConfig, samples []Sample) (*TrainResult, error) {
	if config.Epochs <= 0 {
		return nil, errors.New("modeldsl: epochs must be positive")
	}
	if len(samples) == 0 {
		return nil, errors.New("modeldsl: at least one training sample is required")
	}
	if config.LearningRate <= 0 {
		return nil, errors.New("modeldsl: learning rate must be positive")
	}
	for i, s := range samples {
		if len(s.Input) != m.inputDim {
			return nil, fmt.Errorf("modeldsl: sample %d input has size %d, want %d", i, len(s.Input), m.inputDim)
		}
		if len(s.Target) != m.outputDim {
			return nil, fmt.Errorf("modeldsl: sample %d target has size %d, want %d", i, len(s.Target), m.outputDim)
		}
	}

	// Verify all layers are trainable.
	for _, name := range m.graph.order {
		if _, ok := m.execLayers[name].(trainableLayer); !ok {
			return nil, fmt.Errorf("modeldsl: layer %q does not support training", name)
		}
	}

	result := &TrainResult{
		EpochLoss: make([]float64, config.Epochs),
	}

	for epoch := 0; epoch < config.Epochs; epoch++ {
		var epochLoss float64

		for _, s := range samples {
			loss, err := m.trainStep(s.Input, s.Target, config.LearningRate)
			if err != nil {
				return nil, fmt.Errorf("modeldsl: epoch %d: %w", epoch, err)
			}
			epochLoss += loss
		}

		epochLoss /= float64(len(samples))
		result.EpochLoss[epoch] = epochLoss

		if epoch == 0 || epochLoss < result.BestLoss {
			result.BestLoss = epochLoss
			result.BestEpoch = epoch
		}
	}

	result.FinalLoss = result.EpochLoss[config.Epochs-1]
	return result, nil
}

// trainStep performs one forward-backward-update cycle on a single sample.
func (m *Model) trainStep(input, target []float64, lr float64) (float64, error) {
	// Forward pass — store activations per layer.
	activations := make(map[string][]float64, len(m.graph.order))
	for _, name := range m.graph.order {
		parentNames := m.graph.parents[name]

		var layerInput []float64
		if len(parentNames) == 0 {
			layerInput = input
		} else {
			layerInput = activations[parentNames[0]]
		}

		tl := m.execLayers[name].(trainableLayer)
		out, err := tl.forward(layerInput)
		if err != nil {
			return 0, fmt.Errorf("forward %q: %w", name, err)
		}
		activations[name] = out
	}

	// Compute MSE loss and its gradient.
	outputName := m.graph.outputs[len(m.graph.outputs)-1]
	output := activations[outputName]
	var loss float64
	gradOutput := make([]float64, len(output))
	n := float64(len(output))
	for i, o := range output {
		diff := o - target[i]
		loss += diff * diff
		gradOutput[i] = 2.0 * diff / n
	}
	loss /= n

	// Backward pass — reverse topological order.
	gradMap := make(map[string][]float64, len(m.graph.order))
	gradMap[outputName] = gradOutput

	for i := len(m.graph.order) - 1; i >= 0; i-- {
		name := m.graph.order[i]
		tl := m.execLayers[name].(trainableLayer)

		grad := gradMap[name]
		gradInput, err := tl.backward(grad)
		if err != nil {
			return 0, fmt.Errorf("backward %q: %w", name, err)
		}

		// Propagate gradient to parents.
		parentNames := m.graph.parents[name]
		if len(parentNames) > 0 {
			parent := parentNames[0]
			if existing, ok := gradMap[parent]; ok {
				for j := range existing {
					existing[j] += gradInput[j]
				}
			} else {
				gradMap[parent] = gradInput
			}
		}
	}

	// SGD parameter update.
	for _, name := range m.graph.order {
		tl := m.execLayers[name].(trainableLayer)
		for _, p := range tl.params() {
			for j := range p.Data {
				p.Data[j] -= lr * p.Grad[j]
				p.Grad[j] = 0
			}
		}
	}

	return loss, nil
}

// Parameters returns all trainable parameters of the model.
func (m *Model) Parameters() []*Param {
	var all []*Param
	for _, name := range m.graph.order {
		if tl, ok := m.execLayers[name].(trainableLayer); ok {
			all = append(all, tl.params()...)
		}
	}
	return all
}

// --- Trainable layer implementations ---

// Ensure existing layer types implement trainableLayer.
// We augment the existing types with backward and params methods.

// linearLayerT wraps linearLayer with training support.
type linearLayerT struct {
	weights   *Param
	bias      *Param
	inDim     int
	outDim    int
	lastInput []float64 // cached for backward
}

func newTrainableLinearLayer(inDim, outDim int) *linearLayerT {
	base := newLinearLayer(inDim, outDim)
	return &linearLayerT{
		weights: &Param{Data: base.weights, Grad: make([]float64, len(base.weights))},
		bias:    &Param{Data: base.bias, Grad: make([]float64, len(base.bias))},
		inDim:   inDim,
		outDim:  outDim,
	}
}

func (l *linearLayerT) forward(input []float64) ([]float64, error) {
	if len(input) != l.inDim {
		return nil, fmt.Errorf("linear: expected %d inputs, got %d", l.inDim, len(input))
	}
	l.lastInput = make([]float64, len(input))
	copy(l.lastInput, input)

	out := make([]float64, l.outDim)
	for j := 0; j < l.outDim; j++ {
		sum := l.bias.Data[j]
		for i := 0; i < l.inDim; i++ {
			sum += input[i] * l.weights.Data[i*l.outDim+j]
		}
		out[j] = sum
	}
	return out, nil
}

func (l *linearLayerT) backward(gradOutput []float64) ([]float64, error) {
	// Gradient w.r.t. bias: dL/db = gradOutput
	for j := range gradOutput {
		l.bias.Grad[j] += gradOutput[j]
	}

	// Gradient w.r.t. weights: dL/dW[i][j] = input[i] * gradOutput[j]
	for i := 0; i < l.inDim; i++ {
		for j := 0; j < l.outDim; j++ {
			l.weights.Grad[i*l.outDim+j] += l.lastInput[i] * gradOutput[j]
		}
	}

	// Gradient w.r.t. input: dL/dx[i] = sum_j(W[i][j] * gradOutput[j])
	gradInput := make([]float64, l.inDim)
	for i := 0; i < l.inDim; i++ {
		for j := 0; j < l.outDim; j++ {
			gradInput[i] += l.weights.Data[i*l.outDim+j] * gradOutput[j]
		}
	}
	return gradInput, nil
}

func (l *linearLayerT) params() []*Param {
	return []*Param{l.weights, l.bias}
}

// rmsnormLayerT wraps rmsnormLayer with training support.
type rmsnormLayerT struct {
	epsilon   float64
	lastInput []float64
	lastRMS   float64
}

func (l *rmsnormLayerT) forward(input []float64) ([]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("rmsnorm: empty input")
	}
	l.lastInput = make([]float64, len(input))
	copy(l.lastInput, input)

	var sumSq float64
	for _, v := range input {
		sumSq += v * v
	}
	l.lastRMS = math.Sqrt(sumSq/float64(len(input)) + l.epsilon)

	out := make([]float64, len(input))
	for i, v := range input {
		out[i] = v / l.lastRMS
	}
	return out, nil
}

func (l *rmsnormLayerT) backward(gradOutput []float64) ([]float64, error) {
	n := float64(len(l.lastInput))
	rms := l.lastRMS
	rms3 := rms * rms * rms

	gradInput := make([]float64, len(l.lastInput))

	// dL/dx_i = gradOutput_i / rms - x_i / (n * rms^3) * sum_j(x_j * gradOutput_j)
	var dotProd float64
	for j, x := range l.lastInput {
		dotProd += x * gradOutput[j]
	}

	for i, x := range l.lastInput {
		gradInput[i] = gradOutput[i]/rms - x*dotProd/(n*rms3)
	}
	return gradInput, nil
}

func (l *rmsnormLayerT) params() []*Param { return nil }

// siluLayerT wraps siluLayer with training support.
type siluLayerT struct {
	lastInput []float64
}

func (l *siluLayerT) forward(input []float64) ([]float64, error) {
	l.lastInput = make([]float64, len(input))
	copy(l.lastInput, input)

	out := make([]float64, len(input))
	for i, v := range input {
		out[i] = v * (1.0 / (1.0 + math.Exp(-v)))
	}
	return out, nil
}

func (l *siluLayerT) backward(gradOutput []float64) ([]float64, error) {
	gradInput := make([]float64, len(l.lastInput))
	for i, x := range l.lastInput {
		sig := 1.0 / (1.0 + math.Exp(-x))
		// d(silu)/dx = sig + x * sig * (1 - sig) = sig * (1 + x * (1 - sig))
		gradInput[i] = gradOutput[i] * sig * (1.0 + x*(1.0-sig))
	}
	return gradInput, nil
}

func (l *siluLayerT) params() []*Param { return nil }

// softmaxLayerT wraps softmaxLayer with training support.
type softmaxLayerT struct {
	lastOutput []float64
}

func (l *softmaxLayerT) forward(input []float64) ([]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("softmax: empty input")
	}
	maxVal := input[0]
	for _, v := range input[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	out := make([]float64, len(input))
	var sum float64
	for i, v := range input {
		out[i] = math.Exp(v - maxVal)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	l.lastOutput = make([]float64, len(out))
	copy(l.lastOutput, out)
	return out, nil
}

func (l *softmaxLayerT) backward(gradOutput []float64) ([]float64, error) {
	n := len(l.lastOutput)
	gradInput := make([]float64, n)

	// Jacobian-vector product for softmax:
	// dL/dx_i = s_i * (gradOutput_i - sum_j(s_j * gradOutput_j))
	var dot float64
	for j := 0; j < n; j++ {
		dot += l.lastOutput[j] * gradOutput[j]
	}
	for i := 0; i < n; i++ {
		gradInput[i] = l.lastOutput[i] * (gradOutput[i] - dot)
	}
	return gradInput, nil
}

func (l *softmaxLayerT) params() []*Param { return nil }

// buildTrainableLayer constructs a trainableLayer from a LayerDef.
func buildTrainableLayer(def LayerDef, inDim, outDim int) (trainableLayer, error) {
	switch def.Type {
	case LayerLinear:
		return newTrainableLinearLayer(inDim, outDim), nil
	case LayerRMSNorm:
		eps := 1e-6
		if v, ok := def.Params["epsilon"]; ok {
			f, err := toFloat64(v)
			if err != nil {
				return nil, fmt.Errorf("invalid epsilon: %w", err)
			}
			eps = f
		}
		return &rmsnormLayerT{epsilon: eps}, nil
	case LayerSiLU:
		return &siluLayerT{}, nil
	case LayerSoftmax:
		return &softmaxLayerT{}, nil
	case LayerAttention:
		// Attention training requires multi-layer backward which is complex.
		// For now, we use a simplified trainable attention.
		numHeads := 1
		if v, ok := def.Params["num_heads"]; ok {
			h, err := toInt(v)
			if err != nil {
				return nil, fmt.Errorf("invalid num_heads: %w", err)
			}
			if h <= 0 {
				return nil, fmt.Errorf("num_heads must be positive, got %d", h)
			}
			numHeads = h
		}
		if inDim%numHeads != 0 {
			return nil, fmt.Errorf("input dim %d not divisible by num_heads %d", inDim, numHeads)
		}
		return newTrainableAttentionLayer(inDim, numHeads), nil
	default:
		return nil, fmt.Errorf("unsupported layer type %q", def.Type)
	}
}

// attentionLayerT wraps attentionLayer with training support.
// For single-token attention, the output is wo(v) since softmax of a single
// score is 1.0, so we treat it as a linear transform for backward.
type attentionLayerT struct {
	numHeads int
	headDim  int
	dim      int
	wq       *linearLayerT
	wk       *linearLayerT
	wv       *linearLayerT
	wo       *linearLayerT
	lastV    []float64
}

func newTrainableAttentionLayer(dim, numHeads int) *attentionLayerT {
	headDim := dim / numHeads
	return &attentionLayerT{
		numHeads: numHeads,
		headDim:  headDim,
		dim:      dim,
		wq:       newTrainableLinearLayer(dim, dim),
		wk:       newTrainableLinearLayer(dim, dim),
		wv:       newTrainableLinearLayer(dim, dim),
		wo:       newTrainableLinearLayer(dim, dim),
	}
}

func (a *attentionLayerT) forward(input []float64) ([]float64, error) {
	_, err := a.wq.forward(input)
	if err != nil {
		return nil, err
	}
	_, err = a.wk.forward(input)
	if err != nil {
		return nil, err
	}
	v, err := a.wv.forward(input)
	if err != nil {
		return nil, err
	}
	a.lastV = v

	// Single-token: softmax of single score = 1.0, output = wo(v)
	return a.wo.forward(v)
}

func (a *attentionLayerT) backward(gradOutput []float64) ([]float64, error) {
	// Backward through wo
	gradV, err := a.wo.backward(gradOutput)
	if err != nil {
		return nil, err
	}

	// Backward through wv (gradV is the gradient w.r.t. v output)
	gradInput, err := a.wv.backward(gradV)
	if err != nil {
		return nil, err
	}

	// wq and wk contribute to score but in single-token case,
	// the gradient through softmax(1 score)=1.0 is zero w.r.t. the score.
	// Zero out their gradients for this step (they still accumulate from forward).
	for _, p := range a.wq.params() {
		for j := range p.Grad {
			p.Grad[j] = 0
		}
	}
	for _, p := range a.wk.params() {
		for j := range p.Grad {
			p.Grad[j] = 0
		}
	}

	return gradInput, nil
}

func (a *attentionLayerT) params() []*Param {
	var all []*Param
	all = append(all, a.wq.params()...)
	all = append(all, a.wk.params()...)
	all = append(all, a.wv.params()...)
	all = append(all, a.wo.params()...)
	return all
}

// BuildTrainable instantiates a trainable Model from the graph.
// Unlike Build, the returned Model supports backward pass and parameter updates.
func (g *ModelGraph) BuildTrainable(inputDim, outputDim int) (*Model, error) {
	if inputDim <= 0 {
		return nil, fmt.Errorf("modeldsl: inputDim must be positive, got %d", inputDim)
	}
	if outputDim <= 0 {
		return nil, fmt.Errorf("modeldsl: outputDim must be positive, got %d", outputDim)
	}

	// Resolve dimensions for each layer by propagating through the graph.
	dims := make(map[string]int, len(g.layers))
	for _, name := range g.order {
		def := g.layers[g.layerIndex[name]]
		parentNames := g.parents[name]

		var inDim int
		if len(parentNames) == 0 {
			inDim = inputDim
		} else {
			inDim = dims[parentNames[0]]
		}

		outDim, err := resolveOutputDim(def, inDim, outputDim, g.children[name])
		if err != nil {
			return nil, fmt.Errorf("modeldsl: layer %q: %w", name, err)
		}
		dims[name] = outDim
	}

	// Build trainable layers.
	execLayers := make(map[string]execLayer, len(g.layers))
	for _, name := range g.order {
		def := g.layers[g.layerIndex[name]]
		parentNames := g.parents[name]

		var inDim int
		if len(parentNames) == 0 {
			inDim = inputDim
		} else {
			inDim = dims[parentNames[0]]
		}

		layer, err := buildTrainableLayer(def, inDim, dims[name])
		if err != nil {
			return nil, fmt.Errorf("modeldsl: layer %q: %w", name, err)
		}
		execLayers[name] = layer
	}

	return &Model{
		graph:      g,
		execLayers: execLayers,
		dims:       dims,
		inputDim:   inputDim,
		outputDim:  outputDim,
	}, nil
}
