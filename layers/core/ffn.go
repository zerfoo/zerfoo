package core

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// FFN is a feed-forward network.
type FFN[T tensor.Numeric] struct {
	name         string
	noBias       bool
	useGELU      bool
	w1           *Dense[T]
	w2           *Dense[T]
	w3           *Dense[T]
	swiglu       *activations.SwiGLU[T]
	inputTensor  *tensor.TensorNumeric[T]
	w1Output     *tensor.TensorNumeric[T]
	w3Output     *tensor.TensorNumeric[T]
	swiGLUOutput *tensor.TensorNumeric[T]
	w2Output     *tensor.TensorNumeric[T]

	// Merged gate+up weight for single-GEMV decode optimization.
	mergedGateUp *tensor.TensorNumeric[T]
	gateDim      int // gate output dim (intermediateSize)
	upDim        int // up output dim (intermediateSize)
}

// FFNOpt is a functional option for configuring a FFN layer.
type FFNOpt[T tensor.Numeric] func(*FFN[T])

// WithSwiGLU enables SwiGLU activation.
func WithSwiGLU[T tensor.Numeric]() FFNOpt[T] {
	return func(f *FFN[T]) {
		f.swiglu = activations.NewSwiGLU[T](f.w1.linear.engine, f.w1.linear.ops)
	}
}

// WithGELU enables GELU activation instead of SwiGLU.
// The FFN computes: output = W2(GELU(W1(x)) * W3(x))
func WithGELU[T tensor.Numeric]() FFNOpt[T] {
	return func(f *FFN[T]) {
		f.useGELU = true
	}
}

// FFNConfig holds configuration for FFN layers.
type FFNConfig[T tensor.Numeric] struct {
}

// WithFFNNoBias disables bias for all layers in the FFN.
func WithFFNNoBias[T tensor.Numeric]() FFNOpt[T] {
	return func(f *FFN[T]) {
		f.noBias = true
	}
}

// NewFFN creates a new FFN layer.
func NewFFN[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputDim, hiddenDim, outputDim int,
	opts ...FFNOpt[T],
) (*FFN[T], error) {
	// Create Dense layers with bias by default (NewDense creates bias by default).
	w1, err := NewDense[T](name+"_w1", engine, ops, inputDim, hiddenDim)
	if err != nil {
		return nil, err
	}

	// W2 takes SwiGLU output, which is hiddenDim (SwiGLU halves the concatenated input)
	w2, err := NewDense[T](name+"_w2", engine, ops, hiddenDim, outputDim)
	if err != nil {
		return nil, err
	}

	w3, err := NewDense[T](name+"_w3", engine, ops, inputDim, hiddenDim)
	if err != nil {
		return nil, err
	}

	f := &FFN[T]{
		name:   name,
		w1:     w1,
		w2:     w2,
		w3:     w3,
		swiglu: activations.NewSwiGLU[T](engine, ops),
	}

	for _, opt := range opts {
		opt(f)
	}

	// If WithFFNNoBias was applied, disable bias on all Dense layers.
	if f.noBias {
		f.w1.bias = nil
		f.w2.bias = nil
		f.w3.bias = nil
	}

	return f, nil
}

// NewFFNFromDense constructs an FFN from pre-built Dense layers.
// Unlike NewFFN, this does NOT allocate random weight matrices — callers
// supply Dense layers that already reference pre-loaded weight tensors.
// This is the correct constructor for MoE expert FFNs where weights are
// sliced from a stacked tensor and must not trigger fresh allocations.
func NewFFNFromDense[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	w1, w2, w3 *Dense[T],
	opts ...FFNOpt[T],
) (*FFN[T], error) {
	f := &FFN[T]{
		name:   name,
		w1:     w1,
		w2:     w2,
		w3:     w3,
		swiglu: activations.NewSwiGLU[T](engine, ops),
	}
	for _, opt := range opts {
		opt(f)
	}
	if f.noBias {
		f.w1.bias = nil
		f.w2.bias = nil
		f.w3.bias = nil
	}
	return f, nil
}

// OpType returns the operation type of the layer.
func (f *FFN[T]) OpType() string {
	return "FFN"
}

// Attributes returns the attributes of the layer.
func (f *FFN[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{}
}

// OutputShape returns the output shape of the layer.
func (f *FFN[T]) OutputShape() []int {
	return f.w2.OutputShape()
}

// Forward computes the forward pass of the FFN.
func (f *FFN[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FFN requires exactly one input, got %d", len(inputs))
	}

	input := inputs[0]
	f.inputTensor = input // Cache for backward pass

	// Detect sequence length for decode optimization.
	inputShape := input.Shape()
	seqLen := 1
	if len(inputShape) >= 3 {
		seqLen = inputShape[1]
	}

	var w1Output, w3Output *tensor.TensorNumeric[T]
	if f.mergedGateUp != nil && seqLen == 1 {
		// Merged gate+up: single GEMV + split for decode.
		merged, mergeErr := f.w1.linear.engine.MatMul(ctx, input, f.mergedGateUp)
		if mergeErr != nil {
			return nil, fmt.Errorf("merged gate+up MatMul: %w", mergeErr)
		}
		var splitErr error
		w1Output, w3Output, splitErr = splitMergedGateUp[T](merged, f.gateDim, f.upDim)
		if splitErr != nil {
			return nil, fmt.Errorf("split merged gate+up: %w", splitErr)
		}
	} else {
		var err error
		w1Output, err = f.w1.Forward(ctx, input)
		if err != nil {
			return nil, err
		}
		w3Output, err = f.w3.Forward(ctx, input)
		if err != nil {
			return nil, err
		}
	}
	f.w1Output = w1Output // Cache for backward pass
	f.w3Output = w3Output // Cache for backward pass

	var activationOutput *tensor.TensorNumeric[T]
	if f.useGELU {
		// GELU path: output = GELU(W1(x)) * W3(x)
		geluOutput, geluErr := geluForward(ctx, f.w1.linear.engine, f.w1.linear.ops, w1Output)
		if geluErr != nil {
			return nil, fmt.Errorf("GELU activation: %w", geluErr)
		}
		gated, mulErr := f.w1.linear.engine.Mul(ctx, geluOutput, w3Output)
		if mulErr != nil {
			return nil, fmt.Errorf("GELU gate multiply: %w", mulErr)
		}
		activationOutput = gated
	} else {
		// SwiGLU path: try fused GPU path first, fall back to CPU.
		// Unwrap EngineProxy to detect the real engine type.
		realEngine := compute.Engine[T](f.w1.linear.engine)
		if proxy, ok := f.w1.linear.engine.(*compute.EngineProxy[T]); ok {
			realEngine = proxy.Real()
		}
		if provider, ok := realEngine.(compute.FusedSwiGLUProvider[T]); ok {
			out, fusedErr := provider.GPUFusedSwiGLU(w1Output, w3Output)
			if fusedErr == nil {
				activationOutput = out
			}
		}
		if activationOutput == nil {
			swigluInput, concatErr := f.w1.linear.engine.Concat(ctx, []*tensor.TensorNumeric[T]{w1Output, w3Output}, -1)
			if concatErr != nil {
				return nil, fmt.Errorf("failed to concatenate tensors for SwiGLU input: %w", concatErr)
			}
			swigluOut, swigluErr := f.swiglu.Forward(ctx, swigluInput)
			if swigluErr != nil {
				return nil, swigluErr
			}
			activationOutput = swigluOut
		}
	}
	f.swiGLUOutput = activationOutput // Cache for backward pass

	w2Output, err := f.w2.Forward(ctx, activationOutput)
	if err != nil {
		return nil, err
	}
	f.w2Output = w2Output // Cache for backward pass

	return w2Output, nil
}

// Backward computes the backward pass of the FFN.
func (f *FFN[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Backward through W2
	dSwiGLUOutput, err := f.w2.Backward(ctx, mode, dOut, f.swiGLUOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to backward through w2: %w", err)
	}

	// Concatenate w1Output and w3Output to reconstruct swigluInput for backward
	swigluInput, err := f.w1.linear.engine.Concat(ctx, []*tensor.TensorNumeric[T]{f.w1Output, f.w3Output}, -1)
	if err != nil {
		return nil, fmt.Errorf("failed to concatenate tensors for SwiGLU backward: %w", err)
	}

	// Check if dSwiGLUOutput has at least one element
	if len(dSwiGLUOutput) == 0 {
		return nil, fmt.Errorf("no gradients from w2 backward pass")
	}

	// Backward through SwiGLU
	dSwiGLUInputs, err := f.swiglu.Backward(ctx, mode, dSwiGLUOutput[0], swigluInput)
	if err != nil {
		return nil, fmt.Errorf("failed to backward through swiglu: %w", err)
	}

	// SwiGLU returns a single concatenated gradient, split it back into two parts
	if len(dSwiGLUInputs) != 1 {
		return nil, fmt.Errorf("expected 1 concatenated gradient from SwiGLU backward, got %d", len(dSwiGLUInputs))
	}

	// Split the concatenated gradient back into dW1Output and dW3Output
	splitGrads, err := f.w1.linear.engine.Split(ctx, dSwiGLUInputs[0], 2, len(dSwiGLUInputs[0].Shape())-1)
	if err != nil {
		return nil, fmt.Errorf("failed to split SwiGLU gradients: %w", err)
	}

	dW1Output := splitGrads[0]
	dW3Output := splitGrads[1]

	// Backward through W1
	dInputW1, err := f.w1.Backward(ctx, mode, dW1Output, f.inputTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to backward through w1: %w", err)
	}

	// Backward through W3
	dInputW3, err := f.w3.Backward(ctx, mode, dW3Output, f.inputTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to backward through w3: %w", err)
	}

	// Sum gradients from W1 and W3
	dInput, err := f.w1.linear.engine.Add(ctx, dInputW1[0], dInputW3[0])
	if err != nil {
		return nil, fmt.Errorf("failed to sum gradients: %w", err)
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// SetMergedGateUp sets a merged gate+up weight tensor for single-GEMV decode optimization.
// During decode (seqLen=1), a single MatMul replaces two separate gate and up projections.
func (f *FFN[T]) SetMergedGateUp(weight *tensor.TensorNumeric[T], gateDim, upDim int) {
	f.mergedGateUp = weight
	f.gateDim = gateDim
	f.upDim = upDim
}

// MergedGateUpParameter returns the merged gate+up parameter for GPU upload, or nil if not set.
func (f *FFN[T]) MergedGateUpParameter() *graph.Parameter[T] {
	if f.mergedGateUp == nil {
		return nil
	}
	return &graph.Parameter[T]{Name: f.name + "_merged_gate_up", Value: f.mergedGateUp}
}

// Parameters returns the parameters of the layer.
func (f *FFN[T]) Parameters() []*graph.Parameter[T] {
	params := f.w1.Parameters()
	params = append(params, f.w2.Parameters()...)
	params = append(params, f.w3.Parameters()...)
	if p := f.MergedGateUpParameter(); p != nil {
		params = append(params, p)
	}
	return params
}

// geluForward applies GELU activation using engine primitives.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func geluForward[T tensor.Numeric](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T], x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x2, err := engine.Mul(ctx, x, x)
	if err != nil {
		return nil, err
	}
	x3, err := engine.Mul(ctx, x2, x)
	if err != nil {
		return nil, err
	}
	cubicTerm, err := engine.MulScalar(ctx, x3, ops.FromFloat64(0.044715))
	if err != nil {
		return nil, err
	}
	inner, err := engine.Add(ctx, x, cubicTerm)
	if err != nil {
		return nil, err
	}
	scaled, err := engine.MulScalar(ctx, inner, ops.FromFloat64(math.Sqrt(2/math.Pi)))
	if err != nil {
		return nil, err
	}
	tanhResult, err := engine.Tanh(ctx, scaled)
	if err != nil {
		return nil, err
	}
	onePlusTanh, err := engine.AddScalar(ctx, tanhResult, ops.One())
	if err != nil {
		return nil, err
	}
	xTimesOnePlusTanh, err := engine.Mul(ctx, x, onePlusTanh)
	if err != nil {
		return nil, err
	}
	return engine.MulScalar(ctx, xTimesOnePlusTanh, ops.FromFloat64(0.5))
}

// splitMergedGateUp splits a merged gate+up output into separate gate and up tensors.
// For GPU-resident tensors, uses zero-copy views.
func splitMergedGateUp[T tensor.Numeric](merged *tensor.TensorNumeric[T], gateDim, upDim int) (gate, up *tensor.TensorNumeric[T], err error) {
	shape := merged.Shape()
	if len(shape) < 2 {
		return nil, nil, fmt.Errorf("expected at least 2D tensor, got %dD", len(shape))
	}
	lastDim := shape[len(shape)-1]
	if lastDim != gateDim+upDim {
		return nil, nil, fmt.Errorf("last dim %d != gateDim(%d)+upDim(%d)", lastDim, gateDim, upDim)
	}

	prefix := make([]int, len(shape)-1)
	copy(prefix, shape[:len(shape)-1])
	batchElems := 1
	for _, d := range prefix {
		batchElems *= d
	}

	gateShape := append(append([]int{}, prefix...), gateDim)
	upShape := append(append([]int{}, prefix...), upDim)

	// GPU path: zero-copy views.
	if gs, ok := merged.GetStorage().(*tensor.GPUStorage[T]); ok {
		gateView := tensor.NewGPUStorageView(gs, 0, batchElems*gateDim)
		upView := tensor.NewGPUStorageView(gs, batchElems*gateDim, batchElems*upDim)

		gate, err = tensor.NewWithStorage[T](gateShape, gateView)
		if err != nil {
			return nil, nil, err
		}
		up, err = tensor.NewWithStorage[T](upShape, upView)
		if err != nil {
			return nil, nil, err
		}
		return gate, up, nil
	}

	// CPU path: copy data.
	data := merged.Data()
	gateData := make([]T, batchElems*gateDim)
	upData := make([]T, batchElems*upDim)
	for b := 0; b < batchElems; b++ {
		off := b * lastDim
		copy(gateData[b*gateDim:(b+1)*gateDim], data[off:off+gateDim])
		copy(upData[b*upDim:(b+1)*upDim], data[off+gateDim:off+gateDim+upDim])
	}
	gate, err = tensor.New(gateShape, gateData)
	if err != nil {
		return nil, nil, err
	}
	up, err = tensor.New(upShape, upData)
	if err != nil {
		return nil, nil, err
	}
	return gate, up, nil
}
