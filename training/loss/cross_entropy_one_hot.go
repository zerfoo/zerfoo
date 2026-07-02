package loss

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// logFloor is added to the softmax probabilities before the Log so a class
// probability that underflows to exactly 0 in the element type produces a
// large-but-finite negative log instead of -Inf. For float32 the smallest
// normal is ~1.18e-38, so log(p+1e-40) differs from log(p) only when p is
// already far below any numerically meaningful probability.
const logFloor = 1e-40

// CrossEntropyLossOneHot computes the mean cross-entropy loss against
// one-hot (or soft) target distributions using engine ops only.
//
// Unlike CrossEntropyLoss, which computes the loss value on the host in
// float64 (reading the logits back mid-step) and converts the integer
// targets on the host each call, every Forward/Backward operation here is
// an engine op on the input tensors. This makes the node usable inside a
// CUDA-graph capture region (compute.GraphCapturer): no .Data() reads, no
// host-side math, no per-call host tensor conversions. See
// training.CaptureReplayRunner.
//
// Inputs: predictions (logits, shape [..., C]) and targets as a
// PRE-ENCODED one-hot tensor of the same shape [..., C]. Callers that hold
// integer labels encode them once up front (and, on GPU engines, upload
// them once) instead of per step.
//
// Gradient semantics are identical to CrossEntropyLoss.Backward:
// dL/dlogits = (softmax(logits) - onehot) / positions * dOut, where
// positions is the number of class distributions (total elements / C).
// The loss VALUE is computed as -mean(sum(onehot * log(softmax + floor)))
// in the element type T, which matches CrossEntropyLoss's host-float64
// fused log-softmax up to rounding in T.
type CrossEntropyLossOneHot[T tensor.Numeric] struct {
	engine compute.Engine[T]

	// Cached tensors for backward.
	softmaxOutput *tensor.TensorNumeric[T]
	onehot        *tensor.TensorNumeric[T]
	outputShape   []int
}

// NewCrossEntropyLossOneHot creates a CrossEntropyLossOneHot node.
func NewCrossEntropyLossOneHot[T tensor.Numeric](engine compute.Engine[T]) *CrossEntropyLossOneHot[T] {
	return &CrossEntropyLossOneHot[T]{engine: engine}
}

// OutputShape returns the output shape of the loss (a scalar, [1]).
func (cel *CrossEntropyLossOneHot[T]) OutputShape() []int {
	return cel.outputShape
}

// Parameters returns nil: the loss has no trainable parameters.
func (cel *CrossEntropyLossOneHot[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the mean cross-entropy loss from logits and one-hot
// targets via engine ops only.
func (cel *CrossEntropyLossOneHot[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("CrossEntropyLossOneHot expects 2 inputs (logits, onehot), got %d", len(inputs))
	}
	predictions := inputs[0]
	onehot := inputs[1]

	pShape := predictions.Shape()
	if !tensor.ShapesEqual(pShape, onehot.Shape()) {
		return nil, fmt.Errorf("CrossEntropyLossOneHot: logits shape %v != one-hot targets shape %v",
			pShape, onehot.Shape())
	}
	if len(pShape) == 0 {
		return nil, fmt.Errorf("CrossEntropyLossOneHot: logits must have at least 1 dimension")
	}
	classes := pShape[len(pShape)-1]
	if classes <= 0 {
		return nil, fmt.Errorf("CrossEntropyLossOneHot: class dimension must be positive, got %d", classes)
	}
	positions := 1
	for _, d := range pShape[:len(pShape)-1] {
		positions *= d
	}
	if positions <= 0 {
		return nil, fmt.Errorf("CrossEntropyLossOneHot: no class distributions in shape %v", pShape)
	}

	cel.outputShape = []int{1}
	cel.onehot = onehot

	// Numerically stable softmax (engine kernels subtract the row max).
	softmaxOutput, err := cel.engine.Softmax(ctx, predictions, len(pShape)-1, nil)
	if err != nil {
		return nil, err
	}
	cel.softmaxOutput = softmaxOutput

	// log(softmax + floor): the floor keeps a fully-saturated wrong class
	// finite. See logFloor.
	ops := cel.engine.Ops()
	safe, err := cel.engine.AddScalar(ctx, softmaxOutput, ops.FromFloat64(logFloor))
	if err != nil {
		return nil, err
	}
	logProbs, err := cel.engine.Log(ctx, safe)
	if err != nil {
		return nil, err
	}

	// Pick out the target log-probs and reduce to a scalar:
	// sum(onehot * logProbs) over every axis, highest to lowest.
	picked, err := cel.engine.Mul(ctx, logProbs, onehot)
	if err != nil {
		return nil, err
	}
	total := picked
	for axis := len(pShape) - 1; axis >= 0; axis-- {
		total, err = cel.engine.ReduceSum(ctx, total, axis, false)
		if err != nil {
			return nil, err
		}
	}

	// loss = -(1/positions) * total, as a [1] tensor.
	loss, err := cel.engine.MulScalar(ctx, total, ops.FromFloat64(-1.0/float64(positions)))
	if err != nil {
		return nil, err
	}
	if len(loss.Shape()) != 1 || loss.Shape()[0] != 1 {
		loss, err = cel.engine.Reshape(ctx, loss, []int{1})
		if err != nil {
			return nil, err
		}
	}
	return loss, nil
}

// Backward computes dL/dlogits = (softmax - onehot) / positions * dOut.
// The one-hot targets receive no gradient.
func (cel *CrossEntropyLossOneHot[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if cel.softmaxOutput == nil || cel.onehot == nil {
		return nil, fmt.Errorf("CrossEntropyLossOneHot: Backward called before Forward")
	}
	sShape := cel.softmaxOutput.Shape()
	classes := sShape[len(sShape)-1]
	positions := 1
	for _, d := range sShape[:len(sShape)-1] {
		positions *= d
	}
	_ = classes

	grad, err := cel.engine.Sub(ctx, cel.softmaxOutput, cel.onehot, nil)
	if err != nil {
		return nil, err
	}
	invN := cel.engine.Ops().FromFloat64(1.0 / float64(positions))
	grad, err = cel.engine.MulScalar(ctx, grad, invN)
	if err != nil {
		return nil, err
	}
	// Scale by upstream gradient (usually the [1] loss seed); broadcasts.
	final, err := cel.engine.Mul(ctx, grad, dOut, nil)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[T]{final, nil}, nil
}

// SoftmaxOutput returns the cached softmax output from the most recent
// Forward call.
func (cel *CrossEntropyLossOneHot[T]) SoftmaxOutput() *tensor.TensorNumeric[T] {
	return cel.softmaxOutput
}

// OpType returns the operation type of the loss node.
func (cel *CrossEntropyLossOneHot[T]) OpType() string {
	return "CrossEntropyLossOneHot"
}

// Attributes returns nil: the loss has no serializable attributes.
func (cel *CrossEntropyLossOneHot[T]) Attributes() map[string]interface{} {
	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*CrossEntropyLossOneHot[float32])(nil)
