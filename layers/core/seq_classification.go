package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/layers/regularization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// PoolMode determines how token hidden states are aggregated into a fixed-size
// representation before classification.
type PoolMode int

const (
	// PoolCLS takes the hidden state of the first token (index 0) as the
	// sequence representation. This is the standard approach for BERT-style
	// models where a [CLS] token is prepended.
	PoolCLS PoolMode = iota

	// PoolMean computes the mean of all token hidden states across the
	// sequence dimension to produce the sequence representation.
	PoolMean
)

// SeqClassification is a sequence classification head that pools token-level
// hidden states into a fixed-size vector and projects it to class logits.
//
// Forward expects input of shape [batch, seqLen, hiddenDim] and produces
// output of shape [batch, numClasses].
type SeqClassification[T tensor.Float] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	poolMode    PoolMode
	dense       *Dense[T]
	dropout     *regularization.Dropout[T]
	numClasses  int
	outputShape []int
}

// SeqClassificationOpt is a functional option for configuring a SeqClassification layer.
type SeqClassificationOpt[T tensor.Float] func(*SeqClassification[T])

// WithDropout adds dropout before the linear projection with the given rate.
func WithDropout[T tensor.Float](engine compute.Engine[T], ops numeric.Arithmetic[T], rate T) SeqClassificationOpt[T] {
	return func(s *SeqClassification[T]) {
		s.dropout = regularization.NewDropout[T](engine, ops, rate)
	}
}

// NewSeqClassification creates a new sequence classification head.
func NewSeqClassification[T tensor.Float](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	poolMode PoolMode,
	hiddenDim, numClasses int,
	opts ...SeqClassificationOpt[T],
) (*SeqClassification[T], error) {
	if hiddenDim <= 0 || numClasses <= 0 {
		return nil, fmt.Errorf("hiddenDim and numClasses must be positive, got %d and %d", hiddenDim, numClasses)
	}

	dense, err := NewDense[T]("seq_cls", engine, ops, hiddenDim, numClasses)
	if err != nil {
		return nil, fmt.Errorf("SeqClassification: failed to create dense layer: %w", err)
	}

	s := &SeqClassification[T]{
		engine:     engine,
		ops:        ops,
		poolMode:   poolMode,
		dense:      dense,
		numClasses: numClasses,
	}

	for _, opt := range opts {
		opt(s)
	}

	return s, nil
}

// OpType returns "SeqClassification".
func (s *SeqClassification[T]) OpType() string { return "SeqClassification" }

// Attributes returns the layer attributes.
func (s *SeqClassification[T]) Attributes() map[string]interface{} {
	mode := "cls"
	if s.poolMode == PoolMean {
		mode = "mean"
	}
	attrs := map[string]interface{}{
		"pool_mode":   mode,
		"num_classes": s.numClasses,
	}
	if s.dropout != nil {
		attrs["dropout"] = s.dropout.Attributes()["rate"]
	}
	return attrs
}

// OutputShape returns the output shape from the most recent Forward call.
func (s *SeqClassification[T]) OutputShape() []int { return s.outputShape }

// Forward computes the forward pass.
// Input shape: [batch, seqLen, hiddenDim] -> Output shape: [batch, numClasses].
func (s *SeqClassification[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SeqClassification: expected 1 input, got %d", len(inputs))
	}

	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("SeqClassification: expected 3D input [batch, seqLen, hiddenDim], got shape %v", shape)
	}

	batch, seqLen, hiddenDim := shape[0], shape[1], shape[2]

	// Step 1: Pool over the sequence dimension.
	var pooled *tensor.TensorNumeric[T]
	var err error

	switch s.poolMode {
	case PoolCLS:
		// Take the first token's hidden state: x[:, 0, :]
		pooled, err = s.poolCLS(x, batch, hiddenDim)
	case PoolMean:
		// Average all token hidden states over the sequence dimension.
		pooled, err = s.poolMean(ctx, x, batch, seqLen, hiddenDim)
	default:
		return nil, fmt.Errorf("SeqClassification: unknown pool mode %d", s.poolMode)
	}
	if err != nil {
		return nil, err
	}

	// Step 2: Optional dropout.
	if s.dropout != nil {
		pooled, err = s.dropout.Forward(ctx, pooled)
		if err != nil {
			return nil, fmt.Errorf("SeqClassification: dropout failed: %w", err)
		}
	}

	// Step 3: Linear projection via Dense layer.
	output, err := s.dense.Forward(ctx, pooled)
	if err != nil {
		return nil, fmt.Errorf("SeqClassification: dense forward failed: %w", err)
	}

	s.outputShape = output.Shape()
	return output, nil
}

// poolCLS extracts the first token's hidden state from each batch element.
func (s *SeqClassification[T]) poolCLS(x *tensor.TensorNumeric[T], batch, hiddenDim int) (*tensor.TensorNumeric[T], error) {
	data := x.Data()
	shape := x.Shape()
	seqLen := shape[1]
	outData := make([]T, batch*hiddenDim)

	for b := range batch {
		srcOffset := b * seqLen * hiddenDim
		dstOffset := b * hiddenDim
		copy(outData[dstOffset:dstOffset+hiddenDim], data[srcOffset:srcOffset+hiddenDim])
	}

	return tensor.New[T]([]int{batch, hiddenDim}, outData)
}

// poolMean computes the mean of all token hidden states.
func (s *SeqClassification[T]) poolMean(ctx context.Context, x *tensor.TensorNumeric[T], batch, seqLen, hiddenDim int) (*tensor.TensorNumeric[T], error) {
	data := x.Data()
	scale := s.ops.FromFloat64(1.0 / float64(seqLen))
	outData := make([]T, batch*hiddenDim)
	zero := s.ops.FromFloat64(0)

	for b := range batch {
		for d := range hiddenDim {
			sum := zero
			for t := range seqLen {
				sum = s.ops.Add(sum, data[b*seqLen*hiddenDim+t*hiddenDim+d])
			}
			outData[b*hiddenDim+d] = s.ops.Mul(sum, scale)
		}
	}

	return tensor.New[T]([]int{batch, hiddenDim}, outData)
}

// Backward computes the backward pass.
func (s *SeqClassification[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SeqClassification: expected 1 input, got %d", len(inputs))
	}

	x := inputs[0]
	shape := x.Shape()
	batch, seqLen, hiddenDim := shape[0], shape[1], shape[2]

	// Recompute the pooled representation for Dense backward.
	var pooled *tensor.TensorNumeric[T]
	var err error

	switch s.poolMode {
	case PoolCLS:
		pooled, err = s.poolCLS(x, batch, hiddenDim)
	case PoolMean:
		pooled, err = s.poolMean(ctx, x, batch, seqLen, hiddenDim)
	}
	if err != nil {
		return nil, err
	}

	// Backward through Dense.
	denseGrads, err := s.dense.Backward(ctx, mode, dOut, pooled)
	if err != nil {
		return nil, fmt.Errorf("SeqClassification: dense backward failed: %w", err)
	}
	dPooled := denseGrads[0] // [batch, hiddenDim]

	// Backward through dropout.
	if s.dropout != nil {
		dropGrads, err := s.dropout.Backward(ctx, mode, dPooled, pooled)
		if err != nil {
			return nil, fmt.Errorf("SeqClassification: dropout backward failed: %w", err)
		}
		dPooled = dropGrads[0]
	}

	// Backward through pooling: expand gradient back to [batch, seqLen, hiddenDim].
	dPooledData := dPooled.Data()
	dInput := make([]T, batch*seqLen*hiddenDim)

	switch s.poolMode {
	case PoolCLS:
		// Gradient only flows to position 0 for each batch element.
		for b := range batch {
			srcOffset := b * hiddenDim
			dstOffset := b * seqLen * hiddenDim // position 0
			copy(dInput[dstOffset:dstOffset+hiddenDim], dPooledData[srcOffset:srcOffset+hiddenDim])
		}
	case PoolMean:
		// Gradient is distributed equally across all sequence positions.
		scale := s.ops.FromFloat64(1.0 / float64(seqLen))
		for b := range batch {
			for t := range seqLen {
				for d := range hiddenDim {
					dInput[b*seqLen*hiddenDim+t*hiddenDim+d] = s.ops.Mul(dPooledData[b*hiddenDim+d], scale)
				}
			}
		}
	}

	dInputTensor, err := tensor.New[T]([]int{batch, seqLen, hiddenDim}, dInput)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInputTensor}, nil
}

// Parameters returns the trainable parameters (from the Dense layer).
func (s *SeqClassification[T]) Parameters() []*graph.Parameter[T] {
	return s.dense.Parameters()
}

// SetTraining enables or disables training mode on the dropout sub-layer.
func (s *SeqClassification[T]) SetTraining(training bool) {
	if s.dropout != nil {
		s.dropout.SetTraining(training)
	}
}

// Statically assert that SeqClassification implements graph.Node.
var _ graph.Node[float32] = (*SeqClassification[float32])(nil)
