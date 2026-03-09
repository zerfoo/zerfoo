package compute

import (
	"context"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TraceRecorder is the interface used by EngineProxy to record traced operations.
type TraceRecorder[T tensor.Numeric] interface {
	Record(opName string, inputs []*tensor.TensorNumeric[T], output *tensor.TensorNumeric[T], extra map[string]any)
	RecordMultiOutput(opName string, inputs []*tensor.TensorNumeric[T], outputs []*tensor.TensorNumeric[T], extra map[string]any)
	RecordGather(params *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[T], extra map[string]any)
}

// EngineProxy wraps an Engine[T] and optionally records traced operations.
type EngineProxy[T tensor.Numeric] struct {
	real   Engine[T]
	tracer TraceRecorder[T]
}

// NewEngineProxy creates a new EngineProxy wrapping the given engine.
func NewEngineProxy[T tensor.Numeric](real Engine[T]) *EngineProxy[T] {
	return &EngineProxy[T]{real: real}
}

// Real returns the underlying engine.
func (p *EngineProxy[T]) Real() Engine[T] {
	return p.real
}

// StartTracing enables tracing with the given recorder.
func (p *EngineProxy[T]) StartTracing(tracer TraceRecorder[T]) {
	p.tracer = tracer
}

// StopTracing disables tracing.
func (p *EngineProxy[T]) StopTracing() {
	p.tracer = nil
}

// record is a helper that calls tracer.Record if tracing is active.
func (p *EngineProxy[T]) record(opName string, inputs []*tensor.TensorNumeric[T], output *tensor.TensorNumeric[T], extra map[string]any) {
	if p.tracer != nil {
		p.tracer.Record(opName, inputs, output, extra)
	}
}

// --- Non-traced methods (delegate only) ---

func (p *EngineProxy[T]) Ops() numeric.Arithmetic[T] {
	return p.real.Ops()
}

func (p *EngineProxy[T]) UnaryOp(ctx context.Context, a *tensor.TensorNumeric[T], op func(T) T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.UnaryOp(ctx, a, op, dst...)
	if err == nil && p.tracer != nil {
		p.tracer.Record("UnaryOp", []*tensor.TensorNumeric[T]{a}, result, nil)
		if marker, ok := p.tracer.(interface{ MarkOpaque() }); ok {
			marker.MarkOpaque()
		}
	}
	return result, err
}

func (p *EngineProxy[T]) Zero(ctx context.Context, a *tensor.TensorNumeric[T]) error {
	return p.real.Zero(ctx, a)
}

func (p *EngineProxy[T]) Zeros(ctx context.Context, a *tensor.TensorNumeric[T], shape []int) error {
	return p.real.Zeros(ctx, a, shape)
}

func (p *EngineProxy[T]) Copy(ctx context.Context, dst, src *tensor.TensorNumeric[T]) error {
	return p.real.Copy(ctx, dst, src)
}

func (p *EngineProxy[T]) Fill(ctx context.Context, t *tensor.TensorNumeric[T], value T) error {
	return p.real.Fill(ctx, t, value)
}

func (p *EngineProxy[T]) RandomUniform(ctx context.Context, t *tensor.TensorNumeric[T], minVal, maxVal T) error {
	return p.real.RandomUniform(ctx, t, minVal, maxVal)
}

func (p *EngineProxy[T]) Gather(ctx context.Context, params *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[T]) error {
	err := p.real.Gather(ctx, params, indices, output)
	if err == nil && p.tracer != nil {
		p.tracer.RecordGather(params, indices, output, nil)
	}
	return err
}

func (p *EngineProxy[T]) ScatterAdd(ctx context.Context, dEmbeddingTable *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], dOut *tensor.TensorNumeric[T]) error {
	return p.real.ScatterAdd(ctx, dEmbeddingTable, indices, dOut)
}

func (p *EngineProxy[T]) OneHot(ctx context.Context, input *tensor.TensorNumeric[int], depth int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return p.real.OneHot(ctx, input, depth, dst...)
}

func (p *EngineProxy[T]) TanhPrime(ctx context.Context, a, upstream *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return p.real.TanhPrime(ctx, a, upstream, dst...)
}

// --- Traced methods ---

func (p *EngineProxy[T]) Add(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Add(ctx, a, b, dst...)
	if err == nil {
		p.record("Add", []*tensor.TensorNumeric[T]{a, b}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) Sub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Sub(ctx, a, b, dst...)
	if err == nil {
		p.record("Sub", []*tensor.TensorNumeric[T]{a, b}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) Mul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Mul(ctx, a, b, dst...)
	if err == nil {
		p.record("Mul", []*tensor.TensorNumeric[T]{a, b}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) Div(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Div(ctx, a, b, dst...)
	if err == nil {
		p.record("Div", []*tensor.TensorNumeric[T]{a, b}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) Pow(ctx context.Context, base, exponent *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Pow(ctx, base, exponent, dst...)
	if err == nil {
		p.record("Pow", []*tensor.TensorNumeric[T]{base, exponent}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) MatMul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.MatMul(ctx, a, b, dst...)
	if err == nil {
		p.record("MatMul", []*tensor.TensorNumeric[T]{a, b}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) Exp(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Exp(ctx, a, dst...)
	if err == nil {
		p.record("Exp", []*tensor.TensorNumeric[T]{a}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) Log(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Log(ctx, a, dst...)
	if err == nil {
		p.record("Log", []*tensor.TensorNumeric[T]{a}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) Tanh(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Tanh(ctx, a, dst...)
	if err == nil {
		p.record("Tanh", []*tensor.TensorNumeric[T]{a}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) Sqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Sqrt(ctx, a, dst...)
	if err == nil {
		p.record("Sqrt", []*tensor.TensorNumeric[T]{a}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) Rsqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Rsqrt(ctx, a, dst...)
	if err == nil {
		p.record("Rsqrt", []*tensor.TensorNumeric[T]{a}, result, nil)
	}
	return result, err
}

func (p *EngineProxy[T]) MulScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.MulScalar(ctx, a, scalar, dst...)
	if err == nil {
		p.record("MulScalar", []*tensor.TensorNumeric[T]{a}, result, map[string]any{"scalar": scalar})
	}
	return result, err
}

func (p *EngineProxy[T]) AddScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.AddScalar(ctx, a, scalar, dst...)
	if err == nil {
		p.record("AddScalar", []*tensor.TensorNumeric[T]{a}, result, map[string]any{"scalar": scalar})
	}
	return result, err
}

func (p *EngineProxy[T]) DivScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.DivScalar(ctx, a, scalar, dst...)
	if err == nil {
		p.record("DivScalar", []*tensor.TensorNumeric[T]{a}, result, map[string]any{"scalar": scalar})
	}
	return result, err
}

func (p *EngineProxy[T]) Softmax(ctx context.Context, a *tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Softmax(ctx, a, axis, dst...)
	if err == nil {
		p.record("Softmax", []*tensor.TensorNumeric[T]{a}, result, map[string]any{"axis": axis})
	}
	return result, err
}

func (p *EngineProxy[T]) ReduceSum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.ReduceSum(ctx, a, axis, keepDims, dst...)
	if err == nil {
		p.record("ReduceSum", []*tensor.TensorNumeric[T]{a}, result, map[string]any{"axis": axis, "keepDims": keepDims})
	}
	return result, err
}

func (p *EngineProxy[T]) ReduceMean(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.ReduceMean(ctx, a, axis, keepDims, dst...)
	if err == nil {
		p.record("ReduceMean", []*tensor.TensorNumeric[T]{a}, result, map[string]any{"axis": axis, "keepDims": keepDims})
	}
	return result, err
}

func (p *EngineProxy[T]) Reshape(ctx context.Context, a *tensor.TensorNumeric[T], shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Reshape(ctx, a, shape, dst...)
	if err == nil {
		p.record("Reshape", []*tensor.TensorNumeric[T]{a}, result, map[string]any{"shape": shape})
	}
	return result, err
}

func (p *EngineProxy[T]) Transpose(ctx context.Context, a *tensor.TensorNumeric[T], axes []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Transpose(ctx, a, axes, dst...)
	if err == nil {
		p.record("Transpose", []*tensor.TensorNumeric[T]{a}, result, map[string]any{"axes": axes})
	}
	return result, err
}

func (p *EngineProxy[T]) Concat(ctx context.Context, tensors []*tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Concat(ctx, tensors, axis, dst...)
	if err == nil {
		p.record("Concat", tensors, result, map[string]any{"axis": axis})
	}
	return result, err
}

func (p *EngineProxy[T]) Split(ctx context.Context, a *tensor.TensorNumeric[T], numSplits int, axis int) ([]*tensor.TensorNumeric[T], error) {
	results, err := p.real.Split(ctx, a, numSplits, axis)
	if err == nil && p.tracer != nil {
		p.tracer.RecordMultiOutput("Split", []*tensor.TensorNumeric[T]{a}, results, map[string]any{
			"numSplits": numSplits,
			"axis":      axis,
		})
	}
	return results, err
}

func (p *EngineProxy[T]) Repeat(ctx context.Context, a *tensor.TensorNumeric[T], axis int, repetitions int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Repeat(ctx, a, axis, repetitions, dst...)
	if err == nil {
		p.record("Repeat", []*tensor.TensorNumeric[T]{a}, result, map[string]any{"axis": axis, "repetitions": repetitions})
	}
	return result, err
}

func (p *EngineProxy[T]) Sum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	result, err := p.real.Sum(ctx, a, axis, keepDims, dst...)
	if err == nil {
		p.record("Sum", []*tensor.TensorNumeric[T]{a}, result, map[string]any{"axis": axis, "keepDims": keepDims})
	}
	return result, err
}
