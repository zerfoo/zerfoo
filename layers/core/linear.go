package core

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Linear is a linear layer.
type Linear[T tensor.Numeric] struct {
	name           string
	engine         compute.Engine[T]
	ops            numeric.Arithmetic[T]
	weights        *graph.Parameter[T]
	inputFeatures  int
	outputFeatures int
}

// randomData returns size uniform-random values in [0,1) as element type T.
//
// The conversion goes through ops.FromFloat32 rather than a direct T(...)
// conversion: the reduced-precision float types (float16, bfloat16, float8)
// are defined struct types and a Go conversion from float32 to them does not
// compile, so a layer that called the old T(rand.Float32()) form could not be
// instantiated over bf16. Routing through the Arithmetic ops keeps the
// initializer generic across every numeric element type the engine supports.
//
// ops may be nil: some callers construct a layer with a nil engine/ops to
// validate configuration before wiring an engine (e.g. the Mamba nil-engine
// guard test). In that case the reduced-precision conversion is unavailable, so
// fall back to the built-in conversion for the native float kinds (the only Ts
// the pre-bf16 direct-conversion form supported) and leave others zero-valued.
func randomData[T tensor.Numeric](ops numeric.Arithmetic[T], size int) []T {
	data := make([]T, size)
	if ops == nil {
		var zero T
		for i := range data {
			switch any(zero).(type) {
			case float32:
				data[i] = any(rand.Float32()).(T)
			case float64:
				data[i] = any(float64(rand.Float32())).(T)
			default:
				// No ops to convert into a defined-type T; leave zero. A real
				// engine/ops must be supplied to initialize such a layer.
			}
		}
		return data
	}
	for i := range data {
		data[i] = ops.FromFloat32(rand.Float32())
	}
	return data
}

// NewLinear creates a new Linear layer.
func NewLinear[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputFeatures, outputFeatures int,
) (*Linear[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if inputFeatures <= 0 || outputFeatures <= 0 {
		return nil, fmt.Errorf("input and output features must be positive")
	}
	weightsTensor, err := tensor.New[T](
		[]int{inputFeatures, outputFeatures},
		randomData[T](ops, inputFeatures*outputFeatures),
	)
	if err != nil {
		return nil, err
	}
	weights, err := graph.NewParameter[T](
		name+"_weights",
		weightsTensor,
		tensor.New[T],
	)
	if err != nil {
		return nil, err
	}

	return &Linear[T]{
		name:           name,
		engine:         engine,
		ops:            ops,
		weights:        weights,
		inputFeatures:  inputFeatures,
		outputFeatures: outputFeatures,
	}, nil
}

// OpType returns the operation type of the layer.
func (l *Linear[T]) OpType() string {
	return "Linear"
}

// Attributes returns the attributes of the layer.
func (l *Linear[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"input_features":  l.inputFeatures,
		"output_features": l.outputFeatures,
	}
}

// OutputShape returns the output shape of the layer.
func (l *Linear[T]) OutputShape() []int {
	return []int{-1, l.outputFeatures} // -1 for batch dimension
}

// Forward computes the forward pass of the layer.
func (l *Linear[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Linear layer requires exactly one input, got %d", len(inputs))
	}

	return l.engine.MatMul(ctx, inputs[0], l.weights.Value)
}

// Backward computes the gradients.
func (l *Linear[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Linear layer requires exactly one input, got %d", len(inputs))
	}

	input := inputs[0]

	// Gradient with respect to weights
	inputShape := input.Shape()
	gradShape := outputGradient.Shape()

	// Reshape input and gradient to 2D for matrix operations
	batchSize := 1
	for i := 0; i < len(inputShape)-1; i++ {
		batchSize *= inputShape[i]
	}

	inputReshaped, err := l.engine.Reshape(ctx, input, []int{batchSize, inputShape[len(inputShape)-1]})
	if err != nil {
		return nil, err
	}

	gradReshaped, err := l.engine.Reshape(ctx, outputGradient, []int{batchSize, gradShape[len(gradShape)-1]})
	if err != nil {
		return nil, err
	}

	// Gradient with respect to weights: dW = X^T @ grad.
	//
	// Mirror the dx path (ADR 075 L1/L4): when the engine can multiply by a
	// transposed A directly (MatMulTransposeA), use it instead of an explicit
	// Transpose of X. This is required for the bf16 GPU path -- GPUEngine.Transpose
	// routes non-float32 types to the CPU engine, so an explicit bf16 transpose
	// would force a D2H/H2D round trip per step (breaking CUDA-graph capture and
	// tensor-core throughput). The explicit Transpose+MatMul fallback below is
	// byte-identical for engines without the path (notably the CPU engine), so the
	// CPU result is unchanged.
	var dw *tensor.TensorNumeric[T]
	if ta, ok := l.engine.(interface {
		MatMulTransposeA(context.Context, *tensor.TensorNumeric[T], *tensor.TensorNumeric[T], ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
	}); ok {
		dw, err = ta.MatMulTransposeA(ctx, inputReshaped, gradReshaped)
		if err != nil {
			return nil, err
		}
	} else {
		transposedInput, terr := l.engine.Transpose(ctx, inputReshaped, []int{1, 0})
		if terr != nil {
			return nil, terr
		}
		dw, err = l.engine.MatMul(ctx, transposedInput, gradReshaped)
		if err != nil {
			return nil, err
		}
	}
	l.weights.Gradient, err = l.engine.Add(ctx, l.weights.Gradient, dw, l.weights.Gradient)
	if err != nil {
		return nil, err
	}

	// Gradient with respect to input: dx = grad @ W^T.
	//
	// device-resident-operand path (ADR 075 L1): when the engine can multiply
	// by a transposed B directly (cuBLAS NT), compute dx = MatMulTransposeB(
	// grad, W). This reads the weight parameter -- which is device-resident
	// after UploadWeights -- in its NATURAL [in,out] layout, with NO explicit
	// Transpose allocation/kernel and NO opportunity for a transposed view to
	// drop GPUStorage and trigger a per-op host->device re-upload of the
	// weight. The explicit-transpose fallback below is byte-identical for
	// engines without the NT path (notably the CPU engine), so the CPU result
	// is unchanged.
	var dx *tensor.TensorNumeric[T]
	if tb, ok := l.engine.(interface {
		MatMulTransposeB(context.Context, *tensor.TensorNumeric[T], *tensor.TensorNumeric[T], ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
	}); ok {
		dx, err = tb.MatMulTransposeB(ctx, gradReshaped, l.weights.Value)
		if err != nil {
			return nil, err
		}
	} else {
		transposedWeights, terr := l.engine.Transpose(ctx, l.weights.Value, []int{1, 0})
		if terr != nil {
			return nil, terr
		}
		dx, err = l.engine.MatMul(ctx, gradReshaped, transposedWeights)
		if err != nil {
			return nil, err
		}
	}

	// Reshape back to original input shape
	dxReshaped, err := l.engine.Reshape(ctx, dx, inputShape)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dxReshaped}, nil
}

// Parameters returns the parameters of the layer.
func (l *Linear[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{l.weights}
}

func init() {
	model.RegisterLayer("Linear", func(engine compute.Engine[float32], ops numeric.Arithmetic[float32], name string, params map[string]*graph.Parameter[float32], attributes map[string]interface{}) (graph.Node[float32], error) {
		// Restore from ZMF parameters if available.
		if w, ok := params[name+"_weights"]; ok {
			return NewLinearFromParam(engine, w), nil
		}
		inputFeatures, ok := attributes["input_features"].(int)
		if !ok {
			return nil, fmt.Errorf("missing or invalid attribute 'input_features' for Linear")
		}
		outputFeatures, ok := attributes["output_features"].(int)
		if !ok {
			return nil, fmt.Errorf("missing or invalid attribute 'output_features' for Linear")
		}
		return NewLinear[float32](name, engine, ops, inputFeatures, outputFeatures)
	})
}

// SetName sets the name of the Linear layer.
func (l *Linear[T]) SetName(name string) {
	l.name = name
	l.weights.Name = name + "_weights"
}

// Name returns the name of the Linear layer.
func (l *Linear[T]) Name() string {
	return l.name
}

// NewLinearFromParam creates a Linear layer from an existing parameter.
// This is used for constructing layers from pre-existing parameters during model loading.
func NewLinearFromParam[T tensor.Numeric](engine compute.Engine[T], param *graph.Parameter[T]) *Linear[T] {
	shape := param.Value.Shape()
	return &Linear[T]{
		name:           param.Name,
		engine:         engine,
		ops:            engine.Ops(),
		weights:        param,
		inputFeatures:  shape[0],
		outputFeatures: shape[1],
	}
}
