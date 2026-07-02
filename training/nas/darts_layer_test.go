package nas

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// identityOp is a trivial graph.Node that returns its input scaled by a fixed factor.
type identityOp[T tensor.Numeric] struct {
	scale  T
	engine compute.Engine[T]
	shape  []int
}

func (op *identityOp[T]) OpType() string                     { return "identity" }
func (op *identityOp[T]) Attributes() map[string]interface{} { return nil }
func (op *identityOp[T]) Parameters() []*graph.Parameter[T]  { return nil }
func (op *identityOp[T]) OutputShape() []int                 { return op.shape }

func (op *identityOp[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return op.engine.MulScalar(ctx, inputs[0], op.scale)
}

func (op *identityOp[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	grad, err := op.engine.MulScalar(ctx, dOut, op.scale)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[T]{grad}, nil
}

func TestDARTSLayer(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Create 3 candidate operations: scale by 1.0, 2.0, 3.0.
	candidates := []graph.Node[float32]{
		&identityOp[float32]{scale: 1.0, engine: engine, shape: []int{4}},
		&identityOp[float32]{scale: 2.0, engine: engine, shape: []int{4}},
		&identityOp[float32]{scale: 3.0, engine: engine, shape: []int{4}},
	}

	layer, err := NewDARTSLayer[float32](engine, ops, candidates)
	if err != nil {
		t.Fatalf("NewDARTSLayer: %v", err)
	}

	t.Run("implements graph.Node", func(t *testing.T) {
		var _ graph.Node[float32] = layer
	})

	t.Run("parameters", func(t *testing.T) {
		params := layer.Parameters()
		if len(params) != 1 {
			t.Fatalf("got %d parameters, want 1", len(params))
		}
		if params[0].Name != "alpha" {
			t.Errorf("parameter name = %q, want %q", params[0].Name, "alpha")
		}
		if len(params[0].Value.Data()) != 3 {
			t.Errorf("alpha length = %d, want 3", len(params[0].Value.Data()))
		}
	})

	t.Run("forward", func(t *testing.T) {
		input, err := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
		if err != nil {
			t.Fatal(err)
		}

		out, err := layer.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}

		// With uniform alpha (all zeros), softmax gives [1/3, 1/3, 1/3].
		// Output = (1/3)*1*x + (1/3)*2*x + (1/3)*3*x = (1/3)*(1+2+3)*x = 2*x
		data := out.Data()
		for i, v := range data {
			expected := float32(2.0) * float32(i+1)
			if math.Abs(float64(v-expected)) > 1e-5 {
				t.Errorf("out[%d] = %f, want %f", i, v, expected)
			}
		}
	})

	t.Run("forward with non-uniform alpha", func(t *testing.T) {
		// Set alpha so that the first op dominates.
		alphaData := layer.Parameters()[0].Value.Data()
		alphaData[0] = 10.0
		alphaData[1] = 0.0
		alphaData[2] = 0.0

		input, err := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
		if err != nil {
			t.Fatal(err)
		}

		out, err := layer.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}

		// softmax([10,0,0]) ≈ [0.9999, 0.00005, 0.00005]
		// Output ≈ 1.0 * x (since first op is scale by 1.0)
		data := out.Data()
		for i, v := range data {
			if math.Abs(float64(v-1.0)) > 0.01 {
				t.Errorf("out[%d] = %f, want ≈1.0", i, v)
			}
		}

		// Reset alpha to uniform.
		alphaData[0] = 0
	})

	t.Run("backward gradient check", func(t *testing.T) {
		// Reset alpha to zeros.
		alphaParam := layer.Parameters()[0]
		alphaParam.ClearGradient()
		alphaData := alphaParam.Value.Data()
		for i := range alphaData {
			alphaData[i] = 0
		}

		input, err := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
		if err != nil {
			t.Fatal(err)
		}

		// Forward.
		_, err = layer.Forward(ctx, input)
		if err != nil {
			t.Fatal(err)
		}

		// Create upstream gradient of ones.
		dOut, err := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
		if err != nil {
			t.Fatal(err)
		}

		grads, err := layer.Backward(ctx, types.FullBackprop, dOut, input)
		if err != nil {
			t.Fatalf("Backward: %v", err)
		}

		if len(grads) != 1 {
			t.Fatalf("got %d gradients, want 1", len(grads))
		}

		// Numerical gradient check for alpha parameters.
		eps := float32(1e-3)
		alphaGrad := alphaParam.Gradient.Data()

		for k := range alphaData {
			orig := alphaData[k]

			// f(alpha+eps)
			alphaData[k] = orig + eps
			outPlus, err := layer.Forward(ctx, input)
			if err != nil {
				t.Fatal(err)
			}

			// f(alpha-eps)
			alphaData[k] = orig - eps
			outMinus, err := layer.Forward(ctx, input)
			if err != nil {
				t.Fatal(err)
			}

			alphaData[k] = orig

			// Numerical gradient: sum of (outPlus - outMinus) / (2*eps),
			// since dOut is all ones.
			var numGrad float64
			for i := range outPlus.Data() {
				numGrad += float64(outPlus.Data()[i]-outMinus.Data()[i]) / float64(2*eps)
			}

			analytic := float64(alphaGrad[k])
			if math.Abs(analytic-numGrad) > 0.05 {
				t.Errorf("alpha[%d]: analytic grad = %f, numerical grad = %f", k, analytic, numGrad)
			}
		}

		// Numerical gradient check for input.
		inputGrad := grads[0].Data()
		inputData := input.Data()

		for k := range inputData {
			orig := inputData[k]

			inputData[k] = orig + eps
			outPlus, err := layer.Forward(ctx, input)
			if err != nil {
				t.Fatal(err)
			}

			inputData[k] = orig - eps
			outMinus, err := layer.Forward(ctx, input)
			if err != nil {
				t.Fatal(err)
			}

			inputData[k] = orig

			// Since dOut is ones, numerical gradient = sum of dOut_i * d(out_i)/d(input_k)
			// but each output element only depends on corresponding input element,
			// so it's just (outPlus[k] - outMinus[k]) / (2*eps).
			numGrad := float64(outPlus.Data()[k]-outMinus.Data()[k]) / float64(2*eps)
			analytic := float64(inputGrad[k])
			if math.Abs(analytic-numGrad) > 0.05 {
				t.Errorf("input[%d]: analytic grad = %f, numerical grad = %f", k, analytic, numGrad)
			}
		}
	})

	t.Run("op type and attributes", func(t *testing.T) {
		if layer.OpType() != "DARTSMixedOp" {
			t.Errorf("OpType() = %q, want %q", layer.OpType(), "DARTSMixedOp")
		}
		attrs := layer.Attributes()
		if attrs["num_ops"] != 3 {
			t.Errorf("num_ops = %v, want 3", attrs["num_ops"])
		}
	})

	t.Run("output shape", func(t *testing.T) {
		shape := layer.OutputShape()
		if len(shape) != 1 || shape[0] != 4 {
			t.Errorf("OutputShape() = %v, want [4]", shape)
		}
	})
}

func TestNewDARTSLayerErrors(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	t.Run("empty candidates", func(t *testing.T) {
		_, err := NewDARTSLayer[float32](engine, ops, nil)
		if err == nil {
			t.Fatal("expected error for nil candidates")
		}
	})

	t.Run("single candidate", func(t *testing.T) {
		candidates := []graph.Node[float32]{
			&identityOp[float32]{scale: 1.0, engine: engine, shape: []int{4}},
		}
		_, err := NewDARTSLayer[float32](engine, ops, candidates)
		if err == nil {
			t.Fatal("expected error for single candidate")
		}
	})
}
