package hrm_test

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/model/hrm"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// ---------- NewHRM error paths ----------

func TestNewHRM_HModuleError(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	numHeads := 2

	hAttn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatal(err)
	}
	lAttn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatal(err)
	}

	// ffnDim=0 causes NewTransformerBlock (inside NewHModule) to fail
	_, err = hrm.NewHRM[float32](engine, ops, modelDim, 0, 16, 8, hAttn, lAttn)
	if err == nil {
		t.Error("expected error from HModule creation with ffnDim=0")
	}
}

func TestNewHRM_InputNetError(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	ffnDim := 16
	numHeads := 2

	hAttn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatal(err)
	}
	lAttn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatal(err)
	}

	// inputDim=0 causes NewDense("input_net") to fail
	_, err = hrm.NewHRM[float32](engine, ops, modelDim, ffnDim, 0, 8, hAttn, lAttn)
	if err == nil {
		t.Error("expected error from InputNet creation with inputDim=0")
	}
}

func TestNewHRM_OutputNetError(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	ffnDim := 16
	numHeads := 2

	hAttn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatal(err)
	}
	lAttn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatal(err)
	}

	// outputDim=0 causes NewDense("output_net") to fail
	_, err = hrm.NewHRM[float32](engine, ops, modelDim, ffnDim, 16, 0, hAttn, lAttn)
	if err == nil {
		t.Error("expected error from OutputNet creation with outputDim=0")
	}
}

// ---------- TrainStep error paths ----------

type failOptimizer[T tensor.Numeric] struct{}

func (o *failOptimizer[T]) Step(_ context.Context, _ []*graph.Parameter[T]) error {
	return errors.New("optimizer failed")
}

func TestTrainStep_OptimizerError(t *testing.T) {
	model := makeModel(t)
	lossFn := &mockLoss[float32]{}
	trainer := hrm.NewHRMTrainer[float32](model, lossFn)

	inputTensor, _ := tensor.New[float32]([]int{1, 16}, nil)
	inputs := map[graph.Node[float32]]*tensor.TensorNumeric[float32]{
		model.InputNet: inputTensor,
	}
	targets, _ := tensor.New[float32]([]int{1, 8}, nil)

	_, err := trainer.TrainStep(context.Background(), &failOptimizer[float32]{}, inputs, targets, 1, 1)
	if err == nil {
		t.Error("expected error from optimizer")
	}
}

// ---------- Forward error paths ----------

// errNode is a graph.Node that always fails in Forward
type errNode[T tensor.Numeric] struct {
	params []*graph.Parameter[T]
}

func (n *errNode[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return nil, errors.New("forward error")
}

func (n *errNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, errors.New("backward error")
}

func (n *errNode[T]) Parameters() []*graph.Parameter[T] { return n.params }
func (n *errNode[T]) OutputShape() []int                 { return []int{1} }
func (n *errNode[T]) OpType() string                     { return "ErrNode" }
func (n *errNode[T]) Attributes() map[string]any         { return nil }

func makeModel(t *testing.T) *hrm.HRM[float32] {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	hAttn, err := attention.NewGlobalAttention[float32](engine, ops, 16, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	lAttn, err := attention.NewGlobalAttention[float32](engine, ops, 16, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	model, err := hrm.NewHRM[float32](engine, ops, 16, 16, 16, 8, hAttn, lAttn)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestForward_InputNetError(t *testing.T) {
	model := makeModel(t)
	model.InputNet = &errNode[float32]{}

	input, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err := model.Forward(context.Background(), 1, 1, input)
	if err == nil {
		t.Error("expected error from InputNet.Forward")
	}
}

func TestForward_OutputNetError(t *testing.T) {
	model := makeModel(t)
	model.OutputNet = &errNode[float32]{}

	input, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err := model.Forward(context.Background(), 1, 1, input)
	if err == nil {
		t.Error("expected error from OutputNet.Forward")
	}
}

func TestTrainStep_ForwardError(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	ffnDim := 16
	numHeads := 2

	hAttn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatal(err)
	}
	lAttn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatal(err)
	}

	model, err := hrm.NewHRM[float32](engine, ops, modelDim, ffnDim, 16, 8, hAttn, lAttn)
	if err != nil {
		t.Fatal(err)
	}

	// Replace InputNet with failing node to make Forward fail
	model.InputNet = &errNode[float32]{}

	lossFn := &mockLoss[float32]{}
	trainer := hrm.NewHRMTrainer[float32](model, lossFn)

	inputTensor, _ := tensor.New[float32]([]int{1, 16}, nil)
	inputs := map[graph.Node[float32]]*tensor.TensorNumeric[float32]{
		model.InputNet: inputTensor,
	}
	targets, _ := tensor.New[float32]([]int{1, 8}, nil)

	opt := &mockOptimizer[float32]{}
	_, err = trainer.TrainStep(context.Background(), opt, inputs, targets, 1, 1)
	if err == nil {
		t.Error("expected error from Forward failure in TrainStep")
	}
}

func TestForward_LModuleError(t *testing.T) {
	model := makeModel(t)

	// Set HiddenState to a tensor with incompatible shape so LModule.Forward fails
	wrongShape, _ := tensor.New[float32]([]int{3, 5, 7}, nil)
	model.HModule.HiddenState = wrongShape

	input, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err := model.Forward(context.Background(), 1, 1, input)
	if err == nil {
		t.Error("expected error from LModule.Forward due to shape mismatch")
	}
}

func TestForward_HModuleError(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	ffnDim := 16
	numHeads := 2

	hAttn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatal(err)
	}
	lAttn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatal(err)
	}

	model, err := hrm.NewHRM[float32](engine, ops, modelDim, ffnDim, 16, 8, hAttn, lAttn)
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.New[float32]([]int{1, 16}, nil)

	// Corrupt HModule.HiddenState to an incompatible shape with LModule.HiddenState.
	// Use tSteps=0 so the LModule loop is skipped, going directly to HModule.Forward.
	// HModule.Forward does Add(lState=LModule.HiddenState [1,16], m.HiddenState [3,5,7]) -> error
	wrongShape, _ := tensor.New[float32]([]int{3, 5, 7}, nil)
	model.HModule.HiddenState = wrongShape

	_, err = model.Forward(context.Background(), 1, 0, input)
	if err == nil {
		t.Error("expected error from HModule.Forward due to shape mismatch")
	}
}

// mockOptimizer and mockLoss are reused from trainer_test.go
// They are already defined in the hrm_test package.
