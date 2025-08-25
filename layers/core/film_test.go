package core

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestFiLM_Forward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	contextDim := 4
	featureDim := 8

	film, err := NewFiLM[float32]("test_film", engine, ops, contextDim, featureDim)
	if err != nil {
		t.Fatalf("NewFiLM returned error: %v", err)
	}

	feature, _ := tensor.New[float32]([]int{1, featureDim}, make([]float32, featureDim))
	contextTensor, _ := tensor.New[float32]([]int{1, contextDim}, make([]float32, contextDim))

	output, err := film.Forward(ctx, feature, contextTensor)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if !reflect.DeepEqual([]int{1, featureDim}, output.Shape()) {
		t.Errorf("unexpected output shape: got %v want %v", output.Shape(), []int{1, featureDim})
	}
}

func TestFiLM_Backward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	contextDim := 4
	featureDim := 8

	film, err := NewFiLM[float32]("test_film", engine, ops, contextDim, featureDim)
	if err != nil {
		t.Fatalf("NewFiLM returned error: %v", err)
	}

	featureData := make([]float32, featureDim)
	for i := range featureData {
		featureData[i] = float32(i + 1)
	}
	feature, _ := tensor.New[float32]([]int{1, featureDim}, featureData)

	contextData := make([]float32, contextDim)
	for i := range contextData {
		contextData[i] = float32(i + 1)
	}
	contextTensor, _ := tensor.New[float32]([]int{1, contextDim}, contextData)

	// Forward pass to populate intermediate values
	_, err = film.Forward(ctx, feature, contextTensor)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	// Dummy output gradient
	outputGradientData := make([]float32, featureDim)
	for i := range outputGradientData {
		outputGradientData[i] = 1.0
	}
	outputGradient, _ := tensor.New[float32]([]int{1, featureDim}, outputGradientData)

	inputGradients, err := film.Backward(ctx, types.FullBackprop, outputGradient, feature, contextTensor)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	if len(inputGradients) != 2 {
		t.Fatalf("expected 2 input gradients, got %d", len(inputGradients))
	}

	dFeature := inputGradients[0]
	dContext := inputGradients[1]

	if !reflect.DeepEqual(feature.Shape(), dFeature.Shape()) {
		t.Errorf("unexpected dFeature shape: got %v want %v", dFeature.Shape(), feature.Shape())
	}
	if !reflect.DeepEqual(contextTensor.Shape(), dContext.Shape()) {
		t.Errorf("unexpected dContext shape: got %v want %v", dContext.Shape(), contextTensor.Shape())
	}

	// Check that parameter gradients are populated
	params := film.Parameters()
	for _, p := range params {
		if p.Gradient == nil {
			t.Errorf("expected parameter gradient to be populated")
			continue
		}
		if !reflect.DeepEqual(p.Value.Shape(), p.Gradient.Shape()) {
			t.Errorf("unexpected param grad shape: got %v want %v", p.Gradient.Shape(), p.Value.Shape())
		}
	}
}

func TestFiLM_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	contextDim := 4
	featureDim := 8

	film, err := NewFiLM[float32]("test_film", engine, ops, contextDim, featureDim)
	if err != nil {
		t.Fatalf("NewFiLM returned error: %v", err)
	}

	params := film.Parameters()
	// scaleGen weights + biasGen weights
	if len(params) != 2 {
		t.Fatalf("expected 2 params, got %d", len(params))
	}
}
