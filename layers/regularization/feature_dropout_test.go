package regularization

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func newTestFeatureDropout(rate float32) (*FeatureDropout[float32], compute.Engine[float32]) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	d := NewFeatureDropout(engine, ops, rate)
	return d, engine
}

func TestFeatureDropout_InferencePassthrough(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestFeatureDropout(0.5)

	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	input, err := tensor.New([]int{2, 5}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	output, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	if output != input {
		t.Error("expected output to be the same pointer as input in eval mode")
	}

	outData := output.Data()
	for i, v := range outData {
		if v != inputData[i] {
			t.Errorf("element %d: got %v, want %v", i, v, inputData[i])
		}
	}
}

func TestFeatureDropout_TrainingDropsColumns(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestFeatureDropout(0.5)
	d.SetTraining(true)

	batch, features := 4, 10
	inputData := make([]float32, batch*features)
	for i := range inputData {
		inputData[i] = 1.0
	}

	input, err := tensor.New([]int{batch, features}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	output, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	outData := output.Data()

	// For each feature column, check that either ALL batch rows are zero or ALL are non-zero.
	for j := 0; j < features; j++ {
		allZero := true
		allNonZero := true
		for i := 0; i < batch; i++ {
			v := outData[i*features+j]
			if v == 0 {
				allNonZero = false
			} else {
				allZero = false
			}
		}
		if !allZero && !allNonZero {
			t.Errorf("feature %d: mixed zero/non-zero across batch rows (not column-wise dropout)", j)
		}
	}
}

func TestFeatureDropout_ScaleFactor(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestFeatureDropout(0.5)
	d.SetTraining(true)

	batch, features := 4, 10
	inputData := make([]float32, batch*features)
	for i := range inputData {
		inputData[i] = 1.0
	}

	input, err := tensor.New([]int{batch, features}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	output, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	outData := output.Data()
	scale := float32(1.0 / (1.0 - 0.5))

	for i, v := range outData {
		if v != 0 && v != scale {
			t.Errorf("element %d: got %v, want 0 or %v", i, v, scale)
		}
	}
}

func TestFeatureDropout_RateZero(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestFeatureDropout(0.0)
	d.SetTraining(true)

	batch, features := 4, 10
	inputData := make([]float32, batch*features)
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}

	input, err := tensor.New([]int{batch, features}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	output, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	outData := output.Data()
	for i, v := range outData {
		if v != inputData[i] {
			t.Errorf("element %d: got %v, want %v (rate=0 should preserve all values)", i, v, inputData[i])
		}
	}
}

func TestFeatureDropout_RateOne(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestFeatureDropout(1.0)
	d.SetTraining(true)

	batch, features := 4, 10
	inputData := make([]float32, batch*features)
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}

	input, err := tensor.New([]int{batch, features}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	output, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	outData := output.Data()
	for i, v := range outData {
		if v != 0 {
			t.Errorf("element %d: got %v, want 0 (rate=1.0 should drop all features)", i, v)
		}
	}
}

func TestFeatureDropout_DifferentMaskEachCall(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestFeatureDropout(0.5)
	d.SetTraining(true)

	batch, features := 4, 100
	inputData := make([]float32, batch*features)
	for i := range inputData {
		inputData[i] = 1.0
	}

	input, err := tensor.New([]int{batch, features}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	out1, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("first Forward: %v", err)
	}

	out2, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("second Forward: %v", err)
	}

	data1 := out1.Data()
	data2 := out2.Data()

	differ := false
	for i := range data1 {
		if data1[i] != data2[i] {
			differ = true
			break
		}
	}

	if !differ {
		t.Error("two Forward calls produced identical outputs; expected different masks")
	}
}

func TestFeatureDropout_ApproximateDropRate(t *testing.T) {
	ctx := context.Background()
	rate := float32(0.3)
	d, _ := newTestFeatureDropout(rate)
	d.SetTraining(true)

	batch, features := 4, 10
	trials := 1000
	totalDropped := 0
	totalFeatures := 0

	inputData := make([]float32, batch*features)
	for i := range inputData {
		inputData[i] = 1.0
	}

	input, err := tensor.New([]int{batch, features}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	for trial := 0; trial < trials; trial++ {
		output, err := d.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward trial %d: %v", trial, err)
		}

		outData := output.Data()
		for j := 0; j < features; j++ {
			// Check column j using first batch row.
			if outData[j] == 0 {
				totalDropped++
			}
			totalFeatures++
		}
	}

	observedRate := float64(totalDropped) / float64(totalFeatures)
	if math.Abs(observedRate-float64(rate)) > 0.05 {
		t.Errorf("observed drop rate %.4f, expected ~%.2f (tolerance 0.05)", observedRate, rate)
	}
}

func TestFeatureDropout_Backward(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestFeatureDropout(0.5)
	d.SetTraining(true)

	batch, features := 4, 10
	inputData := make([]float32, batch*features)
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}
	input, err := tensor.New([]int{batch, features}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	_, err = d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	dOutData := make([]float32, batch*features)
	for i := range dOutData {
		dOutData[i] = 1.0
	}
	dOut, err := tensor.New([]int{batch, features}, dOutData)
	if err != nil {
		t.Fatalf("failed to create dOut tensor: %v", err)
	}

	grads, err := d.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient tensor, got %d", len(grads))
	}

	dInputData := grads[0].Data()
	maskData := d.mask.Data()
	for i, grad := range dInputData {
		expected := dOutData[i] * maskData[i]
		if grad != expected {
			t.Errorf("element %d: grad=%v, want %v (dOut=%v * mask=%v)", i, grad, expected, dOutData[i], maskData[i])
		}
	}
}
