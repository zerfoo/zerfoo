//go:build cuda

package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestGPUEngine_LinearForward verifies that a linear layer forward pass
// (MatMul of input and weights) produces the same result on GPUEngine
// and CPUEngine.
func TestGPUEngine_LinearForward(t *testing.T) {
	ops := numeric.Float32Arithmetic{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	// Simulate a linear layer: output = input @ weights
	batchSize, inputDim, outputDim := 4, 8, 3

	inputData := make([]float32, batchSize*inputDim)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}

	weightsData := make([]float32, inputDim*outputDim)
	for i := range weightsData {
		weightsData[i] = float32(i+1) * 0.01
	}

	input, _ := tensor.New[float32]([]int{batchSize, inputDim}, inputData)
	weights, _ := tensor.New[float32]([]int{inputDim, outputDim}, weightsData)

	gpuOut, err := gpuEng.MatMul(ctx, input, weights)
	if err != nil {
		t.Fatalf("GPU MatMul: %v", err)
	}

	cpuOut, err := cpuEng.MatMul(ctx, input, weights)
	if err != nil {
		t.Fatalf("CPU MatMul: %v", err)
	}

	gpuData := gpuOut.Data()
	cpuData := cpuOut.Data()

	if len(gpuData) != len(cpuData) {
		t.Fatalf("length mismatch: GPU=%d, CPU=%d", len(gpuData), len(cpuData))
	}

	for i := range gpuData {
		diff := math.Abs(float64(gpuData[i] - cpuData[i]))
		if diff > 1e-5 {
			t.Errorf("[%d] GPU=%f, CPU=%f, diff=%e", i, gpuData[i], cpuData[i], diff)
		}
	}
}

// TestGPUEngine_LinearBackward verifies that backward pass gradient
// computation (transpose + matmul) produces the same result on GPU and CPU.
func TestGPUEngine_LinearBackward(t *testing.T) {
	ops := numeric.Float32Arithmetic{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	batchSize, inputDim, outputDim := 2, 4, 3

	inputData := make([]float32, batchSize*inputDim)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}

	weightsData := make([]float32, inputDim*outputDim)
	for i := range weightsData {
		weightsData[i] = float32(i+1) * 0.05
	}

	gradOutData := make([]float32, batchSize*outputDim)
	for i := range gradOutData {
		gradOutData[i] = 1.0
	}

	input, _ := tensor.New[float32]([]int{batchSize, inputDim}, inputData)
	weights, _ := tensor.New[float32]([]int{inputDim, outputDim}, weightsData)
	gradOut, _ := tensor.New[float32]([]int{batchSize, outputDim}, gradOutData)

	// Gradient w.r.t. input = gradOut @ weights^T
	weightsT_gpu, err := gpuEng.Transpose(ctx, weights, nil)
	if err != nil {
		t.Fatalf("GPU Transpose: %v", err)
	}

	gpuGradInput, err := gpuEng.MatMul(ctx, gradOut, weightsT_gpu)
	if err != nil {
		t.Fatalf("GPU MatMul gradInput: %v", err)
	}

	weightsT_cpu, err := cpuEng.Transpose(ctx, weights, nil)
	if err != nil {
		t.Fatalf("CPU Transpose: %v", err)
	}

	cpuGradInput, err := cpuEng.MatMul(ctx, gradOut, weightsT_cpu)
	if err != nil {
		t.Fatalf("CPU MatMul gradInput: %v", err)
	}

	gpuData := gpuGradInput.Data()
	cpuData := cpuGradInput.Data()

	for i := range gpuData {
		diff := math.Abs(float64(gpuData[i] - cpuData[i]))
		if diff > 1e-5 {
			t.Errorf("gradInput[%d] GPU=%f, CPU=%f, diff=%e", i, gpuData[i], cpuData[i], diff)
		}
	}

	// Gradient w.r.t. weights = input^T @ gradOut
	inputT_gpu, err := gpuEng.Transpose(ctx, input, nil)
	if err != nil {
		t.Fatalf("GPU Transpose input: %v", err)
	}

	gpuGradWeights, err := gpuEng.MatMul(ctx, inputT_gpu, gradOut)
	if err != nil {
		t.Fatalf("GPU MatMul gradWeights: %v", err)
	}

	inputT_cpu, err := cpuEng.Transpose(ctx, input, nil)
	if err != nil {
		t.Fatalf("CPU Transpose input: %v", err)
	}

	cpuGradWeights, err := cpuEng.MatMul(ctx, inputT_cpu, gradOut)
	if err != nil {
		t.Fatalf("CPU MatMul gradWeights: %v", err)
	}

	gpuWData := gpuGradWeights.Data()
	cpuWData := cpuGradWeights.Data()

	for i := range gpuWData {
		diff := math.Abs(float64(gpuWData[i] - cpuWData[i]))
		if diff > 1e-5 {
			t.Errorf("gradWeights[%d] GPU=%f, CPU=%f, diff=%e", i, gpuWData[i], cpuWData[i], diff)
		}
	}
}

// TestGPUEngine_LinearLayerEndToEnd constructs a Linear layer with GPUEngine
// via graph.Parameter and verifies forward pass shape and data.
func TestGPUEngine_LinearLayerEndToEnd(t *testing.T) {
	ops := numeric.Float32Arithmetic{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	defer func() { _ = gpuEng.Close() }()

	ctx := context.Background()
	inputDim, outputDim := 4, 2

	// Create weights as a Parameter (like a real Linear layer would)
	weightsTensor, _ := tensor.New[float32]([]int{inputDim, outputDim}, []float32{
		0.1, 0.2,
		0.3, 0.4,
		0.5, 0.6,
		0.7, 0.8,
	})
	_, err = graph.NewParameter[float32]("test_weights", weightsTensor, tensor.New[float32])
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}

	// Forward pass: input @ weights
	input, _ := tensor.New[float32]([]int{2, inputDim}, []float32{
		1, 1, 1, 1,
		2, 2, 2, 2,
	})

	output, err := gpuEng.MatMul(ctx, input, weightsTensor)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}

	// Verify shape
	outShape := output.Shape()
	if outShape[0] != 2 || outShape[1] != 2 {
		t.Errorf("expected shape [2 2], got %v", outShape)
	}

	// Row 0: [1,1,1,1] @ [[.1,.2],[.3,.4],[.5,.6],[.7,.8]] = [1.6, 2.0]
	// Row 1: [2,2,2,2] @ same = [3.2, 4.0]
	expected := []float32{1.6, 2.0, 3.2, 4.0}
	data := output.Data()

	for i, want := range expected {
		diff := math.Abs(float64(data[i] - want))
		if diff > 1e-5 {
			t.Errorf("output[%d] = %f, want %f", i, data[i], want)
		}
	}
}
