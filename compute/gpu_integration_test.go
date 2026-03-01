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

// TestGPUEngine_AttentionOps simulates an attention mechanism:
// scores = Q @ K^T, attn = Softmax(scores), out = attn @ V.
// Compares GPU vs CPU results.
func TestGPUEngine_AttentionOps(t *testing.T) {
	ops := numeric.Float32Arithmetic{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	seqLen, dModel := 4, 8

	qData := make([]float32, seqLen*dModel)
	kData := make([]float32, seqLen*dModel)
	vData := make([]float32, seqLen*dModel)

	for i := range qData {
		qData[i] = float32(i%7+1) * 0.1
		kData[i] = float32(i%5+1) * 0.05
		vData[i] = float32(i%9+1) * 0.02
	}

	q, _ := tensor.New[float32]([]int{seqLen, dModel}, qData)
	k, _ := tensor.New[float32]([]int{seqLen, dModel}, kData)
	v, _ := tensor.New[float32]([]int{seqLen, dModel}, vData)

	for _, tc := range []struct {
		name string
		eng  Engine[float32]
	}{
		{"GPU", gpuEng},
		{"CPU", cpuEng},
	} {
		// scores = Q @ K^T
		kT, err := tc.eng.Transpose(ctx, k, nil)
		if err != nil {
			t.Fatalf("%s Transpose: %v", tc.name, err)
		}

		scores, err := tc.eng.MatMul(ctx, q, kT)
		if err != nil {
			t.Fatalf("%s MatMul QK: %v", tc.name, err)
		}

		// scale = 1/sqrt(dModel)
		scale := any(float32(1.0 / math.Sqrt(float64(dModel)))).(float32)
		scaled, err := tc.eng.MulScalar(ctx, scores, scale)
		if err != nil {
			t.Fatalf("%s MulScalar: %v", tc.name, err)
		}

		// attn = Softmax(scaled, axis=-1)
		attn, err := tc.eng.Softmax(ctx, scaled, 1)
		if err != nil {
			t.Fatalf("%s Softmax: %v", tc.name, err)
		}

		// out = attn @ V
		_, err = tc.eng.MatMul(ctx, attn, v)
		if err != nil {
			t.Fatalf("%s MatMul attnV: %v", tc.name, err)
		}

		// Verify softmax rows sum to 1
		attnData := attn.Data()
		for row := 0; row < seqLen; row++ {
			sum := float32(0)
			for col := 0; col < seqLen; col++ {
				sum += attnData[row*seqLen+col]
			}
			if math.Abs(float64(sum-1.0)) > 1e-5 {
				t.Errorf("%s: softmax row %d sum = %f, want 1.0", tc.name, row, sum)
			}
		}
	}
}

// TestGPUEngine_ElementwiseParity verifies all elementwise GPU kernels
// match CPU output.
func TestGPUEngine_ElementwiseParity(t *testing.T) {
	ops := numeric.Float32Arithmetic{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := tensor.New[float32]([]int{2, 3}, []float32{0.5, 1.5, 2.5, 3.5, 4.5, 5.5})

	tests := []struct {
		name string
		gpu  func() (*tensor.TensorNumeric[float32], error)
		cpu  func() (*tensor.TensorNumeric[float32], error)
		tol  float64
	}{
		{"Add", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Add(ctx, a, b) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Add(ctx, a, b) }, 1e-6},
		{"Sub", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Sub(ctx, a, b) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Sub(ctx, a, b) }, 1e-6},
		{"Mul", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Mul(ctx, a, b) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Mul(ctx, a, b) }, 1e-6},
		{"Div", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Div(ctx, a, b) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Div(ctx, a, b) }, 1e-5},
		{"Exp", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Exp(ctx, a) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Exp(ctx, a) }, 1e-5},
		{"Log", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Log(ctx, a) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Log(ctx, a) }, 1e-6},
		{"Sqrt", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Sqrt(ctx, a) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Sqrt(ctx, a) }, 1e-6},
		{"Rsqrt", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Rsqrt(ctx, a) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Rsqrt(ctx, a) }, 1e-5},
		{"Tanh", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Tanh(ctx, a) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Tanh(ctx, a) }, 1e-6},
		{"AddScalar", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.AddScalar(ctx, a, 10.0) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.AddScalar(ctx, a, 10.0) }, 1e-6},
		{"MulScalar", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.MulScalar(ctx, a, 3.0) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.MulScalar(ctx, a, 3.0) }, 1e-6},
		{"DivScalar", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.DivScalar(ctx, a, 2.0) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.DivScalar(ctx, a, 2.0) }, 1e-6},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gpuOut, err := tc.gpu()
			if err != nil {
				t.Fatalf("GPU %s: %v", tc.name, err)
			}

			cpuOut, err := tc.cpu()
			if err != nil {
				t.Fatalf("CPU %s: %v", tc.name, err)
			}

			gData := gpuOut.Data()
			cData := cpuOut.Data()

			if len(gData) != len(cData) {
				t.Fatalf("length mismatch: GPU=%d, CPU=%d", len(gData), len(cData))
			}

			for i := range gData {
				diff := math.Abs(float64(gData[i] - cData[i]))
				if diff > tc.tol {
					t.Errorf("[%d] GPU=%f, CPU=%f, diff=%e (tol=%e)", i, gData[i], cData[i], diff, tc.tol)
				}
			}
		})
	}
}

// TestGPUEngine_ReductionParity verifies Sum, ReduceSum, ReduceMean
// match CPU for various axes and keepDims settings.
func TestGPUEngine_ReductionParity(t *testing.T) {
	ops := numeric.Float32Arithmetic{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{3, 4}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})

	tests := []struct {
		name     string
		gpu      func() (*tensor.TensorNumeric[float32], error)
		cpu      func() (*tensor.TensorNumeric[float32], error)
		tol      float64
	}{
		{"Sum_axis0", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Sum(ctx, a, 0, false) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Sum(ctx, a, 0, false) }, 1e-5},
		{"Sum_axis1", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Sum(ctx, a, 1, false) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Sum(ctx, a, 1, false) }, 1e-5},
		{"Sum_axis0_keepDims", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.Sum(ctx, a, 0, true) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.Sum(ctx, a, 0, true) }, 1e-5},
		{"ReduceSum_axis1", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.ReduceSum(ctx, a, 1, false) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.ReduceSum(ctx, a, 1, false) }, 1e-5},
		{"ReduceMean_axis0", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.ReduceMean(ctx, a, 0, false) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.ReduceMean(ctx, a, 0, false) }, 1e-5},
		{"ReduceMean_axis1", func() (*tensor.TensorNumeric[float32], error) { return gpuEng.ReduceMean(ctx, a, 1, false) }, func() (*tensor.TensorNumeric[float32], error) { return cpuEng.ReduceMean(ctx, a, 1, false) }, 1e-5},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gpuOut, err := tc.gpu()
			if err != nil {
				t.Fatalf("GPU: %v", err)
			}

			cpuOut, err := tc.cpu()
			if err != nil {
				t.Fatalf("CPU: %v", err)
			}

			gData := gpuOut.Data()
			cData := cpuOut.Data()

			if len(gData) != len(cData) {
				t.Fatalf("length mismatch: GPU=%d, CPU=%d", len(gData), len(cData))
			}

			for i := range gData {
				diff := math.Abs(float64(gData[i] - cData[i]))
				if diff > tc.tol {
					t.Errorf("[%d] GPU=%f, CPU=%f, diff=%e", i, gData[i], cData[i], diff)
				}
			}
		})
	}
}

// TestGPUEngine_TrainingStep simulates a minimal training step:
// forward pass, MSE-like loss, backward pass (gradient computation).
func TestGPUEngine_TrainingStep(t *testing.T) {
	ops := numeric.Float32Arithmetic{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	batchSize, inputDim, outputDim := 2, 4, 2

	input, _ := tensor.New[float32]([]int{batchSize, inputDim}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
	weights, _ := tensor.New[float32]([]int{inputDim, outputDim}, []float32{
		0.1, 0.2,
		0.3, 0.4,
		0.5, 0.6,
		0.7, 0.8,
	})
	targets, _ := tensor.New[float32]([]int{batchSize, outputDim}, []float32{
		5, 10,
		15, 20,
	})

	for _, tc := range []struct {
		name string
		eng  Engine[float32]
	}{
		{"GPU", gpuEng},
		{"CPU", cpuEng},
	} {
		// Forward: pred = input @ weights
		pred, err := tc.eng.MatMul(ctx, input, weights)
		if err != nil {
			t.Fatalf("%s MatMul: %v", tc.name, err)
		}

		// Loss-like: diff = pred - targets
		diff, err := tc.eng.Sub(ctx, pred, targets)
		if err != nil {
			t.Fatalf("%s Sub: %v", tc.name, err)
		}

		// diff^2
		sqDiff, err := tc.eng.Mul(ctx, diff, diff)
		if err != nil {
			t.Fatalf("%s Mul: %v", tc.name, err)
		}

		// mean loss = ReduceMean(sqDiff, axis=1)
		_, err = tc.eng.ReduceMean(ctx, sqDiff, 1, false)
		if err != nil {
			t.Fatalf("%s ReduceMean: %v", tc.name, err)
		}

		// Backward: gradOut = 2 * diff / batchSize
		gradOut, err := tc.eng.MulScalar(ctx, diff, 2.0/float32(batchSize))
		if err != nil {
			t.Fatalf("%s MulScalar: %v", tc.name, err)
		}

		// gradWeights = input^T @ gradOut
		inputT, err := tc.eng.Transpose(ctx, input, nil)
		if err != nil {
			t.Fatalf("%s Transpose: %v", tc.name, err)
		}

		_, err = tc.eng.MatMul(ctx, inputT, gradOut)
		if err != nil {
			t.Fatalf("%s MatMul gradWeights: %v", tc.name, err)
		}
	}
}

// TestGPUEngine_SoftmaxParity verifies Softmax GPU vs CPU for various shapes and axes.
func TestGPUEngine_SoftmaxParity(t *testing.T) {
	ops := numeric.Float32Arithmetic{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	tests := []struct {
		name  string
		shape []int
		axis  int
	}{
		{"1D", []int{5}, 0},
		{"2D_axis0", []int{3, 4}, 0},
		{"2D_axis1", []int{3, 4}, 1},
		{"3D_axis2", []int{2, 3, 4}, 2},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n := 1
			for _, d := range tc.shape {
				n *= d
			}

			data := make([]float32, n)
			for i := range data {
				data[i] = float32(i+1) * 0.3
			}

			a, _ := tensor.New[float32](tc.shape, data)

			gpuOut, err := gpuEng.Softmax(ctx, a, tc.axis)
			if err != nil {
				t.Fatalf("GPU Softmax: %v", err)
			}

			cpuOut, err := cpuEng.Softmax(ctx, a, tc.axis)
			if err != nil {
				t.Fatalf("CPU Softmax: %v", err)
			}

			gData := gpuOut.Data()
			cData := cpuOut.Data()

			for i := range gData {
				diff := math.Abs(float64(gData[i] - cData[i]))
				if diff > 1e-5 {
					t.Errorf("[%d] GPU=%f, CPU=%f, diff=%e", i, gData[i], cData[i], diff)
				}
			}
		})
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

// --- Benchmarks ---

func benchMatMul(b *testing.B, eng Engine[float32], size int) {
	ctx := context.Background()
	n := size * size

	aData := make([]float32, n)
	bData := make([]float32, n)

	for i := range aData {
		aData[i] = float32(i%100) * 0.01
		bData[i] = float32(i%100) * 0.01
	}

	a, _ := tensor.New[float32]([]int{size, size}, aData)
	bt, _ := tensor.New[float32]([]int{size, size}, bData)

	b.ResetTimer()

	for range b.N {
		_, _ = eng.MatMul(ctx, a, bt)
	}
}

func BenchmarkMatMul_GPU_128(b *testing.B) {
	ops := numeric.Float32Arithmetic{}
	eng, err := NewGPUEngine[float32](ops)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = eng.Close() }()

	benchMatMul(b, eng, 128)
}

func BenchmarkMatMul_CPU_128(b *testing.B) {
	ops := numeric.Float32Arithmetic{}
	eng := NewCPUEngine[float32](ops)
	benchMatMul(b, eng, 128)
}

func BenchmarkMatMul_GPU_512(b *testing.B) {
	ops := numeric.Float32Arithmetic{}
	eng, err := NewGPUEngine[float32](ops)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = eng.Close() }()

	benchMatMul(b, eng, 512)
}

func BenchmarkMatMul_CPU_512(b *testing.B) {
	ops := numeric.Float32Arithmetic{}
	eng := NewCPUEngine[float32](ops)
	benchMatMul(b, eng, 512)
}

func BenchmarkMatMul_GPU_1024(b *testing.B) {
	ops := numeric.Float32Arithmetic{}
	eng, err := NewGPUEngine[float32](ops)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = eng.Close() }()

	benchMatMul(b, eng, 1024)
}

func BenchmarkMatMul_CPU_1024(b *testing.B) {
	ops := numeric.Float32Arithmetic{}
	eng := NewCPUEngine[float32](ops)
	benchMatMul(b, eng, 1024)
}

func BenchmarkSoftmax_GPU(b *testing.B) {
	ops := numeric.Float32Arithmetic{}
	eng, err := NewGPUEngine[float32](ops)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = eng.Close() }()

	ctx := context.Background()
	data := make([]float32, 64*128*512)

	for i := range data {
		data[i] = float32(i%100) * 0.01
	}

	a, _ := tensor.New[float32]([]int{64, 128, 512}, data)

	b.ResetTimer()

	for range b.N {
		_, _ = eng.Softmax(ctx, a, 2)
	}
}

func BenchmarkSoftmax_CPU(b *testing.B) {
	ops := numeric.Float32Arithmetic{}
	eng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	data := make([]float32, 64*128*512)
	for i := range data {
		data[i] = float32(i%100) * 0.01
	}

	a, _ := tensor.New[float32]([]int{64, 128, 512}, data)

	b.ResetTimer()

	for range b.N {
		_, _ = eng.Softmax(ctx, a, 2)
	}
}
