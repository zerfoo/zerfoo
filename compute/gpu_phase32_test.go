//go:build cuda

package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestGPUEngine_TransposeParity verifies GPU transpose matches CPU for 2D and N-D cases.
func TestGPUEngine_TransposeParity(t *testing.T) {
	ops := numeric.Float32Ops{}
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
		axes  []int
	}{
		{"2D_nil_axes", []int{3, 4}, nil},
		{"2D_explicit", []int{3, 4}, []int{1, 0}},
		{"3D_0_2_1", []int{2, 3, 4}, []int{0, 2, 1}},
		{"3D_2_1_0", []int{2, 3, 4}, []int{2, 1, 0}},
		{"4D_0_2_1_3", []int{2, 3, 4, 5}, []int{0, 2, 1, 3}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n := 1
			for _, d := range tc.shape {
				n *= d
			}
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(i+1) * 0.1
			}

			a, _ := tensor.New[float32](tc.shape, data)

			// Upload to GPU first so GPU transpose kernel is used
			gpuA, err := tensor.ToGPU(a)
			if err != nil {
				t.Fatalf("ToGPU: %v", err)
			}

			gpuOut, err := gpuEng.Transpose(ctx, gpuA, tc.axes)
			if err != nil {
				t.Fatalf("GPU Transpose: %v", err)
			}

			cpuOut, err := cpuEng.Transpose(ctx, a, tc.axes)
			if err != nil {
				t.Fatalf("CPU Transpose: %v", err)
			}

			assertGPUStorage(t, gpuOut, "Transpose output")

			gData := gpuOut.Data()
			cData := cpuOut.Data()

			if len(gData) != len(cData) {
				t.Fatalf("length mismatch: GPU=%d, CPU=%d", len(gData), len(cData))
			}

			for i := range gData {
				if gData[i] != cData[i] {
					t.Errorf("[%d] GPU=%f, CPU=%f", i, gData[i], cData[i])
				}
			}
		})
	}
}

// TestGPUEngine_GatherParity verifies GPU gather matches CPU for 1D and 2D indices.
func TestGPUEngine_GatherParity(t *testing.T) {
	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	// Embedding table: 5 vocab x 4 dim
	tableData := make([]float32, 20)
	for i := range tableData {
		tableData[i] = float32(i+1) * 0.1
	}
	table, _ := tensor.New[float32]([]int{5, 4}, tableData)

	tests := []struct {
		name       string
		idxShape   []int
		idxData    []int
		outShape   []int
	}{
		{"1D_3tokens", []int{3}, []int{0, 2, 4}, []int{3, 4}},
		{"1D_1token", []int{1}, []int{3}, []int{1, 4}},
		{"2D_batch", []int{2, 2}, []int{0, 1, 3, 4}, []int{2, 2, 4}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			indices, _ := tensor.New[int](tc.idxShape, tc.idxData)

			gpuOut, _ := tensor.New[float32](tc.outShape, nil)
			cpuOut, _ := tensor.New[float32](tc.outShape, nil)

			// Upload table to GPU
			gpuTable, err := tensor.ToGPU(table)
			if err != nil {
				t.Fatalf("ToGPU table: %v", err)
			}

			err = gpuEng.Gather(ctx, gpuTable, indices, gpuOut)
			if err != nil {
				t.Fatalf("GPU Gather: %v", err)
			}

			err = cpuEng.Gather(ctx, table, indices, cpuOut)
			if err != nil {
				t.Fatalf("CPU Gather: %v", err)
			}

			gData := gpuOut.Data()
			cData := cpuOut.Data()

			if len(gData) != len(cData) {
				t.Fatalf("length mismatch: GPU=%d, CPU=%d", len(gData), len(cData))
			}

			for i := range gData {
				if gData[i] != cData[i] {
					t.Errorf("[%d] GPU=%f, CPU=%f", i, gData[i], cData[i])
				}
			}
		})
	}
}

// TestGPUEngine_BroadcastParity verifies GPU broadcast binary ops match CPU.
func TestGPUEngine_BroadcastParity(t *testing.T) {
	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	// Matrix [3,4] + row vector [1,4] (row broadcast)
	aData := make([]float32, 12)
	for i := range aData {
		aData[i] = float32(i+1) * 0.5
	}
	a, _ := tensor.New[float32]([]int{3, 4}, aData)
	row, _ := tensor.New[float32]([]int{1, 4}, []float32{10, 20, 30, 40})

	// Column vector [3,1] (column broadcast)
	col, _ := tensor.New[float32]([]int{3, 1}, []float32{100, 200, 300})

	tests := []struct {
		name string
		a, b *tensor.TensorNumeric[float32]
		gpuF func(context.Context, *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)
		cpuF func(context.Context, *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)
		tol  float64
	}{
		{"Add_row", a, row, gpuEng.Add, cpuEng.Add, 1e-5},
		{"Sub_row", a, row, gpuEng.Sub, cpuEng.Sub, 1e-5},
		{"Mul_row", a, row, gpuEng.Mul, cpuEng.Mul, 1e-5},
		{"Div_row", a, row, gpuEng.Div, cpuEng.Div, 1e-5},
		{"Add_col", a, col, gpuEng.Add, cpuEng.Add, 1e-5},
		{"Mul_col", a, col, gpuEng.Mul, cpuEng.Mul, 1e-5},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gpuOut, err := tc.gpuF(ctx, tc.a, tc.b)
			if err != nil {
				t.Fatalf("GPU: %v", err)
			}

			cpuOut, err := tc.cpuF(ctx, tc.a, tc.b)
			if err != nil {
				t.Fatalf("CPU: %v", err)
			}

			assertGPUStorage(t, gpuOut, tc.name+" output")

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

// TestGPUEngine_FusedRMSNormParity verifies GPU fused RMSNorm matches CPU.
func TestGPUEngine_FusedRMSNormParity(t *testing.T) {
	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	tests := []struct {
		name  string
		rows  int
		dim   int
	}{
		{"4x8", 4, 8},
		{"1x256", 1, 256},
		{"8x1152", 8, 1152},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n := tc.rows * tc.dim
			inputData := make([]float32, n)
			for i := range inputData {
				inputData[i] = float32(i%17+1) * 0.1
			}
			weightData := make([]float32, tc.dim)
			for i := range weightData {
				weightData[i] = float32(i%7+1) * 0.15
			}

			input, _ := tensor.New[float32]([]int{tc.rows, tc.dim}, inputData)
			weight, _ := tensor.New[float32]([]int{tc.dim}, weightData)
			eps := float32(1e-6)

			// CPU reference
			cpuOut, cpuScales, err := FusedRMSNorm(input, weight, eps)
			if err != nil {
				t.Fatalf("CPU FusedRMSNorm: %v", err)
			}

			// GPU path via FusedRMSNormer interface
			fused, ok := Engine[float32](gpuEng).(FusedRMSNormer)
			if !ok {
				t.Fatal("GPUEngine does not implement FusedRMSNormer")
			}

			// Upload input to GPU
			gpuInput, err := tensor.ToGPU(input)
			if err != nil {
				t.Fatalf("ToGPU input: %v", err)
			}

			gpuOut, gpuScales, err := fused.FusedRMSNormGPU(gpuInput, weight, eps)
			if err != nil {
				t.Fatalf("GPU FusedRMSNormGPU: %v", err)
			}

			gData := gpuOut.Data()
			cData := cpuOut.Data()

			if len(gData) != len(cData) {
				t.Fatalf("length mismatch: GPU=%d, CPU=%d", len(gData), len(cData))
			}

			maxDiff := float64(0)
			for i := range gData {
				diff := math.Abs(float64(gData[i] - cData[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			if maxDiff > 1e-4 {
				t.Errorf("output max diff = %e (exceeds 1e-4)", maxDiff)
			} else {
				t.Logf("output max diff = %e", maxDiff)
			}

			// Verify scales parity (needed for backward pass).
			gsData := gpuScales.Data()
			csData := cpuScales.Data()
			if len(gsData) != len(csData) {
				t.Fatalf("scales length mismatch: GPU=%d, CPU=%d", len(gsData), len(csData))
			}
			maxScaleDiff := float64(0)
			for i := range gsData {
				diff := math.Abs(float64(gsData[i] - csData[i]))
				if diff > maxScaleDiff {
					maxScaleDiff = diff
				}
			}
			if maxScaleDiff > 1e-4 {
				t.Errorf("scales max diff = %e (exceeds 1e-4)", maxScaleDiff)
			} else {
				t.Logf("scales max diff = %e", maxScaleDiff)
			}
		})
	}
}

// TestGPUEngine_UploadWeights verifies the WeightUploader interface works.
func TestGPUEngine_UploadWeights(t *testing.T) {
	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	// Create CPU-backed tensors simulating model weights
	w1, _ := tensor.New[float32]([]int{4, 8}, make([]float32, 32))
	w2, _ := tensor.New[float32]([]int{8}, make([]float32, 8))
	for i := range 32 {
		w1.Data()[i] = float32(i) * 0.1
	}
	for i := range 8 {
		w2.Data()[i] = float32(i) * 0.5
	}

	// Verify they start as CPUStorage
	if _, ok := w1.GetStorage().(*tensor.GPUStorage[float32]); ok {
		t.Fatal("w1 should start as CPUStorage")
	}

	// Upload
	var uploader WeightUploader = gpuEng
	err = uploader.UploadWeights([]*tensor.TensorNumeric[float32]{w1, w2, nil})
	if err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	// Verify they are now GPUStorage
	assertGPUStorage(t, w1, "w1 after upload")
	assertGPUStorage(t, w2, "w2 after upload")

	// Verify data is preserved
	d1 := w1.Data()
	if math.Abs(float64(d1[5]-0.5)) > 1e-6 {
		t.Errorf("w1[5] = %f, want 0.5", d1[5])
	}

	d2 := w2.Data()
	if math.Abs(float64(d2[3]-1.5)) > 1e-6 {
		t.Errorf("w2[3] = %f, want 1.5", d2[3])
	}

	// Uploading again should be a no-op (already on GPU)
	err = uploader.UploadWeights([]*tensor.TensorNumeric[float32]{w1, w2})
	if err != nil {
		t.Fatalf("second UploadWeights: %v", err)
	}
}

// TestGPUEngine_TransposeChainedWithMatMul verifies GPU transpose output
// can be directly used as MatMul input without D2H.
func TestGPUEngine_TransposeChainedWithMatMul(t *testing.T) {
	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	// Q @ K^T pattern from attention
	q, _ := tensor.New[float32]([]int{4, 8}, make([]float32, 32))
	k, _ := tensor.New[float32]([]int{4, 8}, make([]float32, 32))
	for i := range 32 {
		q.Data()[i] = float32(i%7+1) * 0.1
		k.Data()[i] = float32(i%5+1) * 0.05
	}

	// Upload to GPU
	gpuQ, _ := tensor.ToGPU(q)
	gpuK, _ := tensor.ToGPU(k)

	// GPU: transpose then matmul
	gpuKT, err := gpuEng.Transpose(ctx, gpuK, nil)
	if err != nil {
		t.Fatalf("GPU Transpose: %v", err)
	}
	assertGPUStorage(t, gpuKT, "K^T")

	gpuScores, err := gpuEng.MatMul(ctx, gpuQ, gpuKT)
	if err != nil {
		t.Fatalf("GPU MatMul: %v", err)
	}
	assertGPUStorage(t, gpuScores, "scores")

	// CPU reference
	cpuKT, _ := cpuEng.Transpose(ctx, k, nil)
	cpuScores, _ := cpuEng.MatMul(ctx, q, cpuKT)

	gData := gpuScores.Data()
	cData := cpuScores.Data()

	for i := range gData {
		diff := math.Abs(float64(gData[i] - cData[i]))
		if diff > 1e-4 {
			t.Errorf("[%d] GPU=%f, CPU=%f, diff=%e", i, gData[i], cData[i], diff)
		}
	}
}
