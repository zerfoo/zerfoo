package parallel

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func newEngine() compute.Engine[float32] {
	return compute.NewCPUEngine[float32](numeric.Float32Ops{})
}

func approxEqual(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

func tensorAlmostEqual(t *testing.T, label string, got, want *tensor.TensorNumeric[float32], tol float32) {
	t.Helper()
	gotShape := got.Shape()
	wantShape := want.Shape()
	if len(gotShape) != len(wantShape) {
		t.Fatalf("%s: shape mismatch: got %v, want %v", label, gotShape, wantShape)
	}
	for i := range gotShape {
		if gotShape[i] != wantShape[i] {
			t.Fatalf("%s: shape mismatch at dim %d: got %v, want %v", label, i, gotShape, wantShape)
		}
	}
	gotData := got.Data()
	wantData := want.Data()
	for i := range gotData {
		if !approxEqual(gotData[i], wantData[i], tol) {
			t.Fatalf("%s: data mismatch at index %d: got %f, want %f", label, i, gotData[i], wantData[i])
		}
	}
}

// TestTensorParallelConfig validates configuration checks.
func TestTensorParallelConfig(t *testing.T) {
	tests := []struct {
		name    string
		config  TensorParallelConfig
		wantErr bool
	}{
		{
			name:    "valid 2 GPUs",
			config:  TensorParallelConfig{NumGPUs: 2, DeviceIDs: []int{0, 1}},
			wantErr: false,
		},
		{
			name:    "valid 4 GPUs",
			config:  TensorParallelConfig{NumGPUs: 4, DeviceIDs: []int{0, 1, 2, 3}},
			wantErr: false,
		},
		{
			name:    "zero GPUs",
			config:  TensorParallelConfig{NumGPUs: 0, DeviceIDs: []int{}},
			wantErr: true,
		},
		{
			name:    "mismatched device IDs",
			config:  TensorParallelConfig{NumGPUs: 2, DeviceIDs: []int{0}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestSplitLinearColumnWise verifies column-wise splitting of a weight matrix.
func TestSplitLinearColumnWise(t *testing.T) {
	eng := newEngine()

	// Weight: [4, 8] split into 2 shards -> each [4, 4].
	data := make([]float32, 32)
	for i := range data {
		data[i] = float32(i)
	}
	weight, err := tensor.New[float32]([]int{4, 8}, data)
	if err != nil {
		t.Fatal(err)
	}

	shards, err := SplitLinearColumnWise(eng, weight, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(shards) != 2 {
		t.Fatalf("expected 2 shards, got %d", len(shards))
	}

	for i, shard := range shards {
		shape := shard.Shard.Shape()
		if shape[0] != 4 || shape[1] != 4 {
			t.Errorf("shard %d: expected shape [4, 4], got %v", i, shape)
		}
		if shard.Rank != i {
			t.Errorf("shard %d: expected rank %d, got %d", i, i, shard.Rank)
		}
		if shard.Mode != ColumnSplit {
			t.Errorf("shard %d: expected ColumnSplit mode", i)
		}
	}

	// Verify shard contents: first shard has columns 0-3, second has columns 4-7.
	s0 := shards[0].Shard.Data()
	s1 := shards[1].Shard.Data()
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			wantLeft := float32(row*8 + col)
			wantRight := float32(row*8 + col + 4)
			if !approxEqual(s0[row*4+col], wantLeft, 1e-6) {
				t.Errorf("shard 0 [%d,%d]: got %f, want %f", row, col, s0[row*4+col], wantLeft)
			}
			if !approxEqual(s1[row*4+col], wantRight, 1e-6) {
				t.Errorf("shard 1 [%d,%d]: got %f, want %f", row, col, s1[row*4+col], wantRight)
			}
		}
	}
}

// TestSplitLinearRowWise verifies row-wise splitting of a weight matrix.
func TestSplitLinearRowWise(t *testing.T) {
	eng := newEngine()

	// Weight: [8, 4] split into 2 shards -> each [4, 4].
	data := make([]float32, 32)
	for i := range data {
		data[i] = float32(i)
	}
	weight, err := tensor.New[float32]([]int{8, 4}, data)
	if err != nil {
		t.Fatal(err)
	}

	shards, err := SplitLinearRowWise(eng, weight, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(shards) != 2 {
		t.Fatalf("expected 2 shards, got %d", len(shards))
	}

	for i, shard := range shards {
		shape := shard.Shard.Shape()
		if shape[0] != 4 || shape[1] != 4 {
			t.Errorf("shard %d: expected shape [4, 4], got %v", i, shape)
		}
		if shard.Rank != i {
			t.Errorf("shard %d: expected rank %d, got %d", i, i, shard.Rank)
		}
		if shard.Mode != RowSplit {
			t.Errorf("shard %d: expected RowSplit mode", i)
		}
	}

	// First shard: rows 0-3 of original, second: rows 4-7.
	s0 := shards[0].Shard.Data()
	s1 := shards[1].Shard.Data()
	for i := 0; i < 16; i++ {
		if !approxEqual(s0[i], float32(i), 1e-6) {
			t.Errorf("shard 0 [%d]: got %f, want %f", i, s0[i], float32(i))
		}
		if !approxEqual(s1[i], float32(i+16), 1e-6) {
			t.Errorf("shard 1 [%d]: got %f, want %f", i, s1[i], float32(i+16))
		}
	}
}

// TestSplitErrors checks that invalid inputs produce errors.
func TestSplitErrors(t *testing.T) {
	eng := newEngine()

	// 1-D tensor should fail.
	vec, err := tensor.New[float32]([]int{8}, make([]float32, 8))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := SplitLinearColumnWise(eng, vec, 2); err == nil {
		t.Error("expected error for 1-D tensor in column split")
	}
	if _, err := SplitLinearRowWise(eng, vec, 2); err == nil {
		t.Error("expected error for 1-D tensor in row split")
	}

	// Dimension not divisible.
	mat, err := tensor.New[float32]([]int{3, 4}, make([]float32, 12))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := SplitLinearRowWise(eng, mat, 2); err == nil {
		t.Error("expected error for non-divisible row split")
	}

	mat2, err := tensor.New[float32]([]int{4, 3}, make([]float32, 12))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := SplitLinearColumnWise(eng, mat2, 2); err == nil {
		t.Error("expected error for non-divisible column split")
	}
}

// TestColumnParallelLinear verifies that column-parallel output from N ranks,
// when concatenated, matches a single-GPU matmul.
func TestColumnParallelLinear(t *testing.T) {
	eng := newEngine()
	ctx := context.Background()
	numShards := 2

	// Input: [1, 2, 4] (batch=1, seqLen=2, features=4).
	input, err := tensor.New[float32]([]int{1, 2, 4}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Weight: [4, 6] -> split column-wise into [4,3] x 2.
	wData := make([]float32, 24)
	for i := range wData {
		wData[i] = float32(i+1) * 0.1
	}
	weight, err := tensor.New[float32]([]int{4, 6}, wData)
	if err != nil {
		t.Fatal(err)
	}

	// Single-GPU baseline.
	baseline, err := eng.MatMul(ctx, input, weight)
	if err != nil {
		t.Fatal(err)
	}

	// Column-parallel: split weight, compute per-rank, concatenate.
	shards, err := SplitLinearColumnWise(eng, weight, numShards)
	if err != nil {
		t.Fatal(err)
	}

	partials := make([]*tensor.TensorNumeric[float32], numShards)
	for rank := 0; rank < numShards; rank++ {
		partials[rank], err = ColumnParallelLinear(ctx, eng, input, shards[rank])
		if err != nil {
			t.Fatal(err)
		}
	}

	combined, err := eng.Concat(ctx, partials, 2)
	if err != nil {
		t.Fatal(err)
	}

	tensorAlmostEqual(t, "column-parallel vs baseline", combined, baseline, 1e-4)
}

// TestRowParallelLinear verifies that row-parallel output (with AllReduce)
// matches a single-GPU matmul.
func TestRowParallelLinear(t *testing.T) {
	eng := newEngine()
	ctx := context.Background()
	numShards := 2

	// Input: [1, 2, 4] (batch=1, seqLen=2, features=4).
	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	input, err := tensor.New[float32]([]int{1, 2, 4}, inputData)
	if err != nil {
		t.Fatal(err)
	}

	// Weight: [4, 3] -> split row-wise into [2,3] x 2.
	wData := make([]float32, 12)
	for i := range wData {
		wData[i] = float32(i+1) * 0.1
	}
	weight, err := tensor.New[float32]([]int{4, 3}, wData)
	if err != nil {
		t.Fatal(err)
	}

	// Single-GPU baseline.
	baseline, err := eng.MatMul(ctx, input, weight)
	if err != nil {
		t.Fatal(err)
	}

	// Row-parallel: split weight row-wise, split input accordingly.
	wShards, err := SplitLinearRowWise(eng, weight, numShards)
	if err != nil {
		t.Fatal(err)
	}

	// Split input along feature dimension: each rank gets [1, 2, 2].
	inputParts, err := eng.Split(ctx, input, numShards, 2)
	if err != nil {
		t.Fatal(err)
	}

	// Compute partial matmuls and sum them (simulating AllReduce).
	partials := make([]*tensor.TensorNumeric[float32], numShards)
	for rank := 0; rank < numShards; rank++ {
		partials[rank], err = eng.MatMul(ctx, inputParts[rank], wShards[rank].Shard)
		if err != nil {
			t.Fatal(err)
		}
	}

	// Sum partials = AllReduce.
	combined := partials[0]
	for i := 1; i < numShards; i++ {
		combined, err = eng.Add(ctx, combined, partials[i])
		if err != nil {
			t.Fatal(err)
		}
	}

	tensorAlmostEqual(t, "row-parallel vs baseline", combined, baseline, 1e-4)
}

// TestSumAllReducer tests the in-process AllReduce mock.
func TestSumAllReducer(t *testing.T) {
	eng := newEngine()
	ctx := context.Background()

	a, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.New[float32]([]int{2, 3}, []float32{10, 20, 30, 40, 50, 60})
	if err != nil {
		t.Fatal(err)
	}
	c, err := tensor.New[float32]([]int{2, 3}, []float32{100, 200, 300, 400, 500, 600})
	if err != nil {
		t.Fatal(err)
	}

	reducer := NewSumAllReducer[float32](eng, 3)
	reducer.AddPartial(a)
	reducer.AddPartial(b)

	result, err := reducer.AllReduceSum(ctx, c)
	if err != nil {
		t.Fatal(err)
	}

	want, err := tensor.New[float32]([]int{2, 3}, []float32{111, 222, 333, 444, 555, 666})
	if err != nil {
		t.Fatal(err)
	}
	tensorAlmostEqual(t, "SumAllReducer 3 ranks", result, want, 1e-5)
}

// TestSumAllReducerNoPartials verifies passthrough when no partials registered.
func TestSumAllReducerNoPartials(t *testing.T) {
	eng := newEngine()
	ctx := context.Background()

	a, err := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}

	reducer := NewSumAllReducer[float32](eng, 1)
	result, err := reducer.AllReduceSum(ctx, a)
	if err != nil {
		t.Fatal(err)
	}

	tensorAlmostEqual(t, "passthrough", result, a, 1e-6)
}

// TestTensorParallelWrapper tests end-to-end wrapper functionality.
func TestTensorParallelWrapper(t *testing.T) {
	eng := newEngine()
	ctx := context.Background()
	numGPUs := 2

	config := TensorParallelConfig{
		NumGPUs:   numGPUs,
		DeviceIDs: []int{0, 1},
	}
	engines := []compute.Engine[float32]{eng, eng}
	reducer := NewSumAllReducer[float32](eng, numGPUs)

	wrapper, err := NewTensorParallelWrapper(config, engines, reducer)
	if err != nil {
		t.Fatal(err)
	}

	// Create a weight [4, 8] and add as column-parallel layer.
	colWeight, err := tensor.New[float32]([]int{4, 8}, func() []float32 {
		d := make([]float32, 32)
		for i := range d {
			d[i] = float32(i) * 0.01
		}
		return d
	}())
	if err != nil {
		t.Fatal(err)
	}
	if err := wrapper.AddColumnParallelLayer(colWeight); err != nil {
		t.Fatal(err)
	}
	if wrapper.NumLayers() != 1 {
		t.Fatalf("expected 1 layer, got %d", wrapper.NumLayers())
	}

	// Forward on each rank and verify shapes.
	input, err := tensor.New[float32]([]int{1, 2, 4}, func() []float32 {
		d := make([]float32, 8)
		for i := range d {
			d[i] = float32(i + 1)
		}
		return d
	}())
	if err != nil {
		t.Fatal(err)
	}

	partials := make([]*tensor.TensorNumeric[float32], numGPUs)
	for rank := 0; rank < numGPUs; rank++ {
		partials[rank], err = wrapper.ForwardLayer(ctx, 0, rank, input)
		if err != nil {
			t.Fatal(err)
		}
		shape := partials[rank].Shape()
		// Column split: output features halved.
		if shape[2] != 4 {
			t.Errorf("rank %d: expected output feature dim 4, got %d", rank, shape[2])
		}
	}

	// Concatenate column-parallel outputs.
	combined, err := eng.Concat(ctx, partials, 2)
	if err != nil {
		t.Fatal(err)
	}

	// Compare with single-GPU baseline.
	baseline, err := eng.MatMul(ctx, input, colWeight)
	if err != nil {
		t.Fatal(err)
	}
	tensorAlmostEqual(t, "wrapper column-parallel", combined, baseline, 1e-4)
}

// TestTensorParallelWrapperRowParallel tests row-parallel through the wrapper.
func TestTensorParallelWrapperRowParallel(t *testing.T) {
	eng := newEngine()
	ctx := context.Background()
	numGPUs := 2

	// Weight: [4, 6] split row-wise -> [2, 6] per rank.
	wData := make([]float32, 24)
	for i := range wData {
		wData[i] = float32(i+1) * 0.1
	}
	weight, err := tensor.New[float32]([]int{4, 6}, wData)
	if err != nil {
		t.Fatal(err)
	}

	// Input: [1, 2, 4].
	inputData := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	input, err := tensor.New[float32]([]int{1, 2, 4}, inputData)
	if err != nil {
		t.Fatal(err)
	}

	// Single-GPU baseline.
	baseline, err := eng.MatMul(ctx, input, weight)
	if err != nil {
		t.Fatal(err)
	}

	// For row-parallel, we manually do the split-input + reduce pattern
	// since ForwardLayer with RowSplit expects already-split input and
	// the SumAllReducer needs all partials registered.
	rowShards, err := SplitLinearRowWise(eng, weight, numGPUs)
	if err != nil {
		t.Fatal(err)
	}

	inputParts, err := eng.Split(ctx, input, numGPUs, 2)
	if err != nil {
		t.Fatal(err)
	}

	partials := make([]*tensor.TensorNumeric[float32], numGPUs)
	for rank := 0; rank < numGPUs; rank++ {
		partials[rank], err = eng.MatMul(ctx, inputParts[rank], rowShards[rank].Shard)
		if err != nil {
			t.Fatal(err)
		}
	}

	// Simulate AllReduce by summing partials.
	reducer := NewSumAllReducer[float32](eng, numGPUs)
	reducer.AddPartial(partials[0])
	result, err := reducer.AllReduceSum(ctx, partials[1])
	if err != nil {
		t.Fatal(err)
	}

	tensorAlmostEqual(t, "row-parallel vs baseline", result, baseline, 1e-4)
}

// TestTensorParallelWrapperErrors tests error paths.
func TestTensorParallelWrapperErrors(t *testing.T) {
	eng := newEngine()

	t.Run("nil reducer", func(t *testing.T) {
		config := TensorParallelConfig{NumGPUs: 2, DeviceIDs: []int{0, 1}}
		_, err := NewTensorParallelWrapper[float32](config, []compute.Engine[float32]{eng, eng}, nil)
		if err == nil {
			t.Error("expected error for nil reducer")
		}
	})

	t.Run("engine count mismatch", func(t *testing.T) {
		config := TensorParallelConfig{NumGPUs: 2, DeviceIDs: []int{0, 1}}
		reducer := NewSumAllReducer[float32](eng, 2)
		_, err := NewTensorParallelWrapper[float32](config, []compute.Engine[float32]{eng}, reducer)
		if err == nil {
			t.Error("expected error for engine count mismatch")
		}
	})

	t.Run("layer index out of range", func(t *testing.T) {
		config := TensorParallelConfig{NumGPUs: 1, DeviceIDs: []int{0}}
		reducer := NewSumAllReducer[float32](eng, 1)
		wrapper, err := NewTensorParallelWrapper[float32](config, []compute.Engine[float32]{eng}, reducer)
		if err != nil {
			t.Fatal(err)
		}
		_, err = wrapper.ForwardLayer(context.Background(), 0, 0, nil)
		if err == nil {
			t.Error("expected error for layer index out of range")
		}
	})

	t.Run("rank out of range", func(t *testing.T) {
		config := TensorParallelConfig{NumGPUs: 1, DeviceIDs: []int{0}}
		reducer := NewSumAllReducer[float32](eng, 1)
		wrapper, err := NewTensorParallelWrapper[float32](config, []compute.Engine[float32]{eng}, reducer)
		if err != nil {
			t.Fatal(err)
		}
		w, wErr := tensor.New[float32]([]int{4, 4}, make([]float32, 16))
		if wErr != nil {
			t.Fatal(wErr)
		}
		if err := wrapper.AddColumnParallelLayer(w); err != nil {
			t.Fatal(err)
		}
		_, err = wrapper.ForwardLayer(context.Background(), 0, 5, nil)
		if err == nil {
			t.Error("expected error for rank out of range")
		}
	})
}

// TestTensorParallel is the main acceptance test: verifies end-to-end that
// column-parallel QKV + row-parallel output projection produces the same
// result as single-GPU execution.
func TestTensorParallel(t *testing.T) {
	eng := newEngine()
	ctx := context.Background()
	numGPUs := 2

	// Simulate a transformer attention block:
	// QKV projection: column-parallel (split output dim).
	// Output projection: row-parallel (split input dim, AllReduce).

	hiddenDim := 8
	qkvDim := 12 // 3 * numHeads * headDim for simplicity.

	// Input: [1, 4, 8] (batch=1, seqLen=4, hidden=8).
	inputData := make([]float32, 4*hiddenDim)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}
	input, err := tensor.New[float32]([]int{1, 4, hiddenDim}, inputData)
	if err != nil {
		t.Fatal(err)
	}

	// QKV weight: [8, 12].
	qkvData := make([]float32, hiddenDim*qkvDim)
	for i := range qkvData {
		qkvData[i] = float32(i%7+1) * 0.05
	}
	qkvWeight, err := tensor.New[float32]([]int{hiddenDim, qkvDim}, qkvData)
	if err != nil {
		t.Fatal(err)
	}

	// Output projection weight: [12, 8].
	outData := make([]float32, qkvDim*hiddenDim)
	for i := range outData {
		outData[i] = float32(i%5+1) * 0.03
	}
	outWeight, err := tensor.New[float32]([]int{qkvDim, hiddenDim}, outData)
	if err != nil {
		t.Fatal(err)
	}

	// --- Single-GPU baseline ---
	qkvOut, err := eng.MatMul(ctx, input, qkvWeight)
	if err != nil {
		t.Fatal(err)
	}
	baselineOut, err := eng.MatMul(ctx, qkvOut, outWeight)
	if err != nil {
		t.Fatal(err)
	}

	// --- Tensor-parallel execution ---
	// Step 1: Column-parallel QKV projection.
	qkvShards, err := SplitLinearColumnWise(eng, qkvWeight, numGPUs)
	if err != nil {
		t.Fatal(err)
	}

	qkvPartials := make([]*tensor.TensorNumeric[float32], numGPUs)
	for rank := 0; rank < numGPUs; rank++ {
		qkvPartials[rank], err = ColumnParallelLinear(ctx, eng, input, qkvShards[rank])
		if err != nil {
			t.Fatal(err)
		}
	}

	// Step 2: Row-parallel output projection on each rank's QKV slice.
	outShards, err := SplitLinearRowWise(eng, outWeight, numGPUs)
	if err != nil {
		t.Fatal(err)
	}

	outPartials := make([]*tensor.TensorNumeric[float32], numGPUs)
	for rank := 0; rank < numGPUs; rank++ {
		outPartials[rank], err = eng.MatMul(ctx, qkvPartials[rank], outShards[rank].Shard)
		if err != nil {
			t.Fatal(err)
		}
	}

	// Step 3: AllReduce (sum partials).
	tpResult := outPartials[0]
	for i := 1; i < numGPUs; i++ {
		tpResult, err = eng.Add(ctx, tpResult, outPartials[i])
		if err != nil {
			t.Fatal(err)
		}
	}

	tensorAlmostEqual(t, "tensor-parallel vs single-GPU", tpResult, baselineOut, 1e-3)
}

// TestTensorParallelFourGPUs verifies correctness with 4-way parallelism.
func TestTensorParallelFourGPUs(t *testing.T) {
	eng := newEngine()
	ctx := context.Background()
	numGPUs := 4

	hiddenDim := 16
	outDim := 8

	// Input: [1, 2, 16].
	inputData := make([]float32, 2*hiddenDim)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.05
	}
	input, err := tensor.New[float32]([]int{1, 2, hiddenDim}, inputData)
	if err != nil {
		t.Fatal(err)
	}

	// Weight: [16, 8].
	wData := make([]float32, hiddenDim*outDim)
	for i := range wData {
		wData[i] = float32(i%11+1) * 0.02
	}
	weight, err := tensor.New[float32]([]int{hiddenDim, outDim}, wData)
	if err != nil {
		t.Fatal(err)
	}

	// Single-GPU baseline.
	baseline, err := eng.MatMul(ctx, input, weight)
	if err != nil {
		t.Fatal(err)
	}

	// Column-parallel 4-way.
	shards, err := SplitLinearColumnWise(eng, weight, numGPUs)
	if err != nil {
		t.Fatal(err)
	}

	partials := make([]*tensor.TensorNumeric[float32], numGPUs)
	for rank := 0; rank < numGPUs; rank++ {
		partials[rank], err = ColumnParallelLinear(ctx, eng, input, shards[rank])
		if err != nil {
			t.Fatal(err)
		}
		shape := partials[rank].Shape()
		expectedCols := outDim / numGPUs
		if shape[2] != expectedCols {
			t.Errorf("rank %d: expected %d output cols, got %d", rank, expectedCols, shape[2])
		}
	}

	combined, err := eng.Concat(ctx, partials, 2)
	if err != nil {
		t.Fatal(err)
	}

	tensorAlmostEqual(t, "4-way column-parallel", combined, baseline, 1e-4)
}

// BenchmarkColumnParallelLinear measures column-parallel matmul throughput.
func BenchmarkColumnParallelLinear(b *testing.B) {
	eng := newEngine()
	ctx := context.Background()

	for _, numShards := range []int{2, 4, 8} {
		b.Run(fmt.Sprintf("shards=%d", numShards), func(b *testing.B) {
			inputData := make([]float32, 256*512)
			for i := range inputData {
				inputData[i] = float32(i%100) * 0.01
			}
			input, _ := tensor.New[float32]([]int{1, 256, 512}, inputData)

			wData := make([]float32, 512*512)
			for i := range wData {
				wData[i] = float32(i%50) * 0.01
			}
			weight, _ := tensor.New[float32]([]int{512, 512}, wData)

			shards, _ := SplitLinearColumnWise(eng, weight, numShards)

			b.ResetTimer()
			for b.Loop() {
				for rank := 0; rank < numShards; rank++ {
					_, _ = ColumnParallelLinear(ctx, eng, input, shards[rank])
				}
			}
		})
	}
}
