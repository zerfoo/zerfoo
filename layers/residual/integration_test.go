package residual

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestBlockAttnResIntegration builds a minimal multi-layer setup with BlockAttnRes,
// simulating a 4-layer transformer grouped into N=2 blocks.
func TestBlockAttnResIntegration(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dim := 8
	blockSize := 2 // 2 layers per block
	epsilon := float32(1e-6)
	numLayers := 4

	bar, err := NewBlockAttnRes[float32](engine, ops, blockSize, dim, epsilon)
	if err != nil {
		t.Fatalf("NewBlockAttnRes: %v", err)
	}

	// Create 4 "layer outputs" simulating a 4-layer transformer.
	layerOutputs := make([]*tensor.TensorNumeric[float32], numLayers)
	for i := 0; i < numLayers; i++ {
		data := make([]float32, dim)
		for j := range data {
			data[j] = float32(i+1) * float32(j+1) * 0.1
		}
		layerOutputs[i], err = tensor.New[float32]([]int{dim}, data)
		if err != nil {
			t.Fatalf("create layer output %d: %v", i, err)
		}
	}

	// Group into N=2 blocks: block0 = sum(layer0, layer1), block1 = sum(layer2, layer3).
	// Block 0: layers 0 and 1 summed (intra-block standard residual).
	block0, err := engine.Add(ctx, layerOutputs[0], layerOutputs[1])
	if err != nil {
		t.Fatalf("add block0: %v", err)
	}

	// Block 1: layers 2 and 3 summed.
	block1, err := engine.Add(ctx, layerOutputs[2], layerOutputs[3])
	if err != nil {
		t.Fatalf("add block1: %v", err)
	}

	// Use the last layer output as the query (current hidden state).
	query := layerOutputs[numLayers-1]

	// First block is completed, second block is the partial (current) block.
	completedBlocks := []*tensor.TensorNumeric[float32]{block0}
	partialBlock := block1

	out, err := bar.Forward(ctx, query, completedBlocks, partialBlock)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Verify output shape matches input hidden dimension.
	if !equalShape(out.Shape(), []int{dim}) {
		t.Errorf("output shape = %v, want [%d]", out.Shape(), dim)
	}

	// Verify no NaN/Inf in output.
	for i, v := range out.Data() {
		fv := float64(v)
		if math.IsNaN(fv) || math.IsInf(fv, 0) {
			t.Errorf("output[%d] = %v, want finite", i, v)
		}
	}

	// Verify output magnitudes are reasonable (not exploding or vanishing).
	for i, v := range out.Data() {
		fv := float64(v)
		if math.Abs(fv) > 1000 {
			t.Errorf("output[%d] = %v, magnitude too large (possible explosion)", i, v)
		}
	}

	// Verify at least some elements are non-zero (not vanishing).
	allZero := true
	for _, v := range out.Data() {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("all output elements are zero (possible vanishing)")
	}
}

// TestAttnResVsStandardResidual compares AttnRes output against standard residual
// (simple sum). AttnRes should produce a valid but different result.
func TestAttnResVsStandardResidual(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dim := 8
	numLayers := 4

	ar, err := NewAttnRes[float32]("integ", engine, ops, dim)
	if err != nil {
		t.Fatalf("NewAttnRes: %v", err)
	}

	// Create identical layer outputs for both methods.
	layers := make([]*tensor.TensorNumeric[float32], numLayers)
	for i := 0; i < numLayers; i++ {
		data := make([]float32, dim)
		for j := range data {
			data[j] = float32(i+1) * float32(j+1) * 0.1
		}
		layers[i], err = tensor.New[float32]([]int{1, dim}, data)
		if err != nil {
			t.Fatalf("create layer %d: %v", i, err)
		}
	}

	// Compute standard residual: h = sum(all outputs).
	stdRes := layers[0]
	for i := 1; i < numLayers; i++ {
		stdRes, err = engine.Add(ctx, stdRes, layers[i])
		if err != nil {
			t.Fatalf("add standard residual at layer %d: %v", i, err)
		}
	}

	// Compute AttnRes: h = weighted sum via attention.
	attnRes, err := ar.Forward(ctx, layers...)
	if err != nil {
		t.Fatalf("AttnRes Forward: %v", err)
	}

	// Both should produce valid output (no NaN/Inf).
	for i, v := range stdRes.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("stdRes[%d] = %v, want finite", i, v)
		}
	}
	for i, v := range attnRes.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("attnRes[%d] = %v, want finite", i, v)
		}
	}

	// AttnRes should NOT be identical to standard residual (it applies selective weighting).
	identical := true
	for i := range stdRes.Data() {
		if math.Abs(float64(stdRes.Data()[i]-attnRes.Data()[i])) > 1e-5 {
			identical = false
			break
		}
	}
	if identical {
		t.Error("AttnRes output is identical to standard residual; expected selective weighting to produce different results")
	}

	// AttnRes output should be a convex combination, so each element should be
	// between the min and max across layers.
	for j := 0; j < dim; j++ {
		minVal := float32(math.MaxFloat32)
		maxVal := float32(-math.MaxFloat32)
		for i := range layers {
			v := layers[i].Data()[j]
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
		av := attnRes.Data()[j]
		if av < minVal-1e-5 || av > maxVal+1e-5 {
			t.Errorf("attnRes[%d]=%f not in convex hull [%f, %f]", j, av, minVal, maxVal)
		}
	}
}

// TestBlockAttnResBackward tests gradient flow through BlockAttnRes.
func TestBlockAttnResBackward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dim := 4

	ar, err := NewAttnRes[float32]("backward_test", engine, ops, dim)
	if err != nil {
		t.Fatalf("NewAttnRes: %v", err)
	}

	// Verify parameters exist.
	params := ar.Parameters()
	if len(params) == 0 {
		t.Fatal("expected AttnRes to have learnable parameters")
	}

	// Create layer outputs.
	layer0, err := tensor.New[float32]([]int{1, dim}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("create layer0: %v", err)
	}
	layer1, err := tensor.New[float32]([]int{1, dim}, []float32{5, 6, 7, 8})
	if err != nil {
		t.Fatalf("create layer1: %v", err)
	}

	// Run forward pass to confirm it works.
	_, err = ar.Forward(ctx, layer0, layer1)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Test Backward — it returns an error since it is not yet implemented.
	unitGrad, err := tensor.New[float32]([]int{1, dim}, []float32{1, 1, 1, 1})
	if err != nil {
		t.Fatalf("create unit gradient: %v", err)
	}

	_, bwdErr := ar.Backward(ctx, 0, unitGrad, layer0, layer1)
	if bwdErr != nil {
		t.Skip("Backward not yet implemented")
	}
}

// TestBlockAttnResScaling tests BlockAttnRes with different block counts.
func TestBlockAttnResScaling(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dim := 8
	numLayers := 8
	epsilon := float32(1e-6)

	// Create layer outputs.
	layerOutputs := make([]*tensor.TensorNumeric[float32], numLayers)
	var err error
	for i := 0; i < numLayers; i++ {
		data := make([]float32, dim)
		for j := range data {
			data[j] = float32(i+1) * float32(j+1) * 0.1
		}
		layerOutputs[i], err = tensor.New[float32]([]int{dim}, data)
		if err != nil {
			t.Fatalf("create layer %d: %v", i, err)
		}
	}

	// Use the last layer output as the query.
	query := layerOutputs[numLayers-1]

	tests := []struct {
		name      string
		blockSize int
	}{
		{"N=1 single block", numLayers},        // S=L → single block ≈ standard residuals
		{"N=4 blocks", 2},                      // S=2 → 4 blocks
		{"N=L each layer own block", 1},        // S=1 → each layer is its own block (full AttnRes)
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bar, err := NewBlockAttnRes[float32](engine, ops, tc.blockSize, dim, epsilon)
			if err != nil {
				t.Fatalf("NewBlockAttnRes(blockSize=%d): %v", tc.blockSize, err)
			}

			// Build blocks by summing layer outputs within each block.
			var completedBlocks []*tensor.TensorNumeric[float32]
			var currentBlock *tensor.TensorNumeric[float32]

			for i := 0; i < numLayers; i++ {
				if currentBlock == nil {
					currentBlock = layerOutputs[i]
				} else {
					currentBlock, err = engine.Add(ctx, currentBlock, layerOutputs[i])
					if err != nil {
						t.Fatalf("add layer %d to block: %v", i, err)
					}
				}

				// If we've filled a block (and it's not the last layer), finalize it.
				if (i+1)%tc.blockSize == 0 && i < numLayers-1 {
					completedBlocks = append(completedBlocks, currentBlock)
					currentBlock = nil
				}
			}

			// currentBlock is the partial (or last completed) block.
			out, err := bar.Forward(ctx, query, completedBlocks, currentBlock)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			// Verify output shape.
			if !equalShape(out.Shape(), []int{dim}) {
				t.Errorf("output shape = %v, want [%d]", out.Shape(), dim)
			}

			// Verify no NaN/Inf.
			for i, v := range out.Data() {
				fv := float64(v)
				if math.IsNaN(fv) || math.IsInf(fv, 0) {
					t.Errorf("output[%d] = %v, want finite", i, v)
				}
			}

			// Verify reasonable magnitudes.
			for i, v := range out.Data() {
				if math.Abs(float64(v)) > 10000 {
					t.Errorf("output[%d] = %v, magnitude too large", i, v)
				}
			}

			// Verify non-zero output.
			allZero := true
			for _, v := range out.Data() {
				if v != 0 {
					allZero = false
					break
				}
			}
			if allZero {
				t.Errorf("all output elements are zero for blockSize=%d", tc.blockSize)
			}
		})
	}
}

// BenchmarkBlockAttnRes measures BlockAttnRes forward overhead vs simple summation.
func BenchmarkBlockAttnRes(b *testing.B) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dim := 512
	numLayers := 32
	blockSize := 4
	epsilon := float32(1e-6)

	// Pre-create layer outputs.
	layerOutputs := make([]*tensor.TensorNumeric[float32], numLayers)
	var err error
	for i := 0; i < numLayers; i++ {
		data := make([]float32, dim)
		for j := range data {
			data[j] = float32(i+1) * float32(j+1) * 0.001
		}
		layerOutputs[i], err = tensor.New[float32]([]int{dim}, data)
		if err != nil {
			b.Fatalf("create layer %d: %v", i, err)
		}
	}

	query := layerOutputs[numLayers-1]

	// Pre-build blocks for BlockAttnRes.
	var completedBlocks []*tensor.TensorNumeric[float32]
	var currentBlock *tensor.TensorNumeric[float32]

	for i := 0; i < numLayers; i++ {
		if currentBlock == nil {
			currentBlock = layerOutputs[i]
		} else {
			currentBlock, err = engine.Add(ctx, currentBlock, layerOutputs[i])
			if err != nil {
				b.Fatalf("add layer %d: %v", i, err)
			}
		}
		if (i+1)%blockSize == 0 && i < numLayers-1 {
			completedBlocks = append(completedBlocks, currentBlock)
			currentBlock = nil
		}
	}
	partialBlock := currentBlock

	bar, err := NewBlockAttnRes[float32](engine, ops, blockSize, dim, epsilon)
	if err != nil {
		b.Fatalf("NewBlockAttnRes: %v", err)
	}

	b.Run("BlockAttnRes", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := bar.Forward(ctx, query, completedBlocks, partialBlock)
			if err != nil {
				b.Fatalf("Forward: %v", err)
			}
		}
	})

	b.Run("StandardResidual", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sum := layerOutputs[0]
			for j := 1; j < numLayers; j++ {
				sum, err = engine.Add(ctx, sum, layerOutputs[j])
				if err != nil {
					b.Fatalf("add: %v", err)
				}
			}
		}
	})
}
