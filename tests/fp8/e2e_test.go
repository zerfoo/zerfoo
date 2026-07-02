//go:build integration

package fp8_test

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/training/fp8"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestFP8GemmaE2E builds a tiny Gemma 3 1B-like transformer with FP8 linear
// layers, runs a forward pass, and verifies output shape and numerical validity.
func TestFP8GemmaE2E(t *testing.T) {
	// Tiny Gemma 3-like config (scaled down for CPU testing).
	const (
		batchSize      = 1
		seqLen         = 4
		dModel         = 64
		nHeads         = 2
		nKVHeads       = 2
		intermediateD  = 128
		vocabSize      = 256
		nLayers        = 2
		maxSeqLen      = 16
	)

	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	rng := rand.New(rand.NewPCG(42, 0))

	// --- Build a Gemma 3-like graph with standard layers ---
	builder := graph.NewBuilder[float32](engine)
	input := builder.Input([]int{batchSize, seqLen, dModel})

	hidden := input
	for layer := range nLayers {
		prefix := fmt.Sprintf("layer%d", layer)

		preAttnNorm, err := normalization.NewRMSNorm[float32](
			prefix+"_pre_attn_norm", engine, ops, dModel,
			normalization.WithRMSNormEpsilon[float32](1e-6),
		)
		if err != nil {
			t.Fatalf("create pre-attention norm layer %d: %v", layer, err)
		}
		normed := builder.AddNode(preAttnNorm, hidden)

		gqa, err := attention.NewGroupedQueryAttention[float32](
			engine, ops, dModel, nHeads, nKVHeads,
			attention.WithRopeBase[float32](10000.0),
			attention.WithMaxSeqLen[float32](maxSeqLen),
		)
		if err != nil {
			t.Fatalf("create GQA layer %d: %v", layer, err)
		}
		attnOut := builder.AddNode(gqa, normed)

		add1 := core.NewAdd[float32](engine)
		hidden = builder.AddNode(add1, hidden, attnOut)

		preFfnNorm, err := normalization.NewRMSNorm[float32](
			prefix+"_pre_ffn_norm", engine, ops, dModel,
			normalization.WithRMSNormEpsilon[float32](1e-6),
		)
		if err != nil {
			t.Fatalf("create pre-FFN norm layer %d: %v", layer, err)
		}
		normed2 := builder.AddNode(preFfnNorm, hidden)

		ffn, err := core.NewFFN[float32](
			prefix+"_ffn", engine, ops,
			dModel, intermediateD, dModel,
			core.WithSwiGLU[float32](),
			core.WithFFNNoBias[float32](),
		)
		if err != nil {
			t.Fatalf("create FFN layer %d: %v", layer, err)
		}
		ffnOut := builder.AddNode(ffn, normed2)

		add2 := core.NewAdd[float32](engine)
		hidden = builder.AddNode(add2, hidden, ffnOut)
	}

	finalNorm, err := normalization.NewRMSNorm[float32](
		"final_norm", engine, ops, dModel,
		normalization.WithRMSNormEpsilon[float32](1e-6),
	)
	if err != nil {
		t.Fatalf("create final norm: %v", err)
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// Reshape [batch, seq, dModel] -> [batch*seq, dModel] for the FP8 LM head.
	reshape := core.NewReshape[float32](engine, []int{batchSize * seqLen, dModel})
	flat := builder.AddNode(reshape, normedFinal)

	// FP8 LM head: projects from dModel -> vocabSize using FP8 quantized weights.
	lmHeadInitData := make([]float32, vocabSize*dModel)
	scale := float32(math.Sqrt(2.0 / float64(vocabSize*dModel)))
	for i := range lmHeadInitData {
		lmHeadInitData[i] = (rng.Float32()*2 - 1) * scale
	}
	fp8LMHead, err := fp8.NewFP8Linear[float32]("lm_head", engine, dModel, vocabSize, lmHeadInitData)
	if err != nil {
		t.Fatalf("create FP8 LM head: %v", err)
	}
	output := builder.AddNode(fp8LMHead, flat)

	modelGraph, err := builder.Build(output)
	if err != nil {
		t.Fatalf("build graph: %v", err)
	}

	// Initialize graph parameters with small random weights.
	for _, p := range modelGraph.Parameters() {
		data := p.Value.Data()
		s := float32(math.Sqrt(2.0 / float64(len(data))))
		for i := range data {
			data[i] = (rng.Float32()*2 - 1) * s
		}
	}

	// Create random input embeddings (simulating post-embedding output).
	inputData := make([]float32, batchSize*seqLen*dModel)
	for i := range inputData {
		inputData[i] = (rng.Float32()*2 - 1) * 0.1
	}
	inputTensor, err := tensor.New[float32]([]int{batchSize, seqLen, dModel}, inputData)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	// Forward pass through the graph.
	result, err := modelGraph.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("forward pass failed: %v", err)
	}

	// Verify output shape: [batch*seq, vocabSize].
	shape := result.Shape()
	wantShape := []int{batchSize * seqLen, vocabSize}
	if len(shape) != len(wantShape) {
		t.Fatalf("output shape rank = %d, want %d", len(shape), len(wantShape))
	}
	for i := range wantShape {
		if shape[i] != wantShape[i] {
			t.Fatalf("output shape[%d] = %d, want %d", i, shape[i], wantShape[i])
		}
	}
	t.Logf("output shape: %v (correct)", shape)

	// Verify no NaN or Inf in output.
	data := result.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			t.Fatalf("output[%d] is NaN", i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] is Inf", i)
		}
	}
	t.Logf("all %d output values are finite", len(data))

	// Verify FP8 quantization round-trip via the LM head layer.
	t.Run("fp8_quantize_dequantize", func(t *testing.T) {
		// Create an FP8 linear and verify forward produces valid output.
		testData := make([]float32, 4*4)
		for i := range testData {
			testData[i] = float32(rng.NormFloat64()) * 0.5
		}
		layer, err := fp8.NewFP8Linear[float32]("test_quant", engine, 4, 4, testData)
		if err != nil {
			t.Fatalf("NewFP8Linear: %v", err)
		}

		x, err := tensor.New[float32]([]int{2, 4}, []float32{
			1.0, 0.5, -0.3, 0.8,
			-1.0, 0.2, 0.6, -0.4,
		})
		if err != nil {
			t.Fatalf("create input: %v", err)
		}

		out, err := layer.Forward(ctx, x)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}

		outShape := out.Shape()
		if outShape[0] != 2 || outShape[1] != 4 {
			t.Errorf("output shape = %v, want [2, 4]", outShape)
		}

		for i, v := range out.Data() {
			if math.IsNaN(float64(v)) {
				t.Errorf("quantized output[%d] is NaN", i)
			}
			if math.IsInf(float64(v), 0) {
				t.Errorf("quantized output[%d] is Inf", i)
			}
		}
	})

	// Verify SyncFP8Weights works after modifying master weights.
	t.Run("fp8_sync_weights", func(t *testing.T) {
		initW := make([]float32, 3*3)
		for i := range initW {
			initW[i] = float32(rng.NormFloat64()) * 0.3
		}
		layer, err := fp8.NewFP8Linear[float32]("sync_test", engine, 3, 3, initW)
		if err != nil {
			t.Fatalf("NewFP8Linear: %v", err)
		}

		// Modify master weights (simulating optimizer step).
		params := layer.Parameters()
		if len(params) != 1 {
			t.Fatalf("expected 1 parameter, got %d", len(params))
		}
		masterData := params[0].Value.Data()
		for i := range masterData {
			masterData[i] *= 2.0
		}

		if err := layer.SyncFP8Weights(); err != nil {
			t.Fatalf("SyncFP8Weights: %v", err)
		}

		// Forward should still produce valid output with updated weights.
		x, err := tensor.New[float32]([]int{1, 3}, []float32{1.0, 0.5, -0.3})
		if err != nil {
			t.Fatalf("create input: %v", err)
		}
		out, err := layer.Forward(ctx, x)
		if err != nil {
			t.Fatalf("Forward after sync: %v", err)
		}
		for i, v := range out.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("output[%d] = %v after sync, expected finite", i, v)
			}
		}
	})

	// Verify MasterWeightStore end-to-end.
	t.Run("fp8_master_weight_store", func(t *testing.T) {
		layers := make([]*fp8.FP8Linear[float32], 2)
		for i := range layers {
			w := make([]float32, 4*4)
			for j := range w {
				w[j] = float32(rng.NormFloat64()) * 0.1
			}
			layers[i], err = fp8.NewFP8Linear[float32](fmt.Sprintf("store_layer_%d", i), engine, 4, 4, w)
			if err != nil {
				t.Fatalf("NewFP8Linear[%d]: %v", i, err)
			}
		}

		store, err := fp8.NewMasterWeightStore(layers)
		if err != nil {
			t.Fatalf("NewMasterWeightStore: %v", err)
		}

		fp32Params := store.FP32Params()
		if len(fp32Params) != 2 {
			t.Fatalf("FP32Params count = %d, want 2", len(fp32Params))
		}

		// Verify memory accounting.
		expectedBytes := int64(2 * 4 * 4 * 4) // 2 layers * 4*4 params * 4 bytes
		if store.MemoryBytes() != expectedBytes {
			t.Errorf("MemoryBytes = %d, want %d", store.MemoryBytes(), expectedBytes)
		}

		// SyncToFP8 should succeed.
		if err := store.SyncToFP8(); err != nil {
			t.Fatalf("SyncToFP8: %v", err)
		}
	})

	// Verify LossScaler integration.
	t.Run("fp8_loss_scaler", func(t *testing.T) {
		ls := fp8.NewLossScaler(1024)

		scaled := ls.ScaleLoss(0.5)
		if scaled != 512 {
			t.Errorf("ScaleLoss(0.5) = %f, want 512", scaled)
		}

		grads := [][]float32{{0.1, 0.2, 0.3}}
		ok := ls.CheckGradients(grads)
		if !ok {
			t.Error("CheckGradients returned false for finite gradients")
		}

		ls.UnscaleGradients(grads)
		expected := []float32{0.1 / 1024, 0.2 / 1024, 0.3 / 1024}
		for i, want := range expected {
			if math.Abs(float64(grads[0][i]-want)) > 1e-7 {
				t.Errorf("unscaled grad[%d] = %v, want %v", i, grads[0][i], want)
			}
		}

		ls.Update(false)
		if ls.Scale != 1024 {
			t.Errorf("scale after clean update = %f, want 1024", ls.Scale)
		}
	})
}
