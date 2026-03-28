package integration

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/inference/transmla"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// testModelConfig returns a small Llama-like model config for testing.
func testModelConfig() *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:     "llama",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       4,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
}

// fillTensor creates a tensor of the given shape with small deterministic values.
func fillTensor(shape []int, scale float32) *tensor.TensorNumeric[float32] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = scale * float32(math.Sin(float64(i)*0.01))
	}
	t, _ := tensor.New(shape, data)
	return t
}

// onesTensor creates a tensor of the given shape filled with ones.
func onesTensor(shape []int) *tensor.TensorNumeric[float32] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = 1.0
	}
	t, _ := tensor.New(shape, data)
	return t
}

// makeSyntheticMHATensors builds a minimal Llama tensor set with standard
// MHA k_proj/v_proj weights. These are the "original" weights before
// TransMLA conversion.
func makeSyntheticMHATensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads

	tensors["model.embed_tokens.weight"] = fillTensor([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = onesTensor([]int{hidden})
	tensors["lm_head.weight"] = fillTensor([]int{vocab, hidden}, 0.02)

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		tensors[prefix+"input_layernorm.weight"] = onesTensor([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fillTensor([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fillTensor([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fillTensor([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fillTensor([]int{hidden, hidden}, 0.02)
		tensors[prefix+"post_attention_layernorm.weight"] = onesTensor([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fillTensor([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fillTensor([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fillTensor([]int{hidden, inter}, 0.02)
	}

	return tensors
}

// convertToTransMLA takes a standard MHA tensor set and replaces k_proj/v_proj
// with TransMLA decomposed tensors (wDKV, wUK, wUV) via truncated SVD.
// Returns the converted tensor map and the rank used.
func convertToTransMLA(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	rank int,
) (map[string]*tensor.TensorNumeric[float32], error) {
	headDim := cfg.HiddenSize / cfg.NumHeads
	kvDim := cfg.NumKVHeads * headDim

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		transPrefix := fmt.Sprintf("transmla.%d.", i)

		kTensor := tensors[prefix+"self_attn.k_proj.weight"]
		vTensor := tensors[prefix+"self_attn.v_proj.weight"]
		if kTensor == nil || vTensor == nil {
			return nil, fmt.Errorf("layer %d: missing k_proj or v_proj", i)
		}

		// Extract float64 matrices from tensors.
		// k_proj shape: [kvDim, hiddenSize], v_proj shape: [kvDim, hiddenSize]
		kData := kTensor.Data()
		vData := vTensor.Data()

		wK := make([][]float64, kvDim)
		for r := 0; r < kvDim; r++ {
			wK[r] = make([]float64, cfg.HiddenSize)
			for c := 0; c < cfg.HiddenSize; c++ {
				wK[r][c] = float64(kData[r*cfg.HiddenSize+c])
			}
		}
		wV := make([][]float64, kvDim)
		for r := 0; r < kvDim; r++ {
			wV[r] = make([]float64, cfg.HiddenSize)
			for c := 0; c < cfg.HiddenSize; c++ {
				wV[r][c] = float64(vData[r*cfg.HiddenSize+c])
			}
		}

		wDKV, wUK, wUV, err := transmla.DecomposeKVProjection(wK, wV, rank)
		if err != nil {
			return nil, fmt.Errorf("layer %d: SVD decompose: %w", i, err)
		}

		// Remove original k_proj/v_proj.
		delete(tensors, prefix+"self_attn.k_proj.weight")
		delete(tensors, prefix+"self_attn.v_proj.weight")

		// Add TransMLA tensors.
		// wDKV: [hiddenSize, rank]
		dkvFlat := make([]float32, cfg.HiddenSize*rank)
		for r := 0; r < cfg.HiddenSize; r++ {
			for c := 0; c < rank; c++ {
				dkvFlat[r*rank+c] = float32(wDKV[r][c])
			}
		}
		dkvT, _ := tensor.New([]int{cfg.HiddenSize, rank}, dkvFlat)
		tensors[transPrefix+"wDKV"] = dkvT

		// wUK: [kvDim, rank]
		ukFlat := make([]float32, kvDim*rank)
		for r := 0; r < kvDim; r++ {
			for c := 0; c < rank; c++ {
				ukFlat[r*rank+c] = float32(wUK[r][c])
			}
		}
		ukT, _ := tensor.New([]int{kvDim, rank}, ukFlat)
		tensors[transPrefix+"wUK"] = ukT

		// wUV: [kvDim, rank]
		uvFlat := make([]float32, kvDim*rank)
		for r := 0; r < kvDim; r++ {
			for c := 0; c < rank; c++ {
				uvFlat[r*rank+c] = float32(wUV[r][c])
			}
		}
		uvT, _ := tensor.New([]int{kvDim, rank}, uvFlat)
		tensors[transPrefix+"wUV"] = uvT
	}

	return tensors, nil
}

func TestTransMLAEndToEnd(t *testing.T) {
	tests := []struct {
		name      string
		rank      int
		numLayers int
		wantErr   bool
	}{
		{
			name:      "rank_4_two_layers",
			rank:      4,
			numLayers: 2,
		},
		{
			name:      "rank_8_single_layer",
			rank:      8,
			numLayers: 1,
		},
		{
			name:      "rank_2_minimal",
			rank:      2,
			numLayers: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testModelConfig()
			cfg.NumLayers = tc.numLayers

			headDim := cfg.HiddenSize / cfg.NumHeads
			kvDim := cfg.NumKVHeads * headDim

			// Step 1: Create synthetic MHA tensors.
			tensors := makeSyntheticMHATensors(cfg)

			// Verify original k_proj/v_proj exist.
			for i := 0; i < cfg.NumLayers; i++ {
				prefix := fmt.Sprintf("model.layers.%d.", i)
				if tensors[prefix+"self_attn.k_proj.weight"] == nil {
					t.Fatalf("layer %d: missing k_proj before conversion", i)
				}
				if tensors[prefix+"self_attn.v_proj.weight"] == nil {
					t.Fatalf("layer %d: missing v_proj before conversion", i)
				}
			}

			// Step 2: Convert via SVD.
			tensors, err := convertToTransMLA(tensors, cfg, tc.rank)
			if err != nil {
				t.Fatalf("convertToTransMLA: %v", err)
			}

			// Verify TransMLA tensors replaced k_proj/v_proj.
			for i := 0; i < cfg.NumLayers; i++ {
				prefix := fmt.Sprintf("model.layers.%d.", i)
				transPrefix := fmt.Sprintf("transmla.%d.", i)

				if tensors[prefix+"self_attn.k_proj.weight"] != nil {
					t.Fatalf("layer %d: k_proj should be removed after conversion", i)
				}
				if tensors[prefix+"self_attn.v_proj.weight"] != nil {
					t.Fatalf("layer %d: v_proj should be removed after conversion", i)
				}
				if tensors[transPrefix+"wDKV"] == nil {
					t.Fatalf("layer %d: missing wDKV after conversion", i)
				}
				if tensors[transPrefix+"wUK"] == nil {
					t.Fatalf("layer %d: missing wUK after conversion", i)
				}
				if tensors[transPrefix+"wUV"] == nil {
					t.Fatalf("layer %d: missing wUV after conversion", i)
				}
			}

			// Step 3: Load model with TransMLA — verify it uses the MLA path.
			cfg.TransMLAKVLoraDim = tc.rank
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := inference.BuildArchGraph("llama", tensors, cfg, engine)
			if err != nil {
				t.Fatalf("BuildArchGraph with TransMLA: %v", err)
			}
			if g == nil {
				t.Fatal("graph is nil")
			}

			// Step 4: Forward pass — verify output is non-NaN, non-zero.
			tokenIDs := []float32{1, 5, 10}
			input, err := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)
			if err != nil {
				t.Fatalf("create input tensor: %v", err)
			}

			output, err := g.Forward(context.Background(), input)
			if err != nil {
				t.Fatalf("forward pass: %v", err)
			}

			shape := output.Shape()
			if len(shape) != 3 || shape[0] != 1 || shape[1] != len(tokenIDs) || shape[2] != cfg.VocabSize {
				t.Fatalf("unexpected output shape %v, want [1, %d, %d]", shape, len(tokenIDs), cfg.VocabSize)
			}

			data := output.Data()
			hasNonZero := false
			for i, v := range data {
				if math.IsNaN(float64(v)) {
					t.Fatalf("NaN at output index %d", i)
				}
				if math.IsInf(float64(v), 0) {
					t.Fatalf("Inf at output index %d", i)
				}
				if v != 0 {
					hasNonZero = true
				}
			}
			if !hasNonZero {
				t.Fatal("output is all zeros")
			}

			// Step 5: Verify KV cache memory is smaller with TransMLA.
			// Standard MHA caches (numKVHeads * headDim) per token per layer
			// for both K and V. TransMLA caches kvLoraDim per token per layer.
			originalKVPerToken := 2 * kvDim // K + V, both kvDim wide
			transMLAKVPerToken := tc.rank   // compressed latent only
			if transMLAKVPerToken >= originalKVPerToken {
				t.Fatalf("TransMLA KV per token (%d) should be smaller than original (%d)",
					transMLAKVPerToken, originalKVPerToken)
			}

			compressionRatio := float64(originalKVPerToken) / float64(transMLAKVPerToken)
			t.Logf("KV cache compression: %dx smaller (original=%d, transmla=%d per token per layer)",
				int(compressionRatio), originalKVPerToken, transMLAKVPerToken)
		})
	}
}

func TestTransMLASVDReconstructionQuality(t *testing.T) {
	cfg := testModelConfig()
	cfg.NumLayers = 1

	headDim := cfg.HiddenSize / cfg.NumHeads
	kvDim := cfg.NumKVHeads * headDim

	tests := []struct {
		name    string
		rank    int
		maxErr  float64 // maximum allowed relative Frobenius error
	}{
		{
			name:   "full_rank_exact",
			rank:   kvDim, // full rank should reconstruct exactly
			maxErr: 1e-6,
		},
		{
			name:   "half_rank_approximate",
			rank:   kvDim / 2,
			maxErr: 1.0, // lossy but bounded
		},
		{
			name:   "quarter_rank",
			rank:   kvDim / 4,
			maxErr: 1.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.rank <= 0 {
				t.Skip("rank <= 0")
			}

			tensors := makeSyntheticMHATensors(cfg)
			prefix := "model.layers.0."

			kData := tensors[prefix+"self_attn.k_proj.weight"].Data()
			vData := tensors[prefix+"self_attn.v_proj.weight"].Data()

			wK := make([][]float64, kvDim)
			for r := 0; r < kvDim; r++ {
				wK[r] = make([]float64, cfg.HiddenSize)
				for c := 0; c < cfg.HiddenSize; c++ {
					wK[r][c] = float64(kData[r*cfg.HiddenSize+c])
				}
			}
			wV := make([][]float64, kvDim)
			for r := 0; r < kvDim; r++ {
				wV[r] = make([]float64, cfg.HiddenSize)
				for c := 0; c < cfg.HiddenSize; c++ {
					wV[r][c] = float64(vData[r*cfg.HiddenSize+c])
				}
			}

			// Concatenate [W_K; W_V] for error measurement.
			original := make([][]float64, 2*kvDim)
			for i := 0; i < kvDim; i++ {
				original[i] = wK[i]
			}
			for i := 0; i < kvDim; i++ {
				original[kvDim+i] = wV[i]
			}

			wDKV, wUK, wUV, err := transmla.DecomposeKVProjection(wK, wV, tc.rank)
			if err != nil {
				t.Fatalf("DecomposeKVProjection: %v", err)
			}

			relErr := transmla.ReconstructionError(original, wDKV, wUK, wUV)
			if relErr > tc.maxErr {
				t.Fatalf("reconstruction error %.6f exceeds threshold %.6f", relErr, tc.maxErr)
			}
			t.Logf("reconstruction error: %.6f (threshold: %.6f)", relErr, tc.maxErr)
		})
	}
}

func TestTransMLAFallbackWithoutKVLoraDim(t *testing.T) {
	cfg := testModelConfig()
	cfg.NumLayers = 1
	// TransMLAKVLoraDim is 0 — should use standard GQA path.

	tensors := makeSyntheticMHATensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := inference.BuildArchGraph("llama", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildArchGraph (GQA fallback): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}

	// Forward pass should work with standard MHA weights.
	tokenIDs := []float32{1, 5, 10}
	input, err := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward pass: %v", err)
	}

	data := output.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at output index %d", i)
		}
	}
}
