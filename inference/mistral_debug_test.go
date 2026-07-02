package inference

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestMistralDebugForward loads a real Mistral 7B GGUF and inspects the
// forward pass output to find where the computation diverges from correct
// behavior. Set MISTRAL_GGUF=/path/to/model.gguf to run.
//
// Expected: for prompt "Hello" (tokens [1, 23325] with BOS), the argmax of
// position-1 logits should be a reasonable continuation (e.g. "!" or ",").
// Ollama produces correct output with the same GGUF file.
func TestMistralDebugForward(t *testing.T) {
	ggufPath := os.Getenv("MISTRAL_GGUF")
	if ggufPath == "" {
		t.Skip("MISTRAL_GGUF not set; skipping debug test")
	}

	t.Logf("Loading GGUF from %s", ggufPath)
	gm, err := LoadGGUF(ggufPath)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}

	cfg := gm.Config
	t.Logf("Architecture: %s", cfg.Architecture)
	t.Logf("NumLayers: %d, HiddenSize: %d, IntermediateSize: %d", cfg.NumLayers, cfg.HiddenSize, cfg.IntermediateSize)
	t.Logf("NumHeads: %d, NumKVHeads: %d, HeadDim: %d", cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	t.Logf("VocabSize: %d, MaxSeqLen: %d", cfg.VocabSize, cfg.MaxSeqLen)
	t.Logf("RopeTheta: %f, RMSNormEps: %e", cfg.RopeTheta, cfg.RMSNormEps)
	t.Logf("SlidingWindow: %d, SlidingWindowPattern: %d", cfg.SlidingWindow, cfg.SlidingWindowPattern)

	// Log tensor names and their shapes/storage types for key tensors.
	keyTensors := []string{
		"model.embed_tokens.weight",
		"lm_head.weight",
		"model.norm.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.post_attention_layernorm.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.up_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
	}
	for _, name := range keyTensors {
		if t2, ok := gm.Tensors[name]; ok {
			t.Logf("  %-50s shape=%-20v storage=%T", name, t2.Shape(), t2.GetStorage())
		} else {
			t.Logf("  %-50s MISSING", name)
		}
	}

	// Check if lm_head is tied to embedding.
	_, hasLMHead := gm.Tensors["lm_head.weight"]
	t.Logf("Has separate lm_head.weight: %v", hasLMHead)

	// Create CPU engine.
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Build the graph.
	g, embWeight, err := buildArchGraph(cfg.Architecture, gm.Tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph: %v", err)
	}
	_ = embWeight

	t.Logf("Graph nodes: %d", len(g.Nodes()))

	// === Test 1: Check embedding lookup ===
	t.Run("embedding_sanity", func(t *testing.T) {
		emb := gm.Tensors["model.embed_tokens.weight"]
		if emb == nil {
			t.Fatal("no embedding tensor")
		}
		data := emb.Data()
		hiddenDim := emb.Shape()[1]

		// Check token 1 (BOS) embedding.
		bosEmb := data[1*hiddenDim : 2*hiddenDim]
		var bosSum, bosAbsMax float32
		for _, v := range bosEmb {
			bosSum += v
			if abs := float32(math.Abs(float64(v))); abs > bosAbsMax {
				bosAbsMax = abs
			}
		}
		t.Logf("BOS (token 1) embedding: sum=%f, absMax=%f, first5=%v",
			bosSum, bosAbsMax, bosEmb[:5])

		// Check token 23325 (Hello) embedding.
		if 23325 < emb.Shape()[0] {
			helloEmb := data[23325*hiddenDim : 22558*hiddenDim]
			var helloSum, helloAbsMax float32
			for _, v := range helloEmb {
				helloSum += v
				if abs := float32(math.Abs(float64(v))); abs > helloAbsMax {
					helloAbsMax = abs
				}
			}
			t.Logf("Hello (token 23325) embedding: sum=%f, absMax=%f, first5=%v",
				helloSum, helloAbsMax, helloEmb[:5])
		}
	})

	// === Test 2: Full forward pass — examine logits ===
	t.Run("full_forward_logits", func(t *testing.T) {
		// Input: BOS + "Hello" = tokens [1, 23325] (Mistral v0.3 tokenizer).
		inputData := []float32{1, 23325}
		input, err := tensor.New([]int{1, 2}, inputData)
		if err != nil {
			t.Fatalf("create input tensor: %v", err)
		}

		ctx := context.Background()
		output, err := g.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}

		shape := output.Shape()
		t.Logf("Output shape: %v", shape)

		logits := output.Data()
		vocabSize := shape[len(shape)-1]
		t.Logf("VocabSize from output: %d", vocabSize)

		// Extract logits for position 0 (BOS) and position 1 ("Hello").
		for pos := 0; pos < 2; pos++ {
			posLogits := logits[pos*vocabSize : (pos+1)*vocabSize]

			// Stats.
			var sum, min, max float64
			min = math.MaxFloat64
			max = -math.MaxFloat64
			argmax := 0
			for i, v := range posLogits {
				fv := float64(v)
				sum += fv
				if fv < min {
					min = fv
				}
				if fv > max {
					max = fv
					argmax = i
				}
			}
			mean := sum / float64(vocabSize)

			// Standard deviation.
			var variance float64
			for _, v := range posLogits {
				d := float64(v) - mean
				variance += d * d
			}
			std := math.Sqrt(variance / float64(vocabSize))

			t.Logf("Position %d logits: mean=%.4f, std=%.4f, min=%.4f, max=%.4f, argmax=%d, logit[argmax]=%.4f",
				pos, mean, std, min, max, argmax, float64(posLogits[argmax]))

			// Top-10 tokens.
			type tokenLogit struct {
				id    int
				logit float32
			}
			tls := make([]tokenLogit, vocabSize)
			for i, v := range posLogits {
				tls[i] = tokenLogit{i, v}
			}
			sort.Slice(tls, func(i, j int) bool { return tls[i].logit > tls[j].logit })
			t.Logf("  Top-10 tokens for position %d:", pos)
			for i := 0; i < 10 && i < len(tls); i++ {
				t.Logf("    rank %d: token=%d logit=%.4f", i, tls[i].id, tls[i].logit)
			}

			// Check specific tokens for position 1 (continuation of "Hello").
			if pos == 1 {
				checkTokens := map[string]int{
					" there": 1504, "!": 1686, ",": 1968, " world": 2294, " I": 1083,
					"-": 29501, "'": 29577, "/": 29516,
				}
				for name, id := range checkTokens {
					t.Logf("    logit[%d] (%s) = %.4f", id, name, posLogits[id])
				}
			}

			// Check for pathological patterns.
			if math.IsNaN(mean) || math.IsInf(mean, 0) {
				t.Errorf("Position %d: logits contain NaN/Inf", pos)
			}
			if std < 0.001 {
				t.Errorf("Position %d: logits have near-zero variance (std=%.6f) — all same value?", pos, std)
			}
		}
	})

	// === Test 3: Single BOS token forward (seqLen=1) ===
	t.Run("single_token_forward", func(t *testing.T) {
		inputData := []float32{1} // Just BOS.
		input, err := tensor.New([]int{1, 1}, inputData)
		if err != nil {
			t.Fatalf("create input tensor: %v", err)
		}

		ctx := context.Background()
		output, err := g.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}

		shape := output.Shape()
		logits := output.Data()
		vocabSize := shape[len(shape)-1]

		argmax := 0
		maxVal := logits[0]
		for i, v := range logits {
			if v > maxVal {
				maxVal = v
				argmax = i
			}
		}

		var sum float64
		for _, v := range logits {
			sum += float64(v)
		}
		mean := sum / float64(vocabSize)

		t.Logf("BOS-only forward: vocabSize=%d, argmax=%d, maxLogit=%.4f, mean=%.4f",
			vocabSize, argmax, maxVal, mean)

		// Check for NaN/Inf.
		for i, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("logit[%d] = %v (NaN/Inf)", i, v)
				break
			}
		}
	})

	// === Test 4: Compare position 0 logits (seqLen=1 vs seqLen=2) ===
	// If position 0 logits differ between seqLen=1 and seqLen=2, the attention
	// mask or positional encoding is leaking future information.
	t.Run("position0_consistency", func(t *testing.T) {
		// seqLen=1.
		input1, _ := tensor.New([]int{1, 1}, []float32{1})
		out1, err := g.Forward(context.Background(), input1)
		if err != nil {
			t.Fatalf("Forward seqLen=1: %v", err)
		}
		logits1 := out1.Data()

		// seqLen=2.
		input2, _ := tensor.New([]int{1, 2}, []float32{1, 23325})
		out2, err := g.Forward(context.Background(), input2)
		if err != nil {
			t.Fatalf("Forward seqLen=2: %v", err)
		}
		vocabSize := out2.Shape()[len(out2.Shape())-1]
		logits2 := out2.Data()[:vocabSize] // Position 0 only.

		// Compare.
		var maxDiff float64
		diffIdx := 0
		for i := range logits1 {
			d := math.Abs(float64(logits1[i]) - float64(logits2[i]))
			if d > maxDiff {
				maxDiff = d
				diffIdx = i
			}
		}
		t.Logf("Position 0 logit max difference (seqLen=1 vs seqLen=2): %.6f at token %d", maxDiff, diffIdx)
		t.Logf("  seqLen=1 logit[%d]=%.4f, seqLen=2 logit[%d]=%.4f",
			diffIdx, logits1[diffIdx], diffIdx, logits2[diffIdx])

		// For a causal model, position 0 should NOT depend on future tokens.
		// But in practice, floating-point differences from attention mask can
		// cause small diffs. A diff > 1.0 suggests a problem.
		if maxDiff > 1.0 {
			t.Errorf("Position 0 logits differ by %.4f between seqLen=1 and seqLen=2; possible attention mask issue", maxDiff)
		}
	})

	// === Test 5: Check RMSNorm of first layer ===
	t.Run("first_layer_rmsnorm", func(t *testing.T) {
		normW := gm.Tensors["model.layers.0.input_layernorm.weight"]
		if normW == nil {
			t.Fatal("missing model.layers.0.input_layernorm.weight")
		}

		// Get embedding for token 23325.
		emb := gm.Tensors["model.embed_tokens.weight"]
		hiddenDim := emb.Shape()[1]
		embData := emb.Data()
		helloEmb := embData[23325*hiddenDim : 22558*hiddenDim]

		// Manually compute RMSNorm.
		eps := float64(cfg.RMSNormEps)
		if eps == 0 {
			eps = 1e-5
		}

		var sumSq float64
		for _, v := range helloEmb {
			sumSq += float64(v) * float64(v)
		}
		rms := math.Sqrt(sumSq/float64(hiddenDim) + eps)

		normWData := normW.Data()
		result := make([]float64, hiddenDim)
		for i := 0; i < hiddenDim; i++ {
			result[i] = (float64(helloEmb[i]) / rms) * float64(normWData[i])
		}

		t.Logf("Manual RMSNorm of Hello embedding:")
		t.Logf("  RMS value: %.6f", rms)
		t.Logf("  First 5 output: [%.6f, %.6f, %.6f, %.6f, %.6f]",
			result[0], result[1], result[2], result[3], result[4])
		t.Logf("  Norm weight first 5: [%.6f, %.6f, %.6f, %.6f, %.6f]",
			normWData[0], normWData[1], normWData[2], normWData[3], normWData[4])

		var absMax float64
		for _, v := range result {
			if a := math.Abs(v); a > absMax {
				absMax = a
			}
		}
		t.Logf("  absMax of RMSNorm output: %.6f", absMax)
	})

	// === Test 6a: Compare Q4 GEMV lm_head against F32 manual computation ===
	t.Run("lm_head_q4_vs_f32", func(t *testing.T) {
		// Get the graph output (which uses Q4 GEMV for lm_head).
		inputData := []float32{1, 23325}
		input, err := tensor.New([]int{1, 2}, inputData)
		if err != nil {
			t.Fatalf("create input: %v", err)
		}

		ctx := context.Background()
		output, err := g.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}

		logits := output.Data()
		vocabSize := output.Shape()[2]

		// Position 1 Q4 logits.
		q4Logits := logits[vocabSize:]
		q4Argmax := 0
		for i, v := range q4Logits {
			if v > q4Logits[q4Argmax] {
				q4Argmax = i
			}
		}
		t.Logf("Q4 graph argmax (pos 1): %d (logit=%.4f)", q4Argmax, q4Logits[q4Argmax])

		// Get the lm_head weight in F32 (dequantized).
		lmHead := gm.Tensors["lm_head.weight"]
		if lmHead == nil {
			lmHead = gm.Tensors["model.embed_tokens.weight"]
		}
		lmF32 := lmHead.Data() // [vocabSize, hiddenDim]
		hiddenDim := cfg.HiddenSize

		// Now we need the hidden state BEFORE lm_head to manually compute
		// F32 logits. Run the graph with DEBUG to get the RMSNorm output.
		// Actually, we can extract it by looking at the graph's node outputs.
		//
		// Alternative approach: manually compute logits from the Q4 logits
		// by verifying consistency. If Q4 argmax differs from F32 argmax,
		// the Q4 GEMV is buggy.
		//
		// Since we can't easily get the intermediate hidden state, let's
		// use a different approach: create a tiny F32 lm_head and verify the
		// computation path.
		//
		// Actually, the best approach: manually compute hidden @ lmHead^T
		// using the F32 dequantized weight. For this we need the final
		// hidden state. Let me extract it by building a graph that stops
		// at the final RMSNorm output.

		// For now, just compare the Q4 logit ranking against what a pure F32
		// computation would give using a known test vector.
		// Use a random hidden state and check Q4 vs F32 lm_head.
		testHidden := make([]float32, hiddenDim)
		for i := range testHidden {
			testHidden[i] = float32(math.Sin(float64(i)*0.01)) * 0.5
		}

		// F32 computation: logit[j] = dot(testHidden, lmHead_row[j])
		f32Logits := make([]float32, vocabSize)
		for j := 0; j < vocabSize; j++ {
			var dot float32
			row := lmF32[j*hiddenDim : (j+1)*hiddenDim]
			for i := 0; i < hiddenDim; i++ {
				dot += testHidden[i] * row[i]
			}
			f32Logits[j] = dot
		}

		f32Argmax := 0
		for i, v := range f32Logits {
			if v > f32Logits[f32Argmax] {
				f32Argmax = i
			}
		}
		t.Logf("F32 manual lm_head argmax: %d (logit=%.4f)", f32Argmax, f32Logits[f32Argmax])

		// Q4 computation using engine.MatMul.
		hiddenTensor, _ := tensor.New([]int{1, hiddenDim}, testHidden)
		lmHeadShape := lmHead.Shape()
		lmHeadVT, _ := tensor.NewWithStorage[float32](
			[]int{lmHeadShape[1], lmHeadShape[0]},
			lmHead.GetStorage(),
		)
		q4Out, err := engine.MatMul(ctx, hiddenTensor, lmHeadVT)
		if err != nil {
			t.Fatalf("Q4 MatMul: %v", err)
		}
		q4TestLogits := q4Out.Data()
		q4TestArgmax := 0
		for i, v := range q4TestLogits {
			if v > q4TestLogits[q4TestArgmax] {
				q4TestArgmax = i
			}
		}
		t.Logf("Q4 engine MatMul argmax: %d (logit=%.4f)", q4TestArgmax, q4TestLogits[q4TestArgmax])

		// Compare F32 vs Q4 for the test vector.
		if f32Argmax != q4TestArgmax {
			t.Errorf("MISMATCH: F32 argmax=%d vs Q4 argmax=%d", f32Argmax, q4TestArgmax)
			// Show top 5 for each.
			type tl struct {
				id    int
				logit float32
			}
			f32Top := make([]tl, vocabSize)
			q4Top := make([]tl, vocabSize)
			for i := range vocabSize {
				f32Top[i] = tl{i, f32Logits[i]}
				q4Top[i] = tl{i, q4TestLogits[i]}
			}
			sort.Slice(f32Top, func(a, b int) bool { return f32Top[a].logit > f32Top[b].logit })
			sort.Slice(q4Top, func(a, b int) bool { return q4Top[a].logit > q4Top[b].logit })
			t.Logf("F32 top-5: %v", f32Top[:5])
			t.Logf("Q4  top-5: %v", q4Top[:5])
		} else {
			t.Logf("F32 and Q4 argmax match: %d", f32Argmax)
		}

		// Also check max absolute difference.
		var maxDiff float64
		for i := range vocabSize {
			d := math.Abs(float64(f32Logits[i]) - float64(q4TestLogits[i]))
			if d > maxDiff {
				maxDiff = d
			}
		}
		t.Logf("Max |F32 - Q4| logit difference: %.4f", maxDiff)
	})

	// === Test 6: Full generation via LoadFile + Generate (mirrors bench_tps) ===
	t.Run("full_generate", func(t *testing.T) {
		mdl, err := LoadFile(ggufPath, WithDevice("cpu"))
		if err != nil {
			t.Fatalf("LoadFile: %v", err)
		}

		// Generate with greedy (temp=0).
		result, err := mdl.Generate(context.Background(), "Hello",
			WithMaxTokens(4), WithTemperature(0))
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		t.Logf("Generate(\"Hello\", temp=0, tokens=4) = %q", result)

		// Also try with BOS explicitly, using just 1 token.
		result1, err := mdl.Generate(context.Background(), "Hello",
			WithMaxTokens(1), WithTemperature(0))
		if err != nil {
			t.Fatalf("Generate 1-token: %v", err)
		}
		t.Logf("Generate(\"Hello\", temp=0, tokens=1) = %q", result1)

		// Try "The meaning of life is" (the default bench_tps prompt).
		result2, err := mdl.Generate(context.Background(), "The meaning of life is",
			WithMaxTokens(4), WithTemperature(0))
		if err != nil {
			t.Fatalf("Generate default prompt: %v", err)
		}
		t.Logf("Generate(\"The meaning of life is\", temp=0, tokens=4) = %q", result2)
	})

	// === Test 7: Verify tensor name mapping ===
	t.Run("tensor_name_check", func(t *testing.T) {
		// Check that all expected tensors exist for all layers.
		missing := 0
		for i := 0; i < cfg.NumLayers; i++ {
			prefix := fmt.Sprintf("model.layers.%d.", i)
			expected := []string{
				prefix + "input_layernorm.weight",
				prefix + "self_attn.q_proj.weight",
				prefix + "self_attn.k_proj.weight",
				prefix + "self_attn.v_proj.weight",
				prefix + "self_attn.o_proj.weight",
				prefix + "post_attention_layernorm.weight",
				prefix + "mlp.gate_proj.weight",
				prefix + "mlp.up_proj.weight",
				prefix + "mlp.down_proj.weight",
			}
			for _, name := range expected {
				if _, ok := gm.Tensors[name]; !ok {
					t.Errorf("Missing tensor: %s", name)
					missing++
				}
			}
		}
		if missing == 0 {
			t.Logf("All %d layers have all expected tensors", cfg.NumLayers)
		}
	})
}
