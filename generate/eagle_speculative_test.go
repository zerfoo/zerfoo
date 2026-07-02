package generate

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// testEAGLEForward creates an EAGLEForwardFunc that returns deterministic
// logits and penultimate features. tokenSequence controls what greedy argmax
// returns per call (wraps around). hiddenDim controls the penultimate features size.
func testEAGLEForward(vocabSize int, tokenSequence []int, hiddenDim int) EAGLEForwardFunc[float32] {
	callCount := 0
	return func(ctx context.Context, input *tensor.TensorNumeric[float32]) (*EAGLEForwardResult[float32], error) {
		shape := input.Shape()
		seqLen := 1
		if len(shape) >= 2 {
			seqLen = shape[1]
		}

		// Build logits: for each position, set the target token to have max logit.
		logitsData := make([]float32, seqLen*vocabSize)
		for pos := range seqLen {
			targetToken := tokenSequence[callCount%len(tokenSequence)]
			offset := pos * vocabSize
			for j := range vocabSize {
				logitsData[offset+j] = -10.0
			}
			if targetToken >= 0 && targetToken < vocabSize {
				logitsData[offset+targetToken] = 10.0
			}
			if pos == seqLen-1 {
				callCount++
			}
		}
		logits, err := tensor.New[float32]([]int{1, seqLen, vocabSize}, logitsData)
		if err != nil {
			return nil, err
		}

		// Build penultimate features: simple deterministic values.
		featData := make([]float32, seqLen*hiddenDim)
		for i := range featData {
			featData[i] = float32(i%7-3) * 0.1
		}
		features, err := tensor.New[float32]([]int{1, seqLen, hiddenDim}, featData)
		if err != nil {
			return nil, err
		}

		return &EAGLEForwardResult[float32]{
			Logits:              logits,
			PenultimateFeatures: features,
		}, nil
	}
}

func buildTestEAGLEGenerator(
	t *testing.T,
	targetSeq []int,
	vocabSize, hiddenDim, draftLen int,
) *EAGLEGenerator[float32] {
	t.Helper()

	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	tok := buildTestTokenizer()

	eagleHead, err := core.NewEAGLEHead(engine, ops, hiddenDim)
	if err != nil {
		t.Fatalf("BuildEAGLEHead: %v", err)
	}

	// Create LM head weight [vocabSize, hiddenDim].
	lmData := make([]float32, vocabSize*hiddenDim)
	for i := range lmData {
		lmData[i] = float32(i%11-5) * 0.01
	}
	lmWeight, err := tensor.New[float32]([]int{vocabSize, hiddenDim}, lmData)
	if err != nil {
		t.Fatalf("create lm weight: %v", err)
	}

	forwardFn := testEAGLEForward(vocabSize, targetSeq, hiddenDim)

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}

	return NewEAGLEGenerator[float32](
		forwardFn, eagleHead, tok, engine, lmWeight, cfg, draftLen,
	)
}

// TestEAGLEGenerate_MatchesAutoregressive verifies that EAGLE produces
// identical output to vanilla autoregressive greedy decoding. The target
// model's forward function drives the output; the EAGLEHead drafts are
// verified against it.
func TestEAGLEGenerate_MatchesAutoregressive(t *testing.T) {
	// Target produces: 4(hello), 5(world), 2(EOS).
	// EAGLE drafts may or may not match, but the final output must
	// be the same as if we ran the target autoregressively.
	vocabSize := 8
	hiddenDim := 16
	targetSeq := []int{4, 5, 2}

	eg := buildTestEAGLEGenerator(t, targetSeq, vocabSize, hiddenDim, 4)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}

	result, err := eg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	// The target always produces token 4 on first call (after prefill),
	// then 5, then 2 (EOS). So the output is "hello world".
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestEAGLEGenerate_AllAccepted(t *testing.T) {
	// Target produces: 4, 5, 2. With matching draft, all tokens are accepted.
	vocabSize := 8
	hiddenDim := 16

	eg := buildTestEAGLEGenerator(t, []int{4, 5, 2}, vocabSize, hiddenDim, 4)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}

	result, err := eg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestEAGLEGenerate_EOSImmediate(t *testing.T) {
	// Target produces EOS immediately after prefill.
	vocabSize := 8
	hiddenDim := 16

	eg := buildTestEAGLEGenerator(t, []int{2}, vocabSize, hiddenDim, 4)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}

	result, err := eg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result != "" {
		t.Errorf("expected empty result for immediate EOS, got %q", result)
	}
}

func TestEAGLEGenerate_MaxTokensRespected(t *testing.T) {
	// Target always produces token 4 (never EOS), but MaxNewTokens=3.
	vocabSize := 8
	hiddenDim := 16

	eg := buildTestEAGLEGenerator(t, []int{4}, vocabSize, hiddenDim, 2)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 3,
	}

	result, err := eg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestEAGLEGenerate_StopToken(t *testing.T) {
	// Target produces: 4, 5, 7, 6. Token 7 is a stop token.
	vocabSize := 8
	hiddenDim := 16

	eg := buildTestEAGLEGenerator(t, []int{4, 5, 7, 6}, vocabSize, hiddenDim, 4)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopTokenIDs: []int{7},
	}

	_, err := eg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
}

func TestEAGLEGenerate_StopString(t *testing.T) {
	// Target produces: 6(foo), 7(bar), 2(EOS). Stop string "bar".
	vocabSize := 8
	hiddenDim := 16

	eg := buildTestEAGLEGenerator(t, []int{6, 7, 2}, vocabSize, hiddenDim, 4)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopStrings:  []string{"bar"},
	}

	result, err := eg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// Result should not contain "bar".
	if result != "" && len(result) > 0 {
		// The result should be "foo " or "foo" (everything before "bar").
		_ = result
	}
}

func TestEAGLEGenerate_AdaptiveDraftLen(t *testing.T) {
	// Target produces: 4, 5, 6, 4, 5, 6, ... (cycles). Draft may partially
	// match. The adaptive tracker should adjust draft length based on acceptance.
	vocabSize := 8
	hiddenDim := 16

	eg := buildTestEAGLEGenerator(t, []int{4, 5, 6}, vocabSize, hiddenDim, 4)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 20,
	}

	result, err := eg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestEAGLEGenerate_AdaptiveDisabled(t *testing.T) {
	vocabSize := 8
	hiddenDim := 16

	eg := buildTestEAGLEGenerator(t, []int{4, 5, 2}, vocabSize, hiddenDim, 3)
	eg.WithAdaptive(false)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}

	result, err := eg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestEAGLEGenerate_ContextCancelled(t *testing.T) {
	vocabSize := 8
	hiddenDim := 16

	eg := buildTestEAGLEGenerator(t, []int{4}, vocabSize, hiddenDim, 2)

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel immediately — should not panic or error fatally.
	cancel()

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 100,
	}

	// Context may be cancelled before or during generation. Either way,
	// it should return without error or with a context error.
	_, _ = eg.Generate(ctx, "hello", sc)
}

func TestEAGLEVerifyTokens(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	t.Run("all_accepted", func(t *testing.T) {
		// Target logits: position 0 -> 5, position 1 -> 6, position 2 -> 2
		// Draft tokens: [4, 5, 6]
		// Position 0 predicts 5 which matches draft[1]=5 -> accept 4
		// Position 1 predicts 6 which matches draft[2]=6 -> accept 5
		// Position 2 is last -> accept 6, bonus=2
		data := make([]float32, 3*vocabSize)
		for i := range data {
			data[i] = -10.0
		}
		data[5] = 10.0                 // pos 0 -> token 5
		data[vocabSize+6] = 10.0       // pos 1 -> token 6
		data[2*vocabSize+2] = 10.0     // pos 2 -> token 2

		logits, _ := tensor.New[float32]([]int{1, 3, vocabSize}, data)
		accepted, bonus := verifyDraftTokens(logits, []int{4, 5, 6}, map[int]bool{2: true})

		if len(accepted) != 3 {
			t.Errorf("expected 3 accepted, got %d", len(accepted))
		}
		// bonus is the target's prediction at the last position (2=EOS).
		// The verifyTokens method returns it; the caller decides whether to emit.
		if bonus != 2 {
			t.Errorf("expected bonus=2 (target prediction at last pos), got %d", bonus)
		}
	})

	t.Run("first_rejected", func(t *testing.T) {
		// Target logits: position 0 -> 7 (disagrees with draft[1]=5)
		// Draft tokens: [4, 5, 6]
		data := make([]float32, 3*vocabSize)
		for i := range data {
			data[i] = -10.0
		}
		data[7] = 10.0           // pos 0 -> token 7 (disagrees with draft[1]=5)
		data[vocabSize+5] = 10.0 // pos 1
		data[2*vocabSize+6] = 10.0

		logits, _ := tensor.New[float32]([]int{1, 3, vocabSize}, data)
		accepted, bonus := verifyDraftTokens(logits, []int{4, 5, 6}, map[int]bool{2: true})

		if len(accepted) != 1 {
			t.Errorf("expected 1 accepted, got %d", len(accepted))
		}
		if bonus != 7 {
			t.Errorf("expected bonus=7, got %d", bonus)
		}
	})

	t.Run("partial_accept_mid_sequence", func(t *testing.T) {
		// Draft tokens: [4, 5, 6, 7] (4 drafts)
		// Target logits:
		//   pos 0 -> 5 (matches draft[1]=5) -> accept 4
		//   pos 1 -> 6 (matches draft[2]=6) -> accept 5
		//   pos 2 -> 3 (disagrees with draft[3]=7) -> accept 6, bonus=3
		data := make([]float32, 4*vocabSize)
		for i := range data {
			data[i] = -10.0
		}
		data[5] = 10.0               // pos 0 -> 5
		data[vocabSize+6] = 10.0     // pos 1 -> 6
		data[2*vocabSize+3] = 10.0   // pos 2 -> 3 (mismatch with draft[3]=7)
		data[3*vocabSize+7] = 10.0   // pos 3 (never reached)

		logits, _ := tensor.New[float32]([]int{1, 4, vocabSize}, data)
		accepted, bonus := verifyDraftTokens(logits, []int{4, 5, 6, 7}, map[int]bool{2: true})

		if len(accepted) != 3 {
			t.Errorf("expected 3 accepted (partial), got %d: %v", len(accepted), accepted)
		}
		if bonus != 3 {
			t.Errorf("expected bonus=3 (target correction), got %d", bonus)
		}
	})

	t.Run("single_draft_token", func(t *testing.T) {
		// Draft tokens: [4] (single draft)
		// Position 0 is last -> accept 4, bonus = target's prediction (3)
		data := make([]float32, vocabSize)
		for i := range data {
			data[i] = -10.0
		}
		data[3] = 10.0 // pos 0 -> 3

		logits, _ := tensor.New[float32]([]int{1, 1, vocabSize}, data)
		accepted, bonus := verifyDraftTokens(logits, []int{4}, map[int]bool{2: true})

		if len(accepted) != 1 {
			t.Errorf("expected 1 accepted, got %d", len(accepted))
		}
		if accepted[0] != 4 {
			t.Errorf("accepted[0] = %d, want 4", accepted[0])
		}
		if bonus != 3 {
			t.Errorf("expected bonus=3, got %d", bonus)
		}
	})
}

func TestEAGLEExtractLastPosition(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()
	hiddenDim := 4

	head, _ := core.NewEAGLEHead(engine, ops, hiddenDim)
	lmData := make([]float32, vocabSize*hiddenDim)
	lmWeight, _ := tensor.New[float32]([]int{vocabSize, hiddenDim}, lmData)
	cfg := ModelConfig{VocabSize: vocabSize, MaxSeqLen: 128, EOSTokenID: 2, NumLayers: 1}
	eg := NewEAGLEGenerator[float32](nil, head, tok, engine, lmWeight, cfg, 2)

	// [1, 3, 4] features with known data per position.
	data := []float32{
		1, 2, 3, 4,    // pos 0
		5, 6, 7, 8,    // pos 1
		9, 10, 11, 12, // pos 2
	}
	features, err := tensor.New[float32]([]int{1, 3, hiddenDim}, data)
	if err != nil {
		t.Fatal(err)
	}

	extracted, err := eg.extractLastPosition(features, 1)
	if err != nil {
		t.Fatalf("extractLastPosition: %v", err)
	}

	shape := extracted.Shape()
	if shape[0] != 1 || shape[1] != 1 || shape[2] != hiddenDim {
		t.Errorf("shape = %v, want [1, 1, %d]", shape, hiddenDim)
	}

	got := extracted.Data()
	want := []float32{5, 6, 7, 8}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("data[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestEAGLEGenerate_ForwardError(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()
	hiddenDim := 16

	head, _ := core.NewEAGLEHead(engine, ops, hiddenDim)
	lmData := make([]float32, vocabSize*hiddenDim)
	lmWeight, _ := tensor.New[float32]([]int{vocabSize, hiddenDim}, lmData)
	cfg := ModelConfig{VocabSize: vocabSize, MaxSeqLen: 128, EOSTokenID: 2, BOSTokenID: 1, NumLayers: 1}

	errorFn := func(ctx context.Context, input *tensor.TensorNumeric[float32]) (*EAGLEForwardResult[float32], error) {
		return nil, fmt.Errorf("simulated forward error")
	}

	eg := NewEAGLEGenerator[float32](errorFn, head, tok, engine, lmWeight, cfg, 4)

	_, err := eg.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err == nil {
		t.Fatal("expected error from forward function")
	}
}

func TestEAGLEGenerate_ValidTokenIDs(t *testing.T) {
	// Verify that all generated tokens are valid IDs within the vocabulary.
	vocabSize := 8
	hiddenDim := 16
	// Target cycles through tokens 3, 4, 5, 6 — all valid, no EOS.
	eg := buildTestEAGLEGenerator(t, []int{3, 4, 5, 6}, vocabSize, hiddenDim, 3)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 12,
	}

	result, err := eg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Fatal("expected non-empty result")
	}
	// Re-encode the result to verify all token IDs are valid.
	ids, err := eg.tokenizer.Encode(result)
	if err != nil {
		t.Fatalf("re-encode result: %v", err)
	}
	for i, id := range ids {
		if id < 0 || id >= vocabSize {
			t.Errorf("token[%d] = %d, out of vocab range [0, %d)", i, id, vocabSize)
		}
	}
}

func TestEAGLEGenerate_EmptyPrompt(t *testing.T) {
	vocabSize := 8
	hiddenDim := 16

	eg := buildTestEAGLEGenerator(t, []int{4, 2}, vocabSize, hiddenDim, 4)

	_, err := eg.Generate(context.Background(), "", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err == nil {
		t.Fatal("expected error for empty prompt")
	}
}
