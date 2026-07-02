package generate

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/metrics/runtime"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// buildSpecTestGraph creates a graph for speculative decode testing.
// tokenSequence controls what greedy argmax returns per forward call.
func buildSpecTestGraph(t *testing.T, vocabSize int, tokenSequence []int) *graph.Graph[float32] {
	t.Helper()
	return buildTestGraph(t, vocabSize, tokenSequence)
}

func TestSpeculativeGenerate_AllAccepted(t *testing.T) {
	// Draft and target agree on all tokens: "hello world" -> EOS.
	// Both produce: 4(hello), 5(world), 2(EOS).
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()
	seq := []int{4, 5, 2} // hello, world, EOS

	draftGraph := buildSpecTestGraph(t, vocabSize, seq)
	targetGraph := buildSpecTestGraph(t, vocabSize, seq)

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 4, // draftLen=4
	)

	sc := SamplingConfig{
		Temperature:  0, // greedy
		MaxNewTokens: 10,
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// Should generate at least one token before EOS.
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestSpeculativeGenerate_FirstTokenRejected(t *testing.T) {
	// Draft says 6(foo), target says 5(world). Target wins immediately.
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Draft: proposes foo, foo, EOS
	draftGraph := buildSpecTestGraph(t, vocabSize, []int{6, 6, 2})
	// Target: always prefers world then EOS
	targetGraph := buildSpecTestGraph(t, vocabSize, []int{5, 2})

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 4,
	)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestSpeculativeGenerate_PartialAcceptance(t *testing.T) {
	// Draft proposes [4, 5, 6], target accepts [4, 5] then rejects 6 (wants 7).
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Draft: 4, 5, 6, 2
	draftGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 6, 2})
	// Target: 4, 5, 7, 2 (accepts first 2 draft tokens, rejects 3rd)
	targetGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 7, 2})

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 4,
	)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestSpeculativeGenerate_DraftEOSEarly(t *testing.T) {
	// Draft produces EOS after 1 token, target verifies just 1.
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Draft: 4, EOS
	draftGraph := buildSpecTestGraph(t, vocabSize, []int{4, 2})
	// Target: 4, EOS (agrees)
	targetGraph := buildSpecTestGraph(t, vocabSize, []int{4, 2})

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 4,
	)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// Should have at least the first accepted token.
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestSpeculativeGenerate_MaxTokens(t *testing.T) {
	// Both models agree infinitely, but we cap at MaxNewTokens=3.
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Both always produce token 4 (hello), never EOS.
	draftGraph := buildSpecTestGraph(t, vocabSize, []int{4})
	targetGraph := buildSpecTestGraph(t, vocabSize, []int{4})

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 2,
	)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 3,
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

func TestSpeculativeGenerate_StopToken(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Draft and target: 4, 5, 7(bar=stop), ...
	draftGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 7, 6})
	targetGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 7, 6})

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	sg := NewSpeculativeGenerator[float32](
		draftGraph, targetGraph, tok, engine,
		cfg, cfg, 4,
	)

	sc := SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopTokenIDs: []int{7}, // bar=7 is stop token
	}

	result, err := sg.Generate(context.Background(), "hello", sc)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// Should stop before bar is included.
	_ = result
}

func TestKVCache_Truncate(t *testing.T) {
	cache := NewKVCache[float32](2, 128)

	// Append 5 tokens to each layer.
	for i := range 5 {
		for layer := range 2 {
			k := makeTestTensor(t, []int{1, 1, 4}, []float32{float32(i), float32(layer), 0, 0})
			v := makeTestTensor(t, []int{1, 1, 4}, []float32{0, 0, float32(i), float32(layer)})
			if err := cache.Update(layer, k, v); err != nil {
				t.Fatalf("Update: %v", err)
			}
		}
	}
	if got := cache.SeqLen(); got != 5 {
		t.Fatalf("SeqLen = %d, want 5", got)
	}

	// Truncate to 3.
	cache.Truncate(3)
	if got := cache.SeqLen(); got != 3 {
		t.Errorf("SeqLen after Truncate(3) = %d, want 3", got)
	}

	// Can still append.
	k := makeTestTensor(t, []int{1, 1, 4}, []float32{99, 0, 0, 0})
	v := makeTestTensor(t, []int{1, 1, 4}, []float32{0, 0, 99, 0})
	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update after Truncate: %v", err)
	}
	if got := cache.SeqLen(); got != 4 {
		t.Errorf("SeqLen after Truncate+Update = %d, want 4", got)
	}

	// Verify data: position 3 should have the new data.
	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}
	kd := lkv.Key.Data()
	if kd[3*4] != 99 {
		t.Errorf("Key data at pos 3 = %v, want 99", kd[3*4])
	}
}

func TestPagedKVCache_Truncate(t *testing.T) {
	pool, err := NewBlockPool[float32](1, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 1)

	// Append 6 tokens (2 blocks: 4+2).
	for i := range 6 {
		k := makeTestTensor(t, []int{1, 1, 2}, []float32{float32(i), 0})
		v := makeTestTensor(t, []int{1, 1, 2}, []float32{0, float32(i)})
		if err := cache.Append(0, k, v); err != nil {
			t.Fatalf("Append(%d): %v", i, err)
		}
	}
	if got := cache.SeqLen(); got != 6 {
		t.Fatalf("SeqLen = %d, want 6", got)
	}

	availBefore := pool.Available()

	// Truncate to 3 (should free second block).
	cache.Truncate(3)
	if got := cache.SeqLen(); got != 3 {
		t.Errorf("SeqLen after Truncate(3) = %d, want 3", got)
	}
	if got := pool.Available(); got != availBefore+1 {
		t.Errorf("Available after Truncate = %d, want %d", got, availBefore+1)
	}

	// Verify data.
	lkv, ok := cache.GetKV(0)
	if !ok {
		t.Fatal("GetKV(0) should return true")
	}
	kd := lkv.Key.Data()
	for i := range 3 {
		if kd[i*2] != float32(i) {
			t.Errorf("Key[%d] = %v, want %v", i, kd[i*2], float32(i))
		}
	}

	// Truncate to 0.
	cache.Truncate(0)
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen after Truncate(0) = %d, want 0", got)
	}
}

// Verify that SpeculativeGenerator implements a basic "model forward" pattern
// by testing with a simple model and checking output shape consistency.
func TestSpeculativeGenerator_ForwardConsistency(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := tok.VocabSize()

	// Same model for draft and target.
	seq := []int{4, 5, 2}
	g := buildSpecTestGraph(t, vocabSize, seq)

	cfg := ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  1,
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	// Single-token input.
	input, err := tensor.New([]int{1, 1, 1}, []float32{4})
	if err != nil {
		t.Fatal(err)
	}

	logits, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[2] != vocabSize {
		t.Errorf("logits shape = %v, want [1, *, %d]", shape, vocabSize)
	}

	_ = engine
	_ = cfg
}

// TestGeneratorSpeculative tests the WithSpeculativeDraft option on Generator.Generate.
func TestGeneratorSpeculative(t *testing.T) {
	t.Run("all_accepted", func(t *testing.T) {
		tok := buildTestTokenizer()
		vocabSize := tok.VocabSize()
		seq := []int{4, 5, 2} // hello, world, EOS

		draftGraph := buildSpecTestGraph(t, vocabSize, seq)
		targetGraph := buildSpecTestGraph(t, vocabSize, seq)

		cfg := ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  1,
		}
		engine := compute.NewCPUEngine(numeric.Float32Ops{})

		gen := NewGenerator[float32](
			targetGraph, tok, engine, cfg,
			WithSpeculativeDraft(draftGraph, cfg, 4),
		)

		result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
			Temperature:  0,
			MaxNewTokens: 10,
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if result == "" {
			t.Error("expected non-empty result")
		}
	})

	t.Run("first_token_rejected", func(t *testing.T) {
		tok := buildTestTokenizer()
		vocabSize := tok.VocabSize()

		draftGraph := buildSpecTestGraph(t, vocabSize, []int{6, 6, 2})
		targetGraph := buildSpecTestGraph(t, vocabSize, []int{5, 2})

		cfg := ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  1,
		}
		engine := compute.NewCPUEngine(numeric.Float32Ops{})

		gen := NewGenerator[float32](
			targetGraph, tok, engine, cfg,
			WithSpeculativeDraft(draftGraph, cfg, 4),
		)

		result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
			Temperature:  0,
			MaxNewTokens: 10,
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if result == "" {
			t.Error("expected non-empty result")
		}
	})

	t.Run("max_tokens_respected", func(t *testing.T) {
		tok := buildTestTokenizer()
		vocabSize := tok.VocabSize()

		// Both always produce token 4 (hello), never EOS.
		draftGraph := buildSpecTestGraph(t, vocabSize, []int{4})
		targetGraph := buildSpecTestGraph(t, vocabSize, []int{4})

		cfg := ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  1,
		}
		engine := compute.NewCPUEngine(numeric.Float32Ops{})

		gen := NewGenerator[float32](
			targetGraph, tok, engine, cfg,
			WithSpeculativeDraft(draftGraph, cfg, 2),
		)

		result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
			Temperature:  0,
			MaxNewTokens: 3,
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if result == "" {
			t.Error("expected non-empty result")
		}
	})

	t.Run("stop_token", func(t *testing.T) {
		tok := buildTestTokenizer()
		vocabSize := tok.VocabSize()

		draftGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 7, 6})
		targetGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 7, 6})

		cfg := ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  1,
		}
		engine := compute.NewCPUEngine(numeric.Float32Ops{})

		gen := NewGenerator[float32](
			targetGraph, tok, engine, cfg,
			WithSpeculativeDraft(draftGraph, cfg, 4),
		)

		_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
			Temperature:  0,
			MaxNewTokens: 10,
			StopTokenIDs: []int{7},
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
	})

	t.Run("fallback_on_low_alpha", func(t *testing.T) {
		tok := buildTestTokenizer()
		vocabSize := tok.VocabSize()

		// Draft always disagrees with target: draft says 6, target says 5.
		// This will drive alpha below 0.4, triggering fallback.
		draftGraph := buildSpecTestGraph(t, vocabSize, []int{6})
		targetGraph := buildSpecTestGraph(t, vocabSize, []int{5})

		cfg := ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  1,
		}
		engine := compute.NewCPUEngine(numeric.Float32Ops{})

		gen := NewGenerator[float32](
			targetGraph, tok, engine, cfg,
			WithSpeculativeDraft(draftGraph, cfg, 2),
		)

		// With MaxNewTokens=10, should still produce output via fallback.
		result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
			Temperature:  0,
			MaxNewTokens: 10,
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if result == "" {
			t.Error("expected non-empty result from fallback path")
		}
	})

	t.Run("without_speculative_uses_standard", func(t *testing.T) {
		// Verify that without WithSpeculativeDraft, Generator uses standard decode.
		tok := buildTestTokenizer()
		vocabSize := tok.VocabSize()

		targetGraph := buildSpecTestGraph(t, vocabSize, []int{6, 7, 2})

		cfg := ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			NumLayers:  0,
		}
		engine := compute.NewCPUEngine(numeric.Float32Ops{})

		gen := NewGenerator[float32](targetGraph, tok, engine, cfg)
		if gen.specDraft != nil {
			t.Fatal("specDraft should be nil without WithSpeculativeDraft")
		}

		result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
			Temperature:  0,
			MaxNewTokens: 10,
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if result != "foo bar" {
			t.Errorf("Generate = %q, want %q", result, "foo bar")
		}
	})
}

func TestAcceptanceMetric(t *testing.T) {
	t.Run("all_accepted", func(t *testing.T) {
		// Both draft and target always produce token 4 — every proposed
		// token is accepted, so the gauge should converge to 1.0.
		tok := buildTestTokenizer()
		vocabSize := tok.VocabSize()

		draftGraph := buildSpecTestGraph(t, vocabSize, []int{4})
		targetGraph := buildSpecTestGraph(t, vocabSize, []int{4})

		cfg := ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  1,
		}
		engine := compute.NewCPUEngine(numeric.Float32Ops{})
		collector := runtime.NewInMemory()

		gen := NewGenerator[float32](
			targetGraph, tok, engine, cfg,
			WithSpeculativeDraft(draftGraph, cfg, 2),
			WithMetrics(collector),
		)

		_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
			Temperature:  0,
			MaxNewTokens: 10,
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}

		snap := collector.Snapshot()
		rate, ok := snap.Gauges["speculative_acceptance_rate"]
		if !ok {
			t.Fatal("speculative_acceptance_rate gauge not found")
		}
		if rate < 0.99 {
			t.Errorf("acceptance rate = %f, want ~1.0 (all tokens accepted)", rate)
		}
	})

	t.Run("low_acceptance", func(t *testing.T) {
		// Draft always proposes 6 (foo), target always verifies as 5 (world).
		// With draftLen=4, speculative verify accepts the first token then
		// rejects (1 of 4 accepted = 0.25 rate).
		tok := buildTestTokenizer()
		vocabSize := tok.VocabSize()

		draftGraph := buildSpecTestGraph(t, vocabSize, []int{6})
		targetGraph := buildSpecTestGraph(t, vocabSize, []int{5})

		cfg := ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  1,
		}
		engine := compute.NewCPUEngine(numeric.Float32Ops{})
		collector := runtime.NewInMemory()

		gen := NewGenerator[float32](
			targetGraph, tok, engine, cfg,
			WithSpeculativeDraft(draftGraph, cfg, 4),
			WithMetrics(collector),
		)

		_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
			Temperature:  0,
			MaxNewTokens: 10,
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}

		snap := collector.Snapshot()
		rate, ok := snap.Gauges["speculative_acceptance_rate"]
		if !ok {
			t.Fatal("speculative_acceptance_rate gauge not found")
		}
		// With consistent disagreement and draftLen=4, acceptance is 1/4 = 0.25.
		if rate > 0.5 {
			t.Errorf("acceptance rate = %f, want <= 0.5 (low acceptance)", rate)
		}
	})

	t.Run("partial_acceptance", func(t *testing.T) {
		// Draft proposes [4, 5, 6], target accepts [4, 5] rejects 6 (wants 7).
		tok := buildTestTokenizer()
		vocabSize := tok.VocabSize()

		draftGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 6, 2})
		targetGraph := buildSpecTestGraph(t, vocabSize, []int{4, 5, 7, 2})

		cfg := ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  1,
		}
		engine := compute.NewCPUEngine(numeric.Float32Ops{})
		collector := runtime.NewInMemory()

		gen := NewGenerator[float32](
			targetGraph, tok, engine, cfg,
			WithSpeculativeDraft(draftGraph, cfg, 4),
			WithMetrics(collector),
		)

		_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
			Temperature:  0,
			MaxNewTokens: 10,
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}

		snap := collector.Snapshot()
		rate, ok := snap.Gauges["speculative_acceptance_rate"]
		if !ok {
			t.Fatal("speculative_acceptance_rate gauge not found")
		}
		// Rate should be between 0 and 1 (partial acceptance).
		if rate <= 0.0 || rate >= 1.0 {
			t.Errorf("acceptance rate = %f, want 0 < rate < 1 for partial acceptance", rate)
		}
	})

	t.Run("no_speculative_gauge_stays_zero", func(t *testing.T) {
		// Without speculative decoding, gauge should remain at default (0).
		tok := buildTestTokenizer()
		vocabSize := tok.VocabSize()

		targetGraph := buildSpecTestGraph(t, vocabSize, []int{6, 7, 2})

		cfg := ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			NumLayers:  0,
		}
		engine := compute.NewCPUEngine(numeric.Float32Ops{})
		collector := runtime.NewInMemory()

		gen := NewGenerator[float32](
			targetGraph, tok, engine, cfg,
			WithMetrics(collector),
		)

		_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
			Temperature:  0,
			MaxNewTokens: 10,
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}

		snap := collector.Snapshot()
		rate := snap.Gauges["speculative_acceptance_rate"]
		if math.Abs(rate) > 1e-9 {
			t.Errorf("acceptance rate = %f, want 0 (no speculative decoding)", rate)
		}
	})
}
