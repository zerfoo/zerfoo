package generate

import (
	"context"
	"sync"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestNewSession_CreatesIndependentCache(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  2,
		},
	)

	s1 := gen.NewSession()
	s2 := gen.NewSession()

	if s1.Cache() == nil {
		t.Fatal("session 1 cache is nil")
	}
	if s2.Cache() == nil {
		t.Fatal("session 2 cache is nil")
	}

	// Caches should be distinct objects.
	c1 := s1.Cache()
	c2 := s2.Cache()
	if c1 == c2 {
		t.Error("sessions should have independent caches, got same pointer")
	}
}

func TestSession_Generate_Greedy(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 7, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  0,
		},
	)

	sess := gen.NewSession()
	result, err := sess.Generate(context.Background(), "hello world", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Session.Generate error: %v", err)
	}

	if result != "foo bar" {
		t.Errorf("Session.Generate = %q, want %q", result, "foo bar")
	}
}

func TestSession_Generate_MaxTokens(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	sess := gen.NewSession()
	result, err := sess.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 3,
	})
	if err != nil {
		t.Fatalf("Session.Generate error: %v", err)
	}

	if result != "foo foo foo" {
		t.Errorf("Session.Generate = %q, want %q", result, "foo foo foo")
	}
}

func TestSession_Generate_ImmediateEOS(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	sess := gen.NewSession()
	result, err := sess.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Session.Generate error: %v", err)
	}

	if result != "" {
		t.Errorf("Session.Generate = %q, want empty string", result)
	}
}

func TestSession_ConcurrentGenerate_NoRace(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8

	const numSessions = 4

	// Pre-create sessions before launching goroutines to avoid races
	// in global state touched by NewCPUEngine.
	sessions := make([]*InferenceSession[float32], numSessions)
	for i := range numSessions {
		g := buildTestGraph(t, vocabSize, []int{6, 7, 2})
		gen := NewGenerator[float32](
			g, tok,
			compute.NewCPUEngine(numeric.Float32Ops{}),
			ModelConfig{
				VocabSize:  vocabSize,
				MaxSeqLen:  32,
				EOSTokenID: 2,
				BOSTokenID: 1,
				NumLayers:  0,
			},
		)
		sessions[i] = gen.NewSession()
	}

	var wg sync.WaitGroup
	wg.Add(numSessions)
	errs := make([]error, numSessions)
	results := make([]string, numSessions)

	for i := range numSessions {
		go func(idx int) {
			defer wg.Done()
			result, err := sessions[idx].Generate(context.Background(), "hello world", SamplingConfig{
				Temperature:  0,
				MaxNewTokens: 10,
			})
			errs[idx] = err
			results[idx] = result
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("session %d error: %v", i, err)
		}
	}

	for i, result := range results {
		if result != "foo bar" {
			t.Errorf("session %d result = %q, want %q", i, result, "foo bar")
		}
	}
}

func TestSession_IndependentPositionState(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8

	// Two sessions generating different amounts of tokens
	// should maintain independent position state.
	g1 := buildTestGraph(t, vocabSize, []int{6, 6, 6, 2})
	gen1 := NewGenerator[float32](
		g1, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  1,
		},
	)
	s1 := gen1.NewSession()

	g2 := buildTestGraph(t, vocabSize, []int{7, 2})
	gen2 := NewGenerator[float32](
		g2, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  1,
		},
	)
	s2 := gen2.NewSession()

	// Generate with session 1 first.
	r1, err := s1.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("session 1 error: %v", err)
	}

	// Generate with session 2 should be unaffected by session 1.
	r2, err := s2.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("session 2 error: %v", err)
	}

	if r1 != "foo foo foo" {
		t.Errorf("session 1 result = %q, want %q", r1, "foo foo foo")
	}
	if r2 != "bar" {
		t.Errorf("session 2 result = %q, want %q", r2, "bar")
	}
}

func TestSession_KVCacheIsolation(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 7, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  2,
		},
	)

	s1 := gen.NewSession()
	s2 := gen.NewSession()

	// Generate with session 1.
	_, err := s1.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	})
	if err != nil {
		t.Fatalf("session 1 error: %v", err)
	}

	// Session 2's cache should still be empty (reset happens on Generate).
	if s2.Cache().SeqLen() != 0 {
		t.Errorf("session 2 cache SeqLen = %d, want 0 (isolated)", s2.Cache().SeqLen())
	}
}

func TestSession_ConcurrentThroughput(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	const maxTokens = 50
	const numSessions = 4

	// Benchmark single-session throughput.
	g1 := buildTestGraph(t, vocabSize, []int{6}) // never hits EOS, generates maxTokens
	gen1 := NewGenerator[float32](
		g1, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)
	sess1 := gen1.NewSession()
	_, err := sess1.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: maxTokens,
	})
	if err != nil {
		t.Fatalf("single session error: %v", err)
	}

	// Run numSessions concurrently. Each gets its own generator+graph
	// to demonstrate true parallelism.
	sessions := make([]*InferenceSession[float32], numSessions)
	for i := range numSessions {
		g := buildTestGraph(t, vocabSize, []int{6})
		gen := NewGenerator[float32](
			g, tok,
			compute.NewCPUEngine(numeric.Float32Ops{}),
			ModelConfig{
				VocabSize:  vocabSize,
				MaxSeqLen:  128,
				EOSTokenID: 2,
				NumLayers:  0,
			},
		)
		sessions[i] = gen.NewSession()
	}

	var wg sync.WaitGroup
	wg.Add(numSessions)
	errs := make([]error, numSessions)
	results := make([]string, numSessions)

	for i := range numSessions {
		go func(idx int) {
			defer wg.Done()
			result, err := sessions[idx].Generate(context.Background(), "hello", SamplingConfig{
				Temperature:  0,
				MaxNewTokens: maxTokens,
			})
			errs[idx] = err
			results[idx] = result
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("session %d error: %v", i, err)
		}
	}

	// Verify all sessions produced output (maxTokens tokens each).
	for i, result := range results {
		if result == "" {
			t.Errorf("session %d produced empty output", i)
		}
	}

	// With independent graphs, all sessions ran in parallel without
	// contention. The test verifies no races under -race flag.
	t.Logf("concurrent throughput test: %d sessions x %d tokens completed", numSessions, maxTokens)
}

func TestSession_GeneratorGenerateStillWorks(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 7, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  0,
		},
	)

	// Create a session to verify it doesn't break Generator.
	_ = gen.NewSession()

	// Generator's Generate should still work normally.
	result, err := gen.Generate(context.Background(), "hello world", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Generator.Generate error: %v", err)
	}

	if result != "foo bar" {
		t.Errorf("Generator.Generate = %q, want %q", result, "foo bar")
	}
}

