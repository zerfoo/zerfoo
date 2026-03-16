package generate

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestBatchGenerate_MultipleRequests(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 7, 2, 6, 7, 2})

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

	requests := []BatchRequest{
		{Prompt: "hello", Sampling: SamplingConfig{Temperature: 0, MaxNewTokens: 10}},
		{Prompt: "world", Sampling: SamplingConfig{Temperature: 0, MaxNewTokens: 10}},
	}

	results := gen.BatchGenerate(context.Background(), requests)
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}

	for i, r := range results {
		if r.Err != nil {
			t.Errorf("results[%d] error: %v", i, r.Err)
		}
	}
}

func TestBatchGenerate_Empty(t *testing.T) {
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
			NumLayers:  0,
		},
	)

	results := gen.BatchGenerate(context.Background(), nil)
	if len(results) != 0 {
		t.Errorf("got %d results for empty batch, want 0", len(results))
	}
}

func TestBatchGenerate_SingleRequest(t *testing.T) {
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
			NumLayers:  0,
		},
	)

	requests := []BatchRequest{
		{Prompt: "hello", Sampling: SamplingConfig{Temperature: 0, MaxNewTokens: 10}},
	}

	results := gen.BatchGenerate(context.Background(), requests)
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if results[0].Err != nil {
		t.Errorf("unexpected error: %v", results[0].Err)
	}
	if results[0].Text != "foo" {
		t.Errorf("got %q, want %q", results[0].Text, "foo")
	}
}

func TestBatchGenerate_ContextCancellation(t *testing.T) {
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

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	requests := []BatchRequest{
		{Prompt: "hello", Sampling: SamplingConfig{Temperature: 0, MaxNewTokens: 100}},
		{Prompt: "world", Sampling: SamplingConfig{Temperature: 0, MaxNewTokens: 100}},
	}

	results := gen.BatchGenerate(ctx, requests)
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	// Results should either error or produce short output.
}

func TestBatchGenerateStream_MismatchedLengths(t *testing.T) {
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
			NumLayers:  0,
		},
	)

	requests := []BatchRequest{
		{Prompt: "hello", Sampling: SamplingConfig{Temperature: 0, MaxNewTokens: 10}},
	}

	// Mismatched: 1 request but 0 streams.
	errs := gen.BatchGenerateStream(context.Background(), requests, nil)
	if len(errs) != 1 {
		t.Fatalf("got %d errors, want 1", len(errs))
	}
	if errs[0] == nil {
		t.Error("expected error for mismatched lengths")
	}
}

func TestBatchHelper_PadPrompts(t *testing.T) {
	gen := &Generator[float32]{}
	h := newBatchHelper(gen)

	prompts := [][]int{
		{1, 2, 3},
		{4, 5},
		{6},
	}

	padded, lengths := h.padPrompts(prompts, 0)
	if len(padded) != 3 {
		t.Fatalf("got %d padded, want 3", len(padded))
	}

	// All padded to length 3.
	for i, p := range padded {
		if len(p) != 3 {
			t.Errorf("padded[%d] length = %d, want 3", i, len(p))
		}
	}

	// Check lengths.
	wantLengths := []int{3, 2, 1}
	for i, l := range lengths {
		if l != wantLengths[i] {
			t.Errorf("lengths[%d] = %d, want %d", i, l, wantLengths[i])
		}
	}

	// Check padding: [1,2,3], [0,4,5], [0,0,6]
	if padded[0][0] != 1 || padded[0][1] != 2 || padded[0][2] != 3 {
		t.Errorf("padded[0] = %v, want [1,2,3]", padded[0])
	}
	if padded[1][0] != 0 || padded[1][1] != 4 || padded[1][2] != 5 {
		t.Errorf("padded[1] = %v, want [0,4,5]", padded[1])
	}
	if padded[2][0] != 0 || padded[2][1] != 0 || padded[2][2] != 6 {
		t.Errorf("padded[2] = %v, want [0,0,6]", padded[2])
	}
}
