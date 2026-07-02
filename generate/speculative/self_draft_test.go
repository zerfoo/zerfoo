package speculative

import (
	"context"
	"sync"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

// fixedLogitsForward returns a ForwardFunc that produces logits where the
// token at tokenSequence[callCount % len(tokenSequence)] has the highest value.
// This simulates a transformer that deterministically produces a known sequence.
func fixedLogitsForward(vocabSize int, tokenSequence []int) ForwardFunc[float32] {
	var mu sync.Mutex
	var callCount int

	return func(_ context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		seqLen := 1
		if shape := input.Shape(); len(shape) >= 2 {
			seqLen = shape[1]
		}

		mu.Lock()
		cc := callCount
		data := make([]float32, seqLen*vocabSize)
		for pos := range seqLen {
			targetToken := tokenSequence[(cc+pos)%len(tokenSequence)]
			offset := pos * vocabSize
			for j := range vocabSize {
				data[offset+j] = -10.0
			}
			if targetToken >= 0 && targetToken < vocabSize {
				data[offset+targetToken] = 10.0
			}
		}
		callCount += seqLen
		mu.Unlock()

		return tensor.New([]int{1, seqLen, vocabSize}, data)
	}
}

func TestSelfDraft_Generate(t *testing.T) {
	vocabSize := 8

	// Draft model (partial layers) produces tokens: 4, 5, 6, 7
	draftFn := fixedLogitsForward(vocabSize, []int{4, 5, 6, 7})
	// Full model produces tokens: 4, 5, 6, 7 (same — perfect match)
	verifyFn := fixedLogitsForward(vocabSize, []int{4, 5, 6, 7})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	if sd.DraftDepth() != 2 {
		t.Errorf("DraftDepth() = %d, want 2", sd.DraftDepth())
	}

	// Generate 4 draft tokens starting from token 1.
	draft, err := sd.Generate(context.Background(), []int{1}, 4)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if len(draft) != 4 {
		t.Fatalf("len(draft) = %d, want 4", len(draft))
	}

	// Draft should produce 4, 5, 6, 7
	want := []int{4, 5, 6, 7}
	for i, w := range want {
		if draft[i] != w {
			t.Errorf("draft[%d] = %d, want %d", i, draft[i], w)
		}
	}
}

func TestSelfDraft_Verify_AllAccepted(t *testing.T) {
	vocabSize := 8

	// Verify function produces the same sequence as draft.
	// Draft tokens: [4, 5, 6]. Verify at position 0 predicts 5 (matches [1]),
	// at position 1 predicts 6 (matches [2]), at position 2 predicts 7 (bonus).
	verifyFn := fixedLogitsForward(vocabSize, []int{5, 6, 7})
	draftFn := fixedLogitsForward(vocabSize, []int{4, 5, 6})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	accepted, correction, err := sd.Verify(context.Background(), []int{4, 5, 6})
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}

	if accepted != 3 {
		t.Errorf("accepted = %d, want 3", accepted)
	}
	if correction != 7 {
		t.Errorf("correction = %d, want 7 (bonus token)", correction)
	}
}

func TestSelfDraft_Verify_PartialAcceptance(t *testing.T) {
	vocabSize := 8

	// Draft tokens: [4, 5, 6].
	// Verify predicts: pos 0 -> 5 (matches draft[1]=5), pos 1 -> 3 (mismatch with draft[2]=6).
	verifyFn := fixedLogitsForward(vocabSize, []int{5, 3})
	draftFn := fixedLogitsForward(vocabSize, []int{4, 5, 6})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	accepted, correction, err := sd.Verify(context.Background(), []int{4, 5, 6})
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}

	if accepted != 2 {
		t.Errorf("accepted = %d, want 2", accepted)
	}
	if correction != 3 {
		t.Errorf("correction = %d, want 3", correction)
	}
}

func TestSelfDraft_Verify_FirstRejected(t *testing.T) {
	vocabSize := 8

	// Draft tokens: [4, 5, 6].
	// Verify predicts: pos 0 -> 7 (mismatch with draft[1]=5).
	verifyFn := fixedLogitsForward(vocabSize, []int{7})
	draftFn := fixedLogitsForward(vocabSize, []int{4, 5, 6})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	accepted, correction, err := sd.Verify(context.Background(), []int{4, 5, 6})
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}

	if accepted != 1 {
		t.Errorf("accepted = %d, want 1", accepted)
	}
	if correction != 7 {
		t.Errorf("correction = %d, want 7", correction)
	}
}

func TestSelfDraft_AcceptanceRate_Perfect(t *testing.T) {
	vocabSize := 8

	// Draft and verify produce identical sequences -> alpha = 1.0.
	// Draft generates: 4, 5, 6, 7 (from calling draftFn 4 times).
	// Verify sees [4, 5, 6, 7] and at each position predicts the next one.
	draftFn := fixedLogitsForward(vocabSize, []int{4, 5, 6, 7})
	verifyFn := fixedLogitsForward(vocabSize, []int{5, 6, 7, 4})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	alpha, err := sd.AcceptanceRate(context.Background(), []int{1}, 4)
	if err != nil {
		t.Fatalf("AcceptanceRate: %v", err)
	}

	if alpha != 1.0 {
		t.Errorf("alpha = %f, want 1.0", alpha)
	}
}

func TestSelfDraft_AcceptanceRate_Partial(t *testing.T) {
	vocabSize := 8

	// Draft generates: 4, 5, 6, 7.
	// Verify: pos 0 -> 5 (match), pos 1 -> 6 (match), pos 2 -> 3 (mismatch).
	// Accepted: 3 out of 4 = 0.75.
	draftFn := fixedLogitsForward(vocabSize, []int{4, 5, 6, 7})
	verifyFn := fixedLogitsForward(vocabSize, []int{5, 6, 3})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	alpha, err := sd.AcceptanceRate(context.Background(), []int{1}, 4)
	if err != nil {
		t.Fatalf("AcceptanceRate: %v", err)
	}

	// 3/4 = 0.75
	if alpha < 0.74 || alpha > 0.76 {
		t.Errorf("alpha = %f, want ~0.75", alpha)
	}
}

func TestSelfDraft_AcceptanceRate_AboveThreshold(t *testing.T) {
	vocabSize := 8

	// Simulate a scenario where draft matches at least 40% of the time.
	// Draft: 4, 5, 6, 7, 4. Verify agrees on first 3, disagrees on 4th.
	// 4/5 accepted = 0.8 > 0.4.
	draftFn := fixedLogitsForward(vocabSize, []int{4, 5, 6, 7, 4})
	verifyFn := fixedLogitsForward(vocabSize, []int{5, 6, 7, 4, 5})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	alpha, err := sd.AcceptanceRate(context.Background(), []int{1}, 5)
	if err != nil {
		t.Fatalf("AcceptanceRate: %v", err)
	}

	if alpha <= 0.4 {
		t.Errorf("alpha = %f, want > 0.4", alpha)
	}
}

func TestSelfDraft_Generate_EmptyInput(t *testing.T) {
	vocabSize := 8
	draftFn := fixedLogitsForward(vocabSize, []int{4})
	verifyFn := fixedLogitsForward(vocabSize, []int{4})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	_, err := sd.Generate(context.Background(), nil, 4)
	if err == nil {
		t.Error("expected error for empty input")
	}
}

func TestSelfDraft_Generate_ZeroK(t *testing.T) {
	vocabSize := 8
	draftFn := fixedLogitsForward(vocabSize, []int{4})
	verifyFn := fixedLogitsForward(vocabSize, []int{4})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	draft, err := sd.Generate(context.Background(), []int{1}, 0)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(draft) != 0 {
		t.Errorf("len(draft) = %d, want 0", len(draft))
	}
}

func TestSelfDraft_Verify_Empty(t *testing.T) {
	vocabSize := 8
	draftFn := fixedLogitsForward(vocabSize, []int{4})
	verifyFn := fixedLogitsForward(vocabSize, []int{4})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	accepted, correction, err := sd.Verify(context.Background(), nil)
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if accepted != 0 {
		t.Errorf("accepted = %d, want 0", accepted)
	}
	if correction != -1 {
		t.Errorf("correction = %d, want -1", correction)
	}
}

func TestSelfDraft_ContextCancellation(t *testing.T) {
	vocabSize := 8
	draftFn := fixedLogitsForward(vocabSize, []int{4, 5, 6, 7})
	verifyFn := fixedLogitsForward(vocabSize, []int{4, 5, 6, 7})

	sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, 4, 2)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	draft, err := sd.Generate(ctx, []int{1}, 100)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// Should produce 0 tokens since context is already canceled.
	if len(draft) != 0 {
		t.Errorf("len(draft) = %d, want 0 with canceled context", len(draft))
	}
}

func TestNewSelfDraft_DraftDepthClamping(t *testing.T) {
	vocabSize := 8
	draftFn := fixedLogitsForward(vocabSize, []int{4})
	verifyFn := fixedLogitsForward(vocabSize, []int{4})

	tests := []struct {
		name       string
		numLayers  int
		draftDepth int
		wantDepth  int
	}{
		{"zero depth defaults to N/2", 4, 0, 2},
		{"negative depth defaults to N/2", 4, -1, 2},
		{"depth >= numLayers clamped to N/2", 4, 4, 2},
		{"depth > numLayers clamped to N/2", 4, 10, 2},
		{"1 layer model gets depth 1", 1, 0, 1},
		{"2 layer model with depth 1", 2, 1, 1},
		{"explicit valid depth preserved", 6, 3, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sd := NewSelfDraft[float32](draftFn, verifyFn, vocabSize, tt.numLayers, tt.draftDepth)
			if sd.DraftDepth() != tt.wantDepth {
				t.Errorf("DraftDepth() = %d, want %d", sd.DraftDepth(), tt.wantDepth)
			}
		})
	}
}
