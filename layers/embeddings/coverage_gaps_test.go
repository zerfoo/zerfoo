package embeddings

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// ---------------------------------------------------------------------------
// AttentionScaleFactor: zero value returns 1.0
// ---------------------------------------------------------------------------

func TestRoPE_AttentionScaleFactor_Zero(t *testing.T) {
	rpe := &RotaryPositionalEmbedding[float32]{attnScaleFactor: 0}
	got := rpe.AttentionScaleFactor()
	if got != 1.0 {
		t.Errorf("AttentionScaleFactor() = %v, want 1.0", got)
	}
}

func TestRoPE_AttentionScaleFactor_NonZero(t *testing.T) {
	rpe := &RotaryPositionalEmbedding[float32]{attnScaleFactor: 2.5}
	got := rpe.AttentionScaleFactor()
	if got != 2.5 {
		t.Errorf("AttentionScaleFactor() = %v, want 2.5", got)
	}
}

// ---------------------------------------------------------------------------
// TokenEmbedding Backward: 1D gradient (< 2 dims)
// ---------------------------------------------------------------------------

func TestTokenEmbedding_Backward_1DGradient(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	te, err := NewTokenEmbedding[float32](engine, 4, 3)
	if err != nil {
		t.Fatalf("NewTokenEmbedding: %v", err)
	}

	// Run forward to populate inputTokenIDs cache.
	input, _ := tensor.New[float32]([]int{2}, []float32{0, 1})
	_, err = te.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Backward with 1D gradient should error.
	grad1D, _ := tensor.New[float32]([]int{6}, []float32{1, 2, 3, 4, 5, 6})
	_, err = te.Backward(context.Background(), types.FullBackprop, grad1D)
	if err == nil {
		t.Error("expected error for 1D gradient")
	}
}
