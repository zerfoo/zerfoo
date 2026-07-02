package vision

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestLTXVAEDecoderSkeleton_ForwardShape verifies the E127 conv/norm primitives
// (Conv3d, ConvTranspose3d, GroupNormalization, GELU) compose end-to-end into a
// running decode forward that 2x-upsamples each spatial dim and emits the
// configured output channels, with a finite result.
func TestLTXVAEDecoderSkeleton_ForwardShape(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	cfg := LTXVAEDecoderConfig{
		LatentChannels: 16,
		MidChannels:    8,
		OutChannels:    3,
		NumGroups:      4,
		Epsilon:        1e-5,
	}
	dec, err := NewLTXVAEDecoderSkeleton[float32](engine, ops, cfg)
	if err != nil {
		t.Fatalf("NewLTXVAEDecoderSkeleton: %v", err)
	}

	// Fixture latent [N=1, C=16, D=2, H=3, W=3].
	const n, d, h, w = 1, 2, 3, 3
	size := n * cfg.LatentChannels * d * h * w
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(math.Sin(float64(i) * 0.1)) // deterministic, bounded
	}
	latent, _ := tensor.New[float32]([]int{n, cfg.LatentChannels, d, h, w}, data)

	out, err := dec.Forward(context.Background(), latent)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	want := []int{n, cfg.OutChannels, 2 * d, 2 * h, 2 * w}
	got := out.Shape()
	if len(got) != len(want) {
		t.Fatalf("rank: got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("shape: got %v want %v", got, want)
		}
	}
	for i, v := range out.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite output at %d: %v", i, v)
		}
	}
}
