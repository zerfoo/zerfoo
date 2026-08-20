package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGQA_PrefillDecodeRoPEPositionParity red-proofs zerfoo#990.
//
// A multi-token prefill followed by single-token decode steps must produce the
// same final-token output as one prefill covering the whole sequence. That
// equivalence is the entire justification for KV-cached generation; without it
// every generation past the first token is arithmetic the model never asked for.
//
// The defect this pins: every attention layer took its RoPE position offset
// from CacheProvider.SeqLen(), which reports layer 0's cursor. Layer 0 advances
// that cursor with its own cache Update partway through the forward pass, so
// layers 1..N-1 read a value already advanced by the current chunk's length and
// rotated Q/K to positions shifted by +chunkLen. Inside one pass the shift is
// uniform and RoPE is relative, so a pure prefill and a pure token-by-token
// decode each look correct in isolation — which is why this survived so long.
// The damage appears only at the prefill->decode boundary, where a decode query
// shifted by +1 is scored against keys cached with a shift of +promptLen.
//
// The test therefore needs a stack at least two layers deep sharing one cache,
// and a prefill chunk longer than one token.
func TestGQA_PrefillDecodeRoPEPositionParity(t *testing.T) {
	const (
		modelDim = 32
		numQ     = 4
		numKV    = 2
		numTok   = 10
		numLayer = 3
		split    = 7 // prefill this many tokens, then decode the rest one at a time
		maxSeq   = 64
	)

	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	gqa, err := NewGroupedQueryAttention[float32](
		engine, ops, modelDim, numQ, numKV,
		WithMaxSeqLen[float32](maxSeq),
	)
	if err != nil {
		t.Fatalf("construct GQA: %v", err)
	}

	// Pin the weights. With the default random init this stack turned out to be
	// position-blind (the softmax saturated onto one key that the RoPE shift
	// never dislodged), so an earlier draft of this test passed even with the
	// bug present — exactly the vacuous assertion lore L-0009 warns about. The
	// sensitivity control further down is what keeps that from recurring
	// silently if these constants are ever retuned.
	for pi, p := range gqa.Parameters() {
		d := p.Value.Data()
		for i := range d {
			d[i] = float32(math.Sin(float64(i+1)*0.7 + float64(pi)))
		}
	}

	data := make([]float32, numTok*modelDim)
	for i := range data {
		data[i] = float32(math.Cos(float64(i)*1.3)) * 0.8
	}

	chunk := func(from, to int) *tensor.TensorNumeric[float32] {
		t.Helper()
		tt, tErr := tensor.New([]int{1, to - from, modelDim}, data[from*modelDim:to*modelDim])
		if tErr != nil {
			t.Fatal(tErr)
		}
		return tt
	}

	// forwardStack runs one chunk through a numLayer-deep stack of attention
	// layers that all share the cache carried by ctx.
	forwardStack := func(ctx context.Context, in *tensor.TensorNumeric[float32]) *tensor.TensorNumeric[float32] {
		t.Helper()
		h := in
		for layer := range numLayer {
			gqa.LayerIndex = layer
			out, fErr := gqa.Forward(ctx, h)
			if fErr != nil {
				t.Fatalf("layer %d forward: %v", layer, fErr)
			}
			h = out
		}
		return h
	}

	lastToken := func(out *tensor.TensorNumeric[float32]) []float32 {
		t.Helper()
		shape := out.Shape()
		if len(shape) != 3 || shape[2] != modelDim {
			t.Fatalf("unexpected output shape %v", shape)
		}
		row := make([]float32, modelDim)
		copy(row, out.Data()[(shape[1]-1)*modelDim:shape[1]*modelDim])
		return row
	}

	runFresh := func(chunks [][2]int) []float32 {
		t.Helper()
		ctx := generate.WithCache(context.Background(),
			generate.NewKVCache[float32](numLayer, maxSeq))
		var out *tensor.TensorNumeric[float32]
		for _, c := range chunks {
			out = forwardStack(ctx, chunk(c[0], c[1]))
		}
		return lastToken(out)
	}

	maxDiff := func(a, b []float32) float64 {
		m := 0.0
		for i := range a {
			if d := math.Abs(float64(a[i]) - float64(b[i])); d > m {
				m = d
			}
		}
		return m
	}

	// Reference: one prefill over the whole sequence.
	want := runFresh([][2]int{{0, numTok}})

	// Under test: prefill `split` tokens, then decode the rest one at a time.
	chunks := [][2]int{{0, split}}
	for i := split; i < numTok; i++ {
		chunks = append(chunks, [2]int{i, i + 1})
	}
	got := runFresh(chunks)

	// Sensitivity control. If the stack were position-blind (saturated softmax,
	// degenerate weights) every arrangement would agree and the parity
	// assertion below would be worthless. Prove the output actually depends on
	// how much context precedes the final token before trusting the parity
	// check: the same final token attended with no history must differ.
	blind := runFresh([][2]int{{numTok - 1, numTok}})
	const minSensitivity = 1e-2
	if d := maxDiff(want, blind); d < minSensitivity {
		t.Fatalf("test is not sensitive to context (maxdiff %g < %g): the parity "+
			"assertion below cannot fail and must not be trusted", d, minSensitivity)
	}

	const tol = 1e-3
	if d := maxDiff(want, got); d > tol {
		t.Errorf("prefill(%d)+%d decode steps diverges from one-shot prefill(%d): "+
			"maxdiff %g > %g\n one-shot      = %v\n prefill+decode = %v",
			split, numTok-split, numTok, d, tol, want, got)
	}
}
