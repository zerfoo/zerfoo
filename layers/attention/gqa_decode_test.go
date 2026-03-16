package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// repeatCountingEngine wraps a CPU engine and counts calls to Repeat.
// This detects the Repeat-on-maxSeqLen regression where GQA decode
// created 128 MB temporaries per token by expanding KV heads via Repeat
// across the full cached sequence length.
type repeatCountingEngine struct {
	compute.Engine[float32]
	repeatCalls int
	repeatElems int // total elements passed to Repeat
}

func (e *repeatCountingEngine) Repeat(ctx context.Context, a *tensor.TensorNumeric[float32], axis, count int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	e.repeatCalls++
	elems := 1
	for _, d := range a.Shape() {
		elems *= d
	}
	e.repeatElems += elems * count
	return e.Engine.Repeat(ctx, a, axis, count, dst...)
}

// TestGQA_DecodePathSelection verifies that the cuBLAS SDPA decode path
// handles head replication correctly: MHA (numQ == numKV) needs no Repeat,
// GQA needs Repeat for K and V, and MQA uses broadcast. It also verifies
// that Repeat does not operate on the full cached sequence dimension.
func TestGQA_DecodePathSelection(t *testing.T) {
	modelDim := 16
	headDim := 4

	tests := []struct {
		name              string
		numQ              int
		numKV             int
		wantRepeatCalls   int  // exact count, 0 for non-GQA
		expectRepeatOnKV  bool // whether Repeat is expected on KV heads
	}{
		{
			name:             "MHA_no_repeat",
			numQ:             4,
			numKV:            4,
			wantRepeatCalls:  0,
			expectRepeatOnKV: false,
		},
		{
			name:             "GQA_4q_2kv_needs_repeat",
			numQ:             4,
			numKV:            2,
			wantRepeatCalls:  2, // one for K, one for V
			expectRepeatOnKV: true,
		},
		{
			name:             "GQA_4q_1kv_no_repeat_mqa",
			numQ:             4,
			numKV:            1,
			wantRepeatCalls:  0, // MQA path uses numKVHeads==1 broadcast
			expectRepeatOnKV: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			base := compute.NewCPUEngine(numeric.Float32Ops{})
			eng := &repeatCountingEngine{Engine: base}

			gqa, err := NewGroupedQueryAttention[float32](
				eng, numeric.Float32Ops{}, modelDim, tc.numQ, tc.numKV,
				WithMaxSeqLen[float32](128),
			)
			if err != nil {
				t.Fatalf("construct GQA: %v", err)
			}
			gqa.LayerIndex = 0

			cache := generate.NewKVCache[float32](1, 128)
			ctx := generate.WithKVCache(context.Background(), cache)

			// Prefill 4 tokens to populate the cache.
			prefillData := make([]float32, 4*modelDim)
			for i := range prefillData {
				prefillData[i] = float32(i%7) * 0.1
			}
			prefill, tensorErr := tensor.New([]int{1, 4, modelDim}, prefillData)
			if tensorErr != nil {
				t.Fatal(tensorErr)
			}
			if _, err := gqa.Forward(ctx, prefill); err != nil {
				t.Fatalf("prefill: %v", err)
			}

			// Reset counters before decode step.
			eng.repeatCalls = 0
			eng.repeatElems = 0

			// Decode one token.
			decodeData := make([]float32, modelDim)
			for i := range decodeData {
				decodeData[i] = float32(i%5) * 0.2
			}
			decodeInput, tensorErr := tensor.New([]int{1, 1, modelDim}, decodeData)
			if tensorErr != nil {
				t.Fatal(tensorErr)
			}
			out, err := gqa.Forward(ctx, decodeInput)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}

			// Verify output shape.
			if s := out.Shape(); s[0] != 1 || s[1] != 1 || s[2] != modelDim {
				t.Errorf("output shape = %v, want [1, 1, %d]", s, modelDim)
			}

			// Verify output is finite.
			for i, v := range out.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %v (not finite)", i, v)
				}
			}

			// Verify Repeat call count.
			if eng.repeatCalls != tc.wantRepeatCalls {
				t.Errorf("Repeat calls = %d, want %d", eng.repeatCalls, tc.wantRepeatCalls)
			}

			// Regression guard: if Repeat is called during GQA decode,
			// verify the total elements are bounded. The regression
			// created 128 MB temporaries (maxSeqLen * numHeads * headDim
			// * repFactor). With 5 cached tokens, the Repeat should
			// operate on ~5 * headDim * numKV elements, not maxSeqLen.
			if tc.expectRepeatOnKV {
				cachedSeq := 5 // 4 prefill + 1 decode
				maxReasonableElems := cachedSeq * headDim * tc.numKV * 2 * (tc.numQ / tc.numKV) // K + V
				if eng.repeatElems > maxReasonableElems*2 {
					t.Errorf("Repeat element count = %d, exceeds reasonable bound %d (possible maxSeqLen regression)",
						eng.repeatElems, maxReasonableElems*2)
				}
			}
		})
	}
}

// TestGQA_DecodeGuardPresent verifies that GQA decode with large KV caches
// does not trigger excessive memory allocation via Repeat. This is a
// regression guard against Repeat operating on maxSeqLen-sized buffers.
func TestGQA_DecodeGuardPresent(t *testing.T) {
	base := compute.NewCPUEngine(numeric.Float32Ops{})
	eng := &repeatCountingEngine{Engine: base}

	modelDim := 16
	numQ := 8
	numKV := 2

	gqa, err := NewGroupedQueryAttention[float32](
		eng, numeric.Float32Ops{}, modelDim, numQ, numKV,
		WithMaxSeqLen[float32](2048),
	)
	if err != nil {
		t.Fatalf("construct GQA: %v", err)
	}
	gqa.LayerIndex = 0

	cache := generate.NewKVCache[float32](1, 2048)
	ctx := generate.WithKVCache(context.Background(), cache)

	// Prefill 100 tokens.
	prefillData := make([]float32, 100*modelDim)
	for i := range prefillData {
		prefillData[i] = float32(i%11) * 0.05
	}
	prefill, tensorErr := tensor.New([]int{1, 100, modelDim}, prefillData)
	if tensorErr != nil {
		t.Fatal(tensorErr)
	}
	if _, err := gqa.Forward(ctx, prefill); err != nil {
		t.Fatalf("prefill: %v", err)
	}

	eng.repeatCalls = 0
	eng.repeatElems = 0

	// Decode step.
	decodeData := make([]float32, modelDim)
	decodeInput, tensorErr := tensor.New([]int{1, 1, modelDim}, decodeData)
	if tensorErr != nil {
		t.Fatal(tensorErr)
	}
	if _, err := gqa.Forward(ctx, decodeInput); err != nil {
		t.Fatalf("decode: %v", err)
	}

	// With 101 cached tokens and 8Q/2KV, a Repeat on the full cached
	// KV would create ~101 * 2 * 2 * 4 * headDim elements per call.
	// The allocation should be bounded: < 1 MB for a single decode.
	// 1 MB / 4 bytes per float32 = 262144 elements.
	maxElements := 262144
	if eng.repeatElems > maxElements {
		t.Errorf("Repeat allocated %d elements during decode (> %d), possible maxSeqLen regression",
			eng.repeatElems, maxElements)
	}
}
