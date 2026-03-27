package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// mockKVCache satisfies SparseRoutedKVCache for testing.
type mockKVCache struct {
	layers int
	seqLen int
}

func (m *mockKVCache) NumLayers() int { return m.layers }
func (m *mockKVCache) SeqLen() int    { return m.seqLen }
func (m *mockKVCache) Reset()         {}

func makeSparseRoutedAttention(t *testing.T, numHeads, numKVHeads, headDim, segmentSize, topK int) *SparseRoutedAttention[float32] {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
		context.Background(), engine, headDim, 1024,
		embeddings.WithRotaryBase(10000.0),
	)
	if err != nil {
		t.Fatalf("NewRotaryPositionalEmbedding: %v", err)
	}

	kvCache := &mockKVCache{layers: 1, seqLen: 0}

	sra, err := NewSparseRoutedAttention[float32](
		engine, ops, rope, kvCache,
		numHeads, numKVHeads, headDim, segmentSize, topK,
	)
	if err != nil {
		t.Fatalf("NewSparseRoutedAttention: %v", err)
	}
	return sra
}

func TestSparseRoutedAttention_Forward_OutputShape(t *testing.T) {
	ctx := context.Background()
	numHeads, numKVHeads, headDim := 4, 2, 8
	seqQ, seqKV := 4, 16
	batch := 1
	segmentSize := 4
	topK := 2

	sra := makeSparseRoutedAttention(t, numHeads, numKVHeads, headDim, segmentSize, topK)

	Q, K, V := makeQKV(t, batch, numHeads, numKVHeads, seqQ, seqKV, headDim)

	out, err := sra.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	wantShape := []int{batch, numHeads, seqQ, headDim}
	gotShape := out.Shape()
	if len(gotShape) != len(wantShape) {
		t.Fatalf("output rank: got %d, want %d", len(gotShape), len(wantShape))
	}
	for i := range wantShape {
		if gotShape[i] != wantShape[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
		}
	}
}

func TestSparseRoutedAttention_RoutingSelectsSubset(t *testing.T) {
	ctx := context.Background()
	numHeads, numKVHeads, headDim := 2, 2, 4
	seqQ := 1
	seqKV := 12
	batch := 1
	segmentSize := 4
	topK := 2 // selects 2 out of 3 segments

	sra := makeSparseRoutedAttention(t, numHeads, numKVHeads, headDim, segmentSize, topK)

	// Create Q with a specific pattern that should preferentially match
	// certain KV segments over others.
	qData := make([]float32, batch*numHeads*seqQ*headDim)
	for i := range qData {
		qData[i] = 1.0 // uniform query
	}
	Q, err := tensor.New[float32]([]int{batch, numHeads, seqQ, headDim}, qData)
	if err != nil {
		t.Fatalf("create Q: %v", err)
	}

	// Create K with distinct segments: segment 0 has all 1s, segment 1 has
	// all -1s (dissimilar), segment 2 has all 0.5s.
	kSize := batch * numKVHeads * seqKV * headDim
	kData := make([]float32, kSize)
	vData := make([]float32, kSize)
	for kv := range numKVHeads {
		for t := range seqKV {
			seg := t / segmentSize
			for d := range headDim {
				idx := (kv*seqKV + t) * headDim + d
				switch seg {
				case 0:
					kData[idx] = 1.0
				case 1:
					kData[idx] = -1.0
				case 2:
					kData[idx] = 0.5
				}
				vData[idx] = float32(seg + 1) // different values per segment
			}
		}
	}
	K, err := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, kData)
	if err != nil {
		t.Fatalf("create K: %v", err)
	}
	V, err := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, vData)
	if err != nil {
		t.Fatalf("create V: %v", err)
	}

	out, err := sra.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// With topK=2 out of 3 segments, the routing should select segments 0
	// and 2 (most similar to the all-ones query) and skip segment 1
	// (all -1s). The output should be non-zero (valid attention was computed).
	outData := out.Data()
	allZero := true
	for _, v := range outData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("output is all zeros; expected non-zero output from attention")
	}

	// Verify output shape is correct.
	wantShape := []int{batch, numHeads, seqQ, headDim}
	gotShape := out.Shape()
	for i := range wantShape {
		if gotShape[i] != wantShape[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
		}
	}
}

func TestSparseRoutedAttention_TopKClamped(t *testing.T) {
	// topK > numSegments should clamp to numSegments without error.
	ctx := context.Background()
	numHeads, numKVHeads, headDim := 2, 2, 4
	seqQ, seqKV := 2, 4
	batch := 1
	segmentSize := 4 // 1 segment total
	topK := 5        // much larger than numSegments

	sra := makeSparseRoutedAttention(t, numHeads, numKVHeads, headDim, segmentSize, topK)

	Q, K, V := makeQKV(t, batch, numHeads, numKVHeads, seqQ, seqKV, headDim)

	out, err := sra.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	wantShape := []int{batch, numHeads, seqQ, headDim}
	gotShape := out.Shape()
	for i := range wantShape {
		if gotShape[i] != wantShape[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
		}
	}
}

func TestSparseRoutedAttention_OpType(t *testing.T) {
	sra := makeSparseRoutedAttention(t, 4, 2, 8, 4, 2)
	if got := sra.OpType(); got != "SparseRoutedAttention" {
		t.Errorf("OpType: got %q, want %q", got, "SparseRoutedAttention")
	}
}

func TestSparseRoutedAttention_Attributes(t *testing.T) {
	sra := makeSparseRoutedAttention(t, 4, 2, 8, 4, 2)
	attrs := sra.Attributes()

	checks := map[string]int{
		"num_heads":    4,
		"num_kv_heads": 2,
		"head_dim":     8,
		"segment_size": 4,
		"top_k":        2,
	}
	for k, want := range checks {
		got, ok := attrs[k]
		if !ok {
			t.Errorf("missing attribute %q", k)
			continue
		}
		if got.(int) != want {
			t.Errorf("attribute %q: got %v, want %v", k, got, want)
		}
	}
}

func TestSparseRoutedAttention_Parameters(t *testing.T) {
	sra := makeSparseRoutedAttention(t, 4, 2, 8, 4, 2)
	if params := sra.Parameters(); params != nil {
		t.Errorf("expected nil parameters, got %d", len(params))
	}
}

func TestSparseRoutedAttention_Backward_StraightThrough(t *testing.T) {
	sra := makeSparseRoutedAttention(t, 4, 2, 8, 4, 2)

	dOutData := make([]float32, 1*4*2*8)
	for i := range dOutData {
		dOutData[i] = float32(i) * 0.01
	}
	dOut, err := tensor.New[float32]([]int{1, 4, 2, 8}, dOutData)
	if err != nil {
		t.Fatalf("create dOut: %v", err)
	}

	grads, err := sra.Backward(context.Background(), types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(grads) != 3 {
		t.Fatalf("expected 3 gradients, got %d", len(grads))
	}
	if grads[0] != dOut {
		t.Error("expected Q gradient to be dOut (straight-through)")
	}
	if grads[1] != nil {
		t.Error("expected K gradient to be nil")
	}
	if grads[2] != nil {
		t.Error("expected V gradient to be nil")
	}
}

func TestSparseRoutedAttention_OutputShape(t *testing.T) {
	ctx := context.Background()
	sra := makeSparseRoutedAttention(t, 4, 2, 8, 4, 2)

	// Before forward, OutputShape should be nil.
	if s := sra.OutputShape(); s != nil {
		t.Errorf("OutputShape before Forward: got %v, want nil", s)
	}

	Q, K, V := makeQKV(t, 1, 4, 2, 2, 8, 8)
	if _, err := sra.Forward(ctx, Q, K, V); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	wantShape := []int{1, 4, 2, 8}
	gotShape := sra.OutputShape()
	for i := range wantShape {
		if gotShape[i] != wantShape[i] {
			t.Errorf("OutputShape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
		}
	}
}

func TestSparseRoutedAttention_DocumentBoundaries(t *testing.T) {
	ctx := context.Background()
	numHeads, numKVHeads, headDim := 2, 2, 4
	seqQ, seqKV := 4, 8
	batch := 1
	segmentSize := 4
	topK := 2

	sra := makeSparseRoutedAttention(t, numHeads, numKVHeads, headDim, segmentSize, topK)

	Q, K, V := makeQKV(t, batch, numHeads, numKVHeads, seqQ, seqKV, headDim)

	// Run without document boundaries.
	out1, err := sra.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward without boundaries: %v", err)
	}

	// Set document boundaries and run again — should not error.
	sra.SetDocumentBoundaries([]int{0, 4})
	out2, err := sra.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward with boundaries: %v", err)
	}

	// Both should produce valid output with correct shape.
	for _, out := range []*tensor.TensorNumeric[float32]{out1, out2} {
		wantShape := []int{batch, numHeads, seqQ, headDim}
		gotShape := out.Shape()
		for i := range wantShape {
			if gotShape[i] != wantShape[i] {
				t.Errorf("output shape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
			}
		}
	}

	// Clear boundaries.
	sra.SetDocumentBoundaries(nil)
}

func TestSparseRoutedAttention_TooFewInputs(t *testing.T) {
	sra := makeSparseRoutedAttention(t, 4, 2, 8, 4, 2)
	Q, _, _ := makeQKV(t, 1, 4, 2, 2, 8, 8)

	_, err := sra.Forward(context.Background(), Q)
	if err == nil {
		t.Error("expected error for too few inputs")
	}
}

func TestNewSparseRoutedAttention_ValidationErrors(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	rope, _ := embeddings.NewRotaryPositionalEmbedding[float32](
		context.Background(), engine, 8, 1024,
		embeddings.WithRotaryBase(10000.0),
	)
	kvCache := &mockKVCache{layers: 1, seqLen: 0}

	tests := []struct {
		name       string
		numHeads   int
		numKVHeads int
		segSize    int
		topK       int
	}{
		{"zero numHeads", 0, 2, 4, 2},
		{"zero numKVHeads", 4, 0, 4, 2},
		{"heads not divisible", 3, 2, 4, 2},
		{"zero segmentSize", 4, 2, 0, 2},
		{"zero topK", 4, 2, 4, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewSparseRoutedAttention[float32](
				engine, ops, rope, kvCache,
				tt.numHeads, tt.numKVHeads, 8, tt.segSize, tt.topK,
			)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}
