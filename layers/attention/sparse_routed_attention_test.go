package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/training/loss"
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

func TestSparseRoutedAttention_Forward_OutputShape_TableDriven(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name       string
		batch      int
		numHeads   int
		numKVHeads int
		headDim    int
		seqQ       int
		seqKV      int
		segSize    int
		topK       int
	}{
		{"single_batch_GQA", 1, 4, 2, 8, 4, 16, 4, 2},
		{"multi_batch", 2, 4, 2, 8, 2, 12, 4, 2},
		{"MHA_equal_heads", 1, 4, 4, 8, 3, 16, 4, 2},
		{"single_query_pos", 1, 2, 2, 4, 1, 8, 4, 1},
		{"large_headDim", 1, 2, 2, 16, 2, 8, 4, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sra := makeSparseRoutedAttention(t, tt.numHeads, tt.numKVHeads, tt.headDim, tt.segSize, tt.topK)
			Q, K, V := makeQKV(t, tt.batch, tt.numHeads, tt.numKVHeads, tt.seqQ, tt.seqKV, tt.headDim)

			out, err := sra.Forward(ctx, Q, K, V)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			wantShape := []int{tt.batch, tt.numHeads, tt.seqQ, tt.headDim}
			gotShape := out.Shape()
			if len(gotShape) != len(wantShape) {
				t.Fatalf("output rank: got %d, want %d", len(gotShape), len(wantShape))
			}
			for i := range wantShape {
				if gotShape[i] != wantShape[i] {
					t.Errorf("shape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
				}
			}
		})
	}
}

func TestSparseRoutedAttention_DifferentRoutingProducesDifferentOutput(t *testing.T) {
	ctx := context.Background()
	numHeads, numKVHeads, headDim := 2, 2, 4
	seqQ := 1
	seqKV := 12
	batch := 1
	segmentSize := 4
	topK := 1 // only 1 segment selected — routing choice determines output

	sra := makeSparseRoutedAttention(t, numHeads, numKVHeads, headDim, segmentSize, topK)

	kSize := batch * numKVHeads * seqKV * headDim

	// V segments have distinct constant values so output depends on which
	// segment the router selects.
	makeInputs := func(qVal float32) (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) {
		t.Helper()
		qData := make([]float32, batch*numHeads*seqQ*headDim)
		for i := range qData {
			qData[i] = qVal
		}
		kData := make([]float32, kSize)
		vData := make([]float32, kSize)
		for kv := range numKVHeads {
			for tok := range seqKV {
				seg := tok / segmentSize
				for d := range headDim {
					idx := (kv*seqKV + tok) * headDim + d
					switch seg {
					case 0:
						kData[idx] = 1.0
						vData[idx] = 10.0
					case 1:
						kData[idx] = -1.0
						vData[idx] = 20.0
					case 2:
						kData[idx] = 0.0
						vData[idx] = 30.0
					}
				}
			}
		}
		Q, _ := tensor.New[float32]([]int{batch, numHeads, seqQ, headDim}, qData)
		K, _ := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, kData)
		V, _ := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, vData)
		return Q, K, V
	}

	// Query aligned with segment 0 (all 1s).
	Q1, K1, V1 := makeInputs(1.0)
	out1, err := sra.Forward(ctx, Q1, K1, V1)
	if err != nil {
		t.Fatalf("Forward (q=1.0): %v", err)
	}

	// Query aligned with segment 1 (all -1s).
	Q2, K2, V2 := makeInputs(-1.0)
	out2, err := sra.Forward(ctx, Q2, K2, V2)
	if err != nil {
		t.Fatalf("Forward (q=-1.0): %v", err)
	}

	// The two outputs must differ because different segments are routed.
	d1 := out1.Data()
	d2 := out2.Data()
	same := true
	for i := range d1 {
		if d1[i] != d2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different routing queries produced identical output; expected different attention patterns")
	}
}

func TestSparseRoutedAttention_TopKSelectsHighestScores(t *testing.T) {
	// Verify that top-k selects the segments with highest cosine similarity.
	// We construct K so segment 0 is perfectly aligned with Q, segment 1 is
	// orthogonal, and segment 2 is anti-aligned. With topK=1, only segment 0
	// should be attended and the output should be close to V[segment 0].
	ctx := context.Background()
	numHeads, numKVHeads, headDim := 1, 1, 4
	seqQ := 1
	seqKV := 12
	batch := 1
	segmentSize := 4
	topK := 1

	sra := makeSparseRoutedAttention(t, numHeads, numKVHeads, headDim, segmentSize, topK)

	qData := []float32{1, 0, 0, 0}
	kData := make([]float32, seqKV*headDim)
	vData := make([]float32, seqKV*headDim)

	// Segment 0: keys = [1,0,0,0], values = [5,5,5,5]
	// Segment 1: keys = [0,1,0,0], values = [50,50,50,50] (orthogonal)
	// Segment 2: keys = [-1,0,0,0], values = [500,500,500,500] (anti-aligned)
	for tok := 0; tok < 4; tok++ {
		kData[tok*headDim+0] = 1
		vData[tok*headDim+0] = 5
		vData[tok*headDim+1] = 5
		vData[tok*headDim+2] = 5
		vData[tok*headDim+3] = 5
	}
	for tok := 4; tok < 8; tok++ {
		kData[tok*headDim+1] = 1
		vData[tok*headDim+0] = 50
		vData[tok*headDim+1] = 50
		vData[tok*headDim+2] = 50
		vData[tok*headDim+3] = 50
	}
	for tok := 8; tok < 12; tok++ {
		kData[tok*headDim+0] = -1
		vData[tok*headDim+0] = 500
		vData[tok*headDim+1] = 500
		vData[tok*headDim+2] = 500
		vData[tok*headDim+3] = 500
	}

	Q, _ := tensor.New[float32]([]int{batch, numHeads, seqQ, headDim}, qData)
	K, _ := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, kData)
	V, _ := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, vData)

	out, err := sra.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Output should be close to V[segment 0] = [5,5,5,5] because only the
	// highest-scoring segment is selected.
	outData := out.Data()
	for d := range headDim {
		if math.Abs(float64(outData[d]-5.0)) > 1.0 {
			t.Errorf("output[%d] = %f, want ~5.0 (segment 0 values); top-k may not select highest scores", d, outData[d])
		}
	}
}

func TestSparseRoutedAttention_SegmentSizeBoundary(t *testing.T) {
	ctx := context.Background()

	// Test cases where seqKV is not evenly divisible by segmentSize, plus
	// edge cases like seqKV < segmentSize.
	tests := []struct {
		name    string
		seqKV   int
		segSize int
		topK    int
	}{
		{"seqKV_not_divisible", 10, 4, 2},   // 10/4 = 2 segments (last partial)
		{"seqKV_less_than_segSize", 3, 8, 1}, // 1 segment (truncated)
		{"seqKV_equals_segSize", 4, 4, 1},    // exactly 1 segment
		{"single_token_segment", 5, 1, 3},    // 5 segments of 1 token each
	}

	numHeads, numKVHeads, headDim := 2, 2, 4
	seqQ := 2
	batch := 1

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sra := makeSparseRoutedAttention(t, numHeads, numKVHeads, headDim, tt.segSize, tt.topK)
			Q, K, V := makeQKV(t, batch, numHeads, numKVHeads, seqQ, tt.seqKV, headDim)

			out, err := sra.Forward(ctx, Q, K, V)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			wantShape := []int{batch, numHeads, seqQ, headDim}
			gotShape := out.Shape()
			for i := range wantShape {
				if gotShape[i] != wantShape[i] {
					t.Errorf("shape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
				}
			}

			// Output must contain finite, non-NaN values.
			for i, v := range out.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %f; expected finite value", i, v)
					break
				}
			}
		})
	}
}

func TestSparseRoutedAttention_ContrastiveLossIntegration(t *testing.T) {
	// Verify that routing scores produced during SparseRoutedAttention forward
	// can flow into RoutingContrastive loss and produce a valid scalar loss.
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	numHeads, numKVHeads, headDim := 4, 2, 4
	seqQ := 2
	seqKV := 8
	batch := 1
	segmentSize := 4
	topK := 1
	numSegments := seqKV / segmentSize

	sra := makeSparseRoutedAttention(t, numHeads, numKVHeads, headDim, segmentSize, topK)

	Q, K, V := makeQKV(t, batch, numHeads, numKVHeads, seqQ, seqKV, headDim)

	// Run forward to confirm the layer works.
	_, err := sra.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Simulate per-head routing scores as cosine similarities between each
	// query and segment centroids. Shape: [batch, numHeads, numSegments].
	// In a real pipeline, these scores are computed inside Forward and passed
	// to the contrastive loss during training.
	scoresData := make([]float32, batch*numHeads*numSegments)
	for h := range numHeads {
		for s := range numSegments {
			// Give each head a different score distribution so the contrastive
			// loss has non-trivial pairwise diversity to measure.
			scoresData[h*numSegments+s] = float32(h+1) * float32(s+1) * 0.1
		}
	}
	scores, err := tensor.New[float32]([]int{batch, numHeads, numSegments}, scoresData)
	if err != nil {
		t.Fatalf("create routing scores: %v", err)
	}

	// Feed routing scores into RoutingContrastive loss.
	rcLoss := loss.NewRoutingContrastive[float32](engine, ops, 0.01)
	lossOut, err := rcLoss.Forward(ctx, scores)
	if err != nil {
		t.Fatalf("RoutingContrastive.Forward: %v", err)
	}

	// Loss must be a scalar [1] tensor with a finite value.
	lossShape := lossOut.Shape()
	if len(lossShape) != 1 || lossShape[0] != 1 {
		t.Fatalf("loss shape: got %v, want [1]", lossShape)
	}
	lossVal := lossOut.Data()[0]
	if math.IsNaN(float64(lossVal)) || math.IsInf(float64(lossVal), 0) {
		t.Errorf("loss value is %f; expected finite scalar", lossVal)
	}

	// Backward should produce gradients with the same shape as the routing scores.
	grads, err := rcLoss.Backward(ctx, types.FullBackprop, lossOut)
	if err != nil {
		t.Fatalf("RoutingContrastive.Backward: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient tensor, got %d", len(grads))
	}
	gradShape := grads[0].Shape()
	wantGradShape := []int{batch, numHeads, numSegments}
	for i := range wantGradShape {
		if gradShape[i] != wantGradShape[i] {
			t.Errorf("grad shape[%d]: got %d, want %d", i, gradShape[i], wantGradShape[i])
		}
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
