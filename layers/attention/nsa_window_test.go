package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestNSAWindowAttention_Forward_FullWindow(t *testing.T) {
	// When the window covers the entire sequence, the output should match
	// standard causal attention (no positions are masked by the window).
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batch, heads, seq, dim := 1, 2, 4, 8
	windowSize := seq // window covers full sequence

	nw, err := NewNSAWindowAttention[float32](engine, ops, windowSize, heads, heads, dim)
	if err != nil {
		t.Fatalf("NewNSAWindowAttention: %v", err)
	}

	q := makeSeqTensor(t, batch, heads, seq, dim, 0.01)
	k := makeSeqTensor(t, batch, heads, seq, dim, 0.02)
	v := makeSeqTensor(t, batch, heads, seq, dim, 0.03)

	// NSA window result with full window.
	windowOut, err := nw.Forward(ctx, q, k, v)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Compare against SDPA with causal mask (no window restriction).
	sdpa := NewScaledDotProductAttention[float32](engine, dim)
	causalMask := buildCausalMask(t, ops, seq, seq)
	q3d := reshape3D(t, engine, ctx, q, batch*heads, seq, dim)
	k3d := reshape3D(t, engine, ctx, k, batch*heads, seq, dim)
	v3d := reshape3D(t, engine, ctx, v, batch*heads, seq, dim)
	sdpaOut, err := sdpa.Forward(ctx, q3d, k3d, v3d, causalMask)
	if err != nil {
		t.Fatalf("SDPA Forward: %v", err)
	}

	wData := windowOut.Data()
	sData := sdpaOut.Data()
	if len(wData) != len(sData) {
		t.Fatalf("output size mismatch: window=%d, sdpa=%d", len(wData), len(sData))
	}

	for i := range wData {
		if diff := math.Abs(float64(wData[i] - sData[i])); diff > 1e-5 {
			t.Errorf("mismatch at index %d: window=%f, sdpa=%f, diff=%e", i, wData[i], sData[i], diff)
		}
	}
}

func TestNSAWindowAttention_Forward_SmallWindow(t *testing.T) {
	// With a small window, tokens outside the window should not contribute.
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batch, heads, seq, dim := 1, 1, 6, 4
	windowSize := 2 // each query attends to at most 2 past tokens

	nw, err := NewNSAWindowAttention[float32](engine, ops, windowSize, heads, heads, dim)
	if err != nil {
		t.Fatalf("NewNSAWindowAttention: %v", err)
	}

	q := makeSeqTensor(t, batch, heads, seq, dim, 0.1)
	k := makeSeqTensor(t, batch, heads, seq, dim, 0.1)
	v := makeOneShotV(t, batch, heads, seq, dim)

	out, err := nw.Forward(ctx, q, k, v)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// V is set up so each position j has v[j] = 1 at index j%dim.
	// For query at position 5 with windowSize=2, only positions 4 and 5
	// are in the window. So the output should have non-zero weight only
	// on dimensions 4%dim=0 and 5%dim=1. Positions 0-3 should contribute
	// approximately zero weight.
	outData := out.Data()
	// Check that the output at position 5 exists and is reasonable.
	outShape := out.Shape()
	if len(outShape) != 4 {
		t.Fatalf("expected 4D output, got %v", outShape)
	}
	if outShape[2] != seq {
		t.Fatalf("expected seqQ=%d, got %d", seq, outShape[2])
	}

	// For the last position (q=5), the window covers positions [4, 5].
	// Verify that positions outside window contribute nothing by checking
	// the mask builds correctly.
	mask, err := nw.buildWindowMask(seq, seq)
	if err != nil {
		t.Fatalf("buildWindowMask: %v", err)
	}
	maskData := mask.Data()

	// Position q=5, j=0 should be masked (outside window).
	idx := 5*seq + 0
	if maskData[idx] >= 0 {
		t.Errorf("expected position (5,0) to be masked, got %f", maskData[idx])
	}
	// Position q=5, j=3 should be masked (outside window).
	idx = 5*seq + 3
	if maskData[idx] >= 0 {
		t.Errorf("expected position (5,3) to be masked, got %f", maskData[idx])
	}
	// Position q=5, j=4 should be unmasked.
	idx = 5*seq + 4
	if maskData[idx] != 0 {
		t.Errorf("expected position (5,4) to be unmasked, got %f", maskData[idx])
	}
	// Position q=5, j=5 should be unmasked.
	idx = 5*seq + 5
	if maskData[idx] != 0 {
		t.Errorf("expected position (5,5) to be unmasked, got %f", maskData[idx])
	}

	// Also verify that output is not all zeros (the forward pass produced values).
	allZero := true
	for _, v := range outData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("expected non-zero output from forward pass")
	}
}

func TestNSAWindowAttention_GQA(t *testing.T) {
	// Test with grouped query attention (numHeads > numKVHeads).
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batch, heads, kvHeads, seq, dim := 1, 4, 2, 4, 8
	windowSize := 3

	nw, err := NewNSAWindowAttention[float32](engine, ops, windowSize, heads, kvHeads, dim)
	if err != nil {
		t.Fatalf("NewNSAWindowAttention: %v", err)
	}

	q := makeSeqTensor(t, batch, heads, seq, dim, 0.01)
	k := makeSeqTensor(t, batch, kvHeads, seq, dim, 0.02)
	v := makeSeqTensor(t, batch, kvHeads, seq, dim, 0.03)

	out, err := nw.Forward(ctx, q, k, v)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	want := []int{batch, heads, seq, dim}
	got := out.Shape()
	if !intSliceEq(got, want) {
		t.Errorf("output shape: got %v, want %v", got, want)
	}
}

func TestNSAWindowAttention_InvalidArgs(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name                              string
		windowSize, heads, kvHeads, hdim int
	}{
		{"zero window", 0, 4, 4, 8},
		{"negative window", -1, 4, 4, 8},
		{"zero heads", 3, 0, 4, 8},
		{"zero kv heads", 3, 4, 0, 8},
		{"heads not divisible", 3, 5, 3, 8},
		{"zero head dim", 3, 4, 4, 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewNSAWindowAttention[float32](engine, ops, tc.windowSize, tc.heads, tc.kvHeads, tc.hdim)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestNSAWindowAttention_WindowMask(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	nw, err := NewNSAWindowAttention[float32](engine, ops, 3, 2, 2, 4)
	if err != nil {
		t.Fatalf("NewNSAWindowAttention: %v", err)
	}

	mask, err := nw.buildWindowMask(5, 5)
	if err != nil {
		t.Fatalf("buildWindowMask: %v", err)
	}

	data := mask.Data()
	seqLen := 5

	tests := []struct {
		q, k     int
		wantZero bool // true = unmasked (0), false = masked (-1e9)
	}{
		// q=0: window is [0, 0]
		{0, 0, true},
		{0, 1, false}, // future position (causal)
		// q=1: window is [0, 1]
		{1, 0, true},
		{1, 1, true},
		{1, 2, false},
		// q=2: window is [0, 2]
		{2, 0, true},
		{2, 1, true},
		{2, 2, true},
		{2, 3, false},
		// q=3: window is [1, 3] (windowSize=3 means positions 1,2,3)
		{3, 0, false}, // outside window
		{3, 1, true},
		{3, 2, true},
		{3, 3, true},
		{3, 4, false},
		// q=4: window is [2, 4]
		{4, 0, false},
		{4, 1, false},
		{4, 2, true},
		{4, 3, true},
		{4, 4, true},
	}

	for _, tc := range tests {
		idx := tc.q*seqLen + tc.k
		val := data[idx]
		if tc.wantZero && val != 0 {
			t.Errorf("mask[%d,%d] = %f, want 0", tc.q, tc.k, val)
		}
		if !tc.wantZero && val >= 0 {
			t.Errorf("mask[%d,%d] = %f, want < 0 (masked)", tc.q, tc.k, val)
		}
	}
}

func TestNSAWindowAttention_Scale(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	headDim := 64
	nw, err := NewNSAWindowAttention[float32](engine, ops, 3, 2, 2, headDim)
	if err != nil {
		t.Fatalf("NewNSAWindowAttention: %v", err)
	}

	want := 1.0 / math.Sqrt(float64(headDim))
	got := nw.Scale()
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("Scale() = %f, want %f", got, want)
	}
}

// --- helpers ---

func makeSeqTensor(t *testing.T, dims ...interface{}) *tensor.TensorNumeric[float32] {
	t.Helper()
	// Accept (t, d0, d1, d2, d3, scale) where scale is float64
	if len(dims) != 5 {
		t.Fatalf("makeSeqTensor: expected 5 args (4 dims + scale), got %d", len(dims))
	}
	d0 := dims[0].(int)
	d1 := dims[1].(int)
	d2 := dims[2].(int)
	d3 := dims[3].(int)
	scale := dims[4].(float64)

	size := d0 * d1 * d2 * d3
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(float64(i) * scale)
	}
	out, err := tensor.New[float32]([]int{d0, d1, d2, d3}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	return out
}

func makeOneShotV(t *testing.T, batch, heads, seq, dim int) *tensor.TensorNumeric[float32] {
	t.Helper()
	size := batch * heads * seq * dim
	data := make([]float32, size)
	for b := range batch {
		for h := range heads {
			for s := range seq {
				idx := ((b*heads+h)*seq+s)*dim + (s % dim)
				data[idx] = 1.0
			}
		}
	}
	out, err := tensor.New[float32]([]int{batch, heads, seq, dim}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	return out
}

func buildCausalMask(t *testing.T, ops numeric.Float32Ops, seqQ, seqKV int) *tensor.TensorNumeric[float32] {
	t.Helper()
	data := make([]float32, seqQ*seqKV)
	largeNeg := ops.FromFloat64(-1e9)
	offset := seqKV - seqQ
	for i := range seqQ {
		for j := range seqKV {
			if j <= i+offset {
				data[i*seqKV+j] = 0
			} else {
				data[i*seqKV+j] = largeNeg
			}
		}
	}
	mask, err := tensor.New[float32]([]int{1, 1, seqQ, seqKV}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	return mask
}

func reshape3D(t *testing.T, engine compute.Engine[float32], ctx context.Context, x *tensor.TensorNumeric[float32], d0, d1, d2 int) *tensor.TensorNumeric[float32] {
	t.Helper()
	out, err := engine.Reshape(ctx, x, []int{d0, d1, d2})
	if err != nil {
		t.Fatalf("Reshape: %v", err)
	}
	return out
}

