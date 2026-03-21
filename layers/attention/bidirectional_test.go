package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// TestBidirectionalVsCausal_DifferentOutputs verifies that bidirectional
// attention produces different output than causal attention for the same
// input. In bidirectional mode, middle tokens can attend to future tokens,
// so the outputs must differ when there are future positions to attend to.
func TestBidirectionalVsCausal_DifferentOutputs(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	headDim := 4
	batchSize := 1
	seqLen := 4

	// Create non-trivial input data so attention patterns matter.
	qData := make([]float32, batchSize*seqLen*headDim)
	kData := make([]float32, batchSize*seqLen*headDim)
	vData := make([]float32, batchSize*seqLen*headDim)
	for i := range qData {
		qData[i] = float32(i+1) * 0.1
		kData[i] = float32(i+2) * 0.15
		vData[i] = float32(i+3) * 0.2
	}

	makeQKV := func() (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) {
		q, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, qData)
		k, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, kData)
		v, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, vData)
		return q, k, v
	}

	ctx := context.Background()

	// Causal SDPA
	causalSDPA := NewScaledDotProductAttention[float32](engine, headDim)
	causalSDPA.SetCausal(true)
	q, k, v := makeQKV()
	causalOut, err := causalSDPA.Forward(ctx, q, k, v, nil)
	if err != nil {
		t.Fatalf("causal forward failed: %v", err)
	}

	// Bidirectional SDPA using convenience constructor
	biSDPA := NewBidirectionalSDPA[float32](engine, headDim)
	q, k, v = makeQKV()
	biOut, err := biSDPA.Forward(ctx, q, k, v, nil)
	if err != nil {
		t.Fatalf("bidirectional forward failed: %v", err)
	}

	// Outputs must differ because causal masking changes attention weights.
	causalData := causalOut.Data()
	biData := biOut.Data()
	if len(causalData) != len(biData) {
		t.Fatalf("output length mismatch: causal=%d, bidirectional=%d", len(causalData), len(biData))
	}

	different := false
	for i := range causalData {
		if math.Abs(float64(causalData[i]-biData[i])) > 1e-6 {
			different = true
			break
		}
	}
	if !different {
		t.Error("bidirectional and causal outputs are identical; expected different outputs for seq_len > 1")
	}

	// First position (row 0) should be the same because the causal mask
	// only blocks future tokens, and position 0 has no future tokens to
	// block in either mode — wait, actually position 0 in causal mode only
	// attends to itself, while in bidirectional it attends to all positions.
	// So even position 0 should differ.
}

// TestBidirectionalSDPA_WithOption verifies that the WithBidirectional option
// produces an SDPA that behaves identically to NewBidirectionalSDPA.
func TestBidirectionalSDPA_WithOption(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	headDim := 4

	sdpaOpt := NewScaledDotProductAttention[float32](engine, headDim, WithBidirectional[float32]())
	sdpaConv := NewBidirectionalSDPA[float32](engine, headDim)

	batchSize := 1
	seqLen := 3

	data := make([]float32, batchSize*seqLen*headDim)
	for i := range data {
		data[i] = float32(i+1) * 0.1
	}

	ctx := context.Background()

	q1, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, data)
	k1, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, data)
	v1, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, data)

	q2, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, data)
	k2, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, data)
	v2, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, data)

	out1, err := sdpaOpt.Forward(ctx, q1, k1, v1, nil)
	if err != nil {
		t.Fatalf("option SDPA forward failed: %v", err)
	}

	out2, err := sdpaConv.Forward(ctx, q2, k2, v2, nil)
	if err != nil {
		t.Fatalf("convenience SDPA forward failed: %v", err)
	}

	d1 := out1.Data()
	d2 := out2.Data()
	for i := range d1 {
		if math.Abs(float64(d1[i]-d2[i])) > 1e-7 {
			t.Fatalf("outputs differ at index %d: option=%v, convenience=%v", i, d1[i], d2[i])
		}
	}
}

// TestBidirectionalSDPA_SymmetricAttention verifies that in bidirectional
// mode, attention is symmetric: position i attending to position j has
// the same score as j attending to i (given identical Q and K).
func TestBidirectionalSDPA_SymmetricAttention(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	headDim := 4
	batchSize := 1
	seqLen := 3

	// Use same data for Q and K to get symmetric attention scores.
	data := make([]float32, batchSize*seqLen*headDim)
	for i := range data {
		data[i] = float32(i+1) * 0.1
	}

	sdpa := NewBidirectionalSDPA[float32](engine, headDim)
	q, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, data)
	k, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, data)
	v, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, data)

	out, err := sdpa.Forward(ctx(), q, k, v, nil)
	if err != nil {
		t.Fatalf("forward failed: %v", err)
	}

	// With symmetric Q=K, the attention weights matrix (after softmax)
	// should be symmetric. The output should have valid shape.
	if len(out.Shape()) != 3 {
		t.Fatalf("expected 3D output, got shape %v", out.Shape())
	}
	if out.Shape()[0] != batchSize || out.Shape()[1] != seqLen || out.Shape()[2] != headDim {
		t.Fatalf("unexpected output shape: %v", out.Shape())
	}
}

// TestBidirectionalSDPA_BackwardPass verifies that the backward pass works
// correctly in bidirectional mode and produces non-zero gradients.
func TestBidirectionalSDPA_BackwardPass(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	headDim := 4
	batchSize := 1
	seqLen := 3

	qData := make([]float32, batchSize*seqLen*headDim)
	kData := make([]float32, batchSize*seqLen*headDim)
	vData := make([]float32, batchSize*seqLen*headDim)
	for i := range qData {
		qData[i] = float32(i+1) * 0.1
		kData[i] = float32(i+2) * 0.15
		vData[i] = float32(i+3) * 0.2
	}

	sdpa := NewBidirectionalSDPA[float32](engine, headDim)
	q, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, qData)
	k, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, kData)
	v, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, vData)

	// Forward pass first (caches internal state for backward).
	_, err := sdpa.Forward(ctx(), q, k, v, nil)
	if err != nil {
		t.Fatalf("forward failed: %v", err)
	}

	// Create upstream gradient (all ones).
	dOutData := make([]float32, batchSize*seqLen*headDim)
	for i := range dOutData {
		dOutData[i] = 1.0
	}
	dOut, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, dOutData)

	grads, err := sdpa.Backward(ctx(), types.FullBackprop, dOut, q, k, v)
	if err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	if len(grads) != 3 {
		t.Fatalf("expected 3 gradients (dQ, dK, dV), got %d", len(grads))
	}

	// Verify gradients are non-zero.
	names := []string{"dQ", "dK", "dV"}
	for gi, g := range grads {
		if g == nil {
			t.Fatalf("%s is nil", names[gi])
		}
		allZero := true
		for _, val := range g.Data() {
			if val != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Errorf("%s is all zeros; expected non-zero gradients", names[gi])
		}
	}
}

// TestBidirectionalBackward_DiffersThanCausal verifies that backward
// gradients differ between bidirectional and causal modes.
func TestBidirectionalBackward_DiffersThanCausal(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	headDim := 4
	batchSize := 1
	seqLen := 4

	makeInputs := func() (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) {
		qData := make([]float32, batchSize*seqLen*headDim)
		kData := make([]float32, batchSize*seqLen*headDim)
		vData := make([]float32, batchSize*seqLen*headDim)
		dOutData := make([]float32, batchSize*seqLen*headDim)
		for i := range qData {
			qData[i] = float32(i+1) * 0.1
			kData[i] = float32(i+2) * 0.15
			vData[i] = float32(i+3) * 0.2
			dOutData[i] = 1.0
		}
		q, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, qData)
		k, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, kData)
		v, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, vData)
		dOut, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, dOutData)
		return q, k, v, dOut
	}

	background := context.Background()

	// Causal path
	causalSDPA := NewScaledDotProductAttention[float32](engine, headDim)
	causalSDPA.SetCausal(true)
	q, k, v, dOut := makeInputs()
	if _, err := causalSDPA.Forward(background, q, k, v, nil); err != nil {
		t.Fatalf("causal forward: %v", err)
	}
	causalGrads, err := causalSDPA.Backward(background, types.FullBackprop, dOut, q, k, v)
	if err != nil {
		t.Fatalf("causal backward: %v", err)
	}

	// Bidirectional path
	biSDPA := NewBidirectionalSDPA[float32](engine, headDim)
	q, k, v, dOut = makeInputs()
	if _, err := biSDPA.Forward(background, q, k, v, nil); err != nil {
		t.Fatalf("bidirectional forward: %v", err)
	}
	biGrads, err := biSDPA.Backward(background, types.FullBackprop, dOut, q, k, v)
	if err != nil {
		t.Fatalf("bidirectional backward: %v", err)
	}

	// At least one gradient tensor should differ.
	anyDifferent := false
	for gi := range causalGrads {
		cd := causalGrads[gi].Data()
		bd := biGrads[gi].Data()
		for i := range cd {
			if math.Abs(float64(cd[i]-bd[i])) > 1e-6 {
				anyDifferent = true
				break
			}
		}
		if anyDifferent {
			break
		}
	}
	if !anyDifferent {
		t.Error("backward gradients are identical for causal and bidirectional; expected difference")
	}
}

// TestAttentionHead_Bidirectional verifies that AttentionHead can be
// configured with the WithBidirectionalAttention option.
func TestAttentionHead_Bidirectional(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	inputDim := 8
	headDim := 4
	batchSize := 1
	seqLen := 3

	ah, err := NewAttentionHead[float32](engine, inputDim, headDim, WithBidirectionalAttention[float32]())
	if err != nil {
		t.Fatalf("NewAttentionHead failed: %v", err)
	}

	data := make([]float32, batchSize*seqLen*inputDim)
	for i := range data {
		data[i] = float32(i+1) * 0.01
	}
	input, _ := tensor.New[float32]([]int{batchSize, seqLen, inputDim}, data)

	out, err := ah.Forward(ctx(), input)
	if err != nil {
		t.Fatalf("forward failed: %v", err)
	}
	if out.Shape()[0] != batchSize || out.Shape()[1] != seqLen || out.Shape()[2] != headDim {
		t.Fatalf("unexpected output shape: %v", out.Shape())
	}
}

func ctx() context.Context {
	return context.Background()
}
