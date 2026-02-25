package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestScaledDotProductAttention_ForwardWithMask(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	headDim := 4
	sdpa := NewScaledDotProductAttention[float32](engine, headDim)

	batchSize := 2
	numHeads := 2
	seqLen := 3

	// Q, K, V shape: (batchSize*numHeads, seqLen, headDim) = (4, 3, 4)
	q, _ := tensor.New[float32]([]int{batchSize * numHeads, seqLen, headDim}, nil)
	k, _ := tensor.New[float32]([]int{batchSize * numHeads, seqLen, headDim}, nil)
	v, _ := tensor.New[float32]([]int{batchSize * numHeads, seqLen, headDim}, nil)

	for i := range q.Data() {
		q.Data()[i] = float32(i%5+1) * 0.1
	}
	for i := range k.Data() {
		k.Data()[i] = float32(i%3+1) * 0.1
	}
	for i := range v.Data() {
		v.Data()[i] = float32(i%7+1) * 0.1
	}

	// Mask shape: (batchSize, numHeads, seqLen, seqLen) = (2, 2, 3, 3)
	mask, _ := tensor.New[float32]([]int{batchSize, numHeads, seqLen, seqLen}, nil)
	// Causal mask: set upper-triangular to large negative
	for b := range batchSize {
		for h := range numHeads {
			for i := range seqLen {
				for j := range seqLen {
					idx := b*numHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
					if j > i {
						mask.Data()[idx] = -1e9
					}
				}
			}
		}
	}

	out, err := sdpa.Forward(context.Background(), q, k, v, mask)
	if err != nil {
		t.Fatalf("Forward with mask failed: %v", err)
	}

	expectedShape := []int{batchSize * numHeads, seqLen, headDim}
	for i := range 3 {
		if out.Shape()[i] != expectedShape[i] {
			t.Errorf("output shape[%d] = %d, want %d", i, out.Shape()[i], expectedShape[i])
		}
	}

	// Verify output is finite
	for i, v := range out.Data() {
		if v != v { // NaN check
			t.Fatalf("output contains NaN at index %d", i)
		}
	}
}

func TestScaledDotProductAttention_BackwardFullFlow(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	headDim := 4
	sdpa := NewScaledDotProductAttention[float32](engine, headDim)

	batchSize := 1
	seqLen := 3

	q, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, nil)
	k, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, nil)
	v, _ := tensor.New[float32]([]int{batchSize, seqLen, headDim}, nil)

	for i := range q.Data() {
		q.Data()[i] = float32(i%5+1) * 0.1
	}
	for i := range k.Data() {
		k.Data()[i] = float32(i%3+1) * 0.1
	}
	for i := range v.Data() {
		v.Data()[i] = float32(i%7+1) * 0.1
	}

	// Forward first to populate caches
	out, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Backward
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	grads, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	if len(grads) != 3 {
		t.Fatalf("expected 3 gradients, got %d", len(grads))
	}

	// Check gradient shapes
	if grads[0].Shape()[2] != headDim {
		t.Errorf("dQ shape mismatch: got %v", grads[0].Shape())
	}
	if grads[1].Shape()[2] != headDim {
		t.Errorf("dK shape mismatch: got %v", grads[1].Shape())
	}
	if grads[2].Shape()[2] != headDim {
		t.Errorf("dV shape mismatch: got %v", grads[2].Shape())
	}

	// Verify all gradients are finite
	for gi, g := range grads {
		for i, val := range g.Data() {
			if val != val {
				t.Fatalf("gradient[%d][%d] is NaN", gi, i)
			}
		}
	}
}

func TestScaledDotProductAttention_ForwardWithOptions(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Test that options are accepted (even if no options defined yet)
	opt := func(o *ScaledDotProductAttentionOptions[float32]) {
		// noop option to exercise the loop
	}
	sdpa := NewScaledDotProductAttention(engine, 4, opt)
	if sdpa == nil {
		t.Fatal("expected non-nil SDPA with option")
	}
}

func makeSDPATensors(batch, seq, headDim int) (q, k, v *tensor.TensorNumeric[float32]) {
	q, _ = tensor.New[float32]([]int{batch, seq, headDim}, nil)
	k, _ = tensor.New[float32]([]int{batch, seq, headDim}, nil)
	v, _ = tensor.New[float32]([]int{batch, seq, headDim}, nil)
	for i := range q.Data() {
		q.Data()[i] = float32(i%5+1) * 0.1
	}
	for i := range k.Data() {
		k.Data()[i] = float32(i%3+1) * 0.1
	}
	for i := range v.Data() {
		v.Data()[i] = float32(i%7+1) * 0.1
	}
	return
}

func TestSDPA_Forward_TransposeError(t *testing.T) {
	fe := newFailingEngine(map[string]int{"Transpose": 1})
	sdpa := NewScaledDotProductAttention[float32](fe, 4)
	q, k, v := makeSDPATensors(1, 3, 4)
	_, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err == nil {
		t.Error("expected error from Transpose failure")
	}
}

func TestSDPA_Forward_MatMulError(t *testing.T) {
	fe := newFailingEngine(map[string]int{"MatMul": 1})
	sdpa := NewScaledDotProductAttention[float32](fe, 4)
	q, k, v := makeSDPATensors(1, 3, 4)
	_, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err == nil {
		t.Error("expected error from MatMul failure")
	}
}

func TestSDPA_Forward_MulScalarError(t *testing.T) {
	fe := newFailingEngine(map[string]int{"MulScalar": 1})
	sdpa := NewScaledDotProductAttention[float32](fe, 4)
	q, k, v := makeSDPATensors(1, 3, 4)
	_, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err == nil {
		t.Error("expected error from MulScalar failure")
	}
}

func TestSDPA_Forward_SoftmaxError(t *testing.T) {
	fe := newFailingEngine(map[string]int{"Softmax": 1})
	sdpa := NewScaledDotProductAttention[float32](fe, 4)
	q, k, v := makeSDPATensors(1, 3, 4)
	_, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err == nil {
		t.Error("expected error from Softmax failure")
	}
}

func TestSDPA_Forward_FinalMatMulError(t *testing.T) {
	fe := newFailingEngine(map[string]int{"MatMul": 2}) // 2nd MatMul (weights*V)
	sdpa := NewScaledDotProductAttention[float32](fe, 4)
	q, k, v := makeSDPATensors(1, 3, 4)
	_, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err == nil {
		t.Error("expected error from final MatMul failure")
	}
}

func TestSDPA_Forward_MaskReshapeError(t *testing.T) {
	fe := newFailingEngine(map[string]int{"Reshape": 1})
	sdpa := NewScaledDotProductAttention[float32](fe, 4)
	q, k, v := makeSDPATensors(2, 3, 4)
	mask, _ := tensor.New[float32]([]int{1, 2, 3, 3}, nil)
	_, err := sdpa.Forward(context.Background(), q, k, v, mask)
	if err == nil {
		t.Error("expected error from mask Reshape failure")
	}
}

func TestSDPA_Forward_MaskAddError(t *testing.T) {
	fe := newFailingEngine(map[string]int{"Add": 1})
	sdpa := NewScaledDotProductAttention[float32](fe, 4)
	q, k, v := makeSDPATensors(2, 3, 4)
	mask, _ := tensor.New[float32]([]int{1, 2, 3, 3}, nil)
	_, err := sdpa.Forward(context.Background(), q, k, v, mask)
	if err == nil {
		t.Error("expected error from mask Add failure")
	}
}

func TestSDPA_Forward_MaskReshapeBackError(t *testing.T) {
	fe := newFailingEngine(map[string]int{"Reshape": 2}) // 2nd Reshape (reshape back after mask)
	sdpa := NewScaledDotProductAttention[float32](fe, 4)
	q, k, v := makeSDPATensors(2, 3, 4)
	mask, _ := tensor.New[float32]([]int{1, 2, 3, 3}, nil)
	_, err := sdpa.Forward(context.Background(), q, k, v, mask)
	if err == nil {
		t.Error("expected error from reshape back failure")
	}
}

func TestSDPA_Backward_TransposeError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	// Replace engine with failing one for backward
	fe := newFailingEngine(map[string]int{"Transpose": 1})
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from Transpose failure in Backward")
	}
}

func TestSDPA_Backward_MatMulDVError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"MatMul": 1})
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from MatMul failure in Backward (dV)")
	}
}

func TestSDPA_Backward_TransposeVError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"Transpose": 2}) // 2nd Transpose (V^T)
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from Transpose V failure in Backward")
	}
}

func TestSDPA_Backward_MulError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"Mul": 1})
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from Mul failure in Backward")
	}
}

func TestSDPA_Backward_ReduceSumError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"ReduceSum": 1})
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from ReduceSum failure in Backward")
	}
}

func TestSDPA_Backward_SubError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"Sub": 1})
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from Sub failure in Backward")
	}
}

func TestSDPA_Backward_ScaleMulError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"Mul": 2}) // 2nd Mul (softmax grad)
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from 2nd Mul failure in Backward")
	}
}

func TestSDPA_Backward_MulScalarError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"MulScalar": 1})
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from MulScalar failure in Backward")
	}
}

func TestSDPA_Backward_MatMulDQError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"MatMul": 3}) // 3rd MatMul = dQ
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from MatMul dQ failure in Backward")
	}
}

func TestSDPA_Backward_TransposeDScoresError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"Transpose": 3}) // 3rd Transpose = dScores^T
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from Transpose dScores failure in Backward")
	}
}

func TestSDPA_Backward_MatMulDKError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"MatMul": 4}) // 4th MatMul = dK
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from MatMul dK failure in Backward")
	}
}

func TestSDPA_Backward_MatMulDAttWeightsError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 4)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, _ := sdpa.Forward(context.Background(), q, k, v, nil)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	fe := newFailingEngine(map[string]int{"MatMul": 2}) // 2nd MatMul = dAttWeights
	sdpa.engine = fe
	_, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error from MatMul dAttWeights failure in Backward")
	}
}

func TestSDPA_Forward_HeadDimZeroFallback(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	// Construct with headDim=0 so Forward falls back to inferring from Q shape
	sdpa := NewScaledDotProductAttention[float32](engine, 0)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward with headDim=0 fallback failed: %v", err)
	}
	if out == nil {
		t.Fatal("expected non-nil output")
	}
}

func TestSDPA_Backward_HeadDimZeroFallback(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 0)
	q, k, v := makeSDPATensors(1, 3, 4)

	out, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	grads, err := sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err != nil {
		t.Fatalf("Backward with headDim=0 fallback failed: %v", err)
	}
	if len(grads) != 3 {
		t.Errorf("expected 3 gradients, got %d", len(grads))
	}
}

func TestSDPA_Forward_HeadDimZero_NilQ(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 0)

	// Pass a 2D Q so the headDim fallback check (len(q.Shape()) < 3) triggers
	q2D, _ := tensor.New[float32]([]int{1, 4}, nil)
	k, _ := tensor.New[float32]([]int{1, 4}, nil)
	v, _ := tensor.New[float32]([]int{1, 4}, nil)

	// This should fail because with headDim=0 and 2D Q, can't infer head dim
	_, err := sdpa.Forward(context.Background(), q2D, k, v, nil)
	if err == nil {
		t.Error("expected error when headDim=0 and Q is 2D")
	}
}

func TestSDPA_Backward_HeadDimZero_NilQ(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sdpa := NewScaledDotProductAttention[float32](engine, 0)

	// Do a valid Forward first so backward has cached state
	q, k, v := makeSDPATensors(1, 3, 4)
	out, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	// Clear the cached Q to trigger the Backward fallback error path
	sdpa.q = nil
	_, err = sdpa.Backward(context.Background(), types.FullBackprop, dOut, q, k, v)
	if err == nil {
		t.Error("expected error when headDim=0 and cached Q is nil in Backward")
	}
}
