package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// newTestMLAForBackward creates an MLA with deterministic weights for backward testing.
func newTestMLAForBackward(t *testing.T) (*MultiHeadLatentAttention[float32], int) {
	t.Helper()
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	numHeads := 2
	headDim := 4
	kvLoraDim := 3
	hiddenDim := 8

	qkDim := numHeads * headDim // 8

	wQ, err := core.NewDense[float32]("wQ", engine, ops, hiddenDim, qkDim, core.WithoutBias[float32]())
	if err != nil {
		t.Fatalf("wQ: %v", err)
	}
	wDKV, err := core.NewDense[float32]("wDKV", engine, ops, hiddenDim, kvLoraDim, core.WithoutBias[float32]())
	if err != nil {
		t.Fatalf("wDKV: %v", err)
	}
	wUK, err := core.NewDense[float32]("wUK", engine, ops, kvLoraDim, qkDim, core.WithoutBias[float32]())
	if err != nil {
		t.Fatalf("wUK: %v", err)
	}
	wUV, err := core.NewDense[float32]("wUV", engine, ops, kvLoraDim, qkDim, core.WithoutBias[float32]())
	if err != nil {
		t.Fatalf("wUV: %v", err)
	}
	wO, err := core.NewDense[float32]("wO", engine, ops, qkDim, hiddenDim, core.WithoutBias[float32]())
	if err != nil {
		t.Fatalf("wO: %v", err)
	}

	// Seed weights with small deterministic values.
	for i, p := range wQ.Parameters() {
		for j := range p.Value.Data() {
			p.Value.Data()[j] = float32(((j*7+i*3+1)%19)-9) / 40.0
		}
	}
	for i, p := range wDKV.Parameters() {
		for j := range p.Value.Data() {
			p.Value.Data()[j] = float32(((j*11+i*5+2)%17)-8) / 40.0
		}
	}
	for i, p := range wUK.Parameters() {
		for j := range p.Value.Data() {
			p.Value.Data()[j] = float32(((j*13+i*7+3)%23)-11) / 40.0
		}
	}
	for i, p := range wUV.Parameters() {
		for j := range p.Value.Data() {
			p.Value.Data()[j] = float32(((j*17+i*11+5)%29)-14) / 40.0
		}
	}
	for i, p := range wO.Parameters() {
		for j := range p.Value.Data() {
			p.Value.Data()[j] = float32(((j*19+i*13+7)%31)-15) / 40.0
		}
	}

	rope, err := embeddings.NewRotaryPositionalEmbedding[float32](ctx, engine, headDim, 16)
	if err != nil {
		t.Fatalf("rope: %v", err)
	}

	mla := NewMultiHeadLatentAttention(engine, ops, numHeads, headDim, kvLoraDim, 0, wQ, wDKV, wUK, wUV, wO, rope)
	return mla, hiddenDim
}

func TestMLABackward(t *testing.T) {
	ctx := context.Background()
	mla, hiddenDim := newTestMLAForBackward(t)

	batch := 1
	seqLen := 2

	// Create deterministic input.
	inputData := make([]float32, batch*seqLen*hiddenDim)
	for i := range inputData {
		inputData[i] = float32(((i*7+3)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, hiddenDim}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	// Forward pass.
	out, err := mla.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Create upstream gradient.
	dOutData := make([]float32, len(out.Data()))
	for i := range dOutData {
		dOutData[i] = float32(((i*11+5)%13)-6) / 10.0
	}
	dOut, _ := tensor.New[float32](out.Shape(), dOutData)

	// Backward pass.
	grads, err := mla.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient, got %d", len(grads))
	}

	dInput := grads[0]
	if !intSliceEq(dInput.Shape(), input.Shape()) {
		t.Fatalf("dInput shape = %v, want %v", dInput.Shape(), input.Shape())
	}

	// Verify no NaN in gradient.
	for i, v := range dInput.Data() {
		if v != v {
			t.Fatalf("NaN in dInput at index %d", i)
		}
	}

	// Finite-difference verification.
	eps := float32(1e-3)
	tol := float32(5e-2)

	analyticalGrad := make([]float32, len(dInput.Data()))
	copy(analyticalGrad, dInput.Data())

	lossFn := func() float32 {
		o, err := mla.Forward(ctx, input)
		if err != nil {
			t.Fatalf("lossFn Forward: %v", err)
		}
		return dotProduct(o.Data(), dOutData)
	}

	data := input.Data()
	numFailed := 0
	for i := range data {
		orig := data[i]

		data[i] = orig + eps
		lPlus := lossFn()

		data[i] = orig - eps
		lMinus := lossFn()

		data[i] = orig

		numerical := (lPlus - lMinus) / (2 * eps)
		a := analyticalGrad[i]
		diff := float32(math.Abs(float64(a - numerical)))
		denom := float32(math.Max(1.0, math.Max(math.Abs(float64(a)), math.Abs(float64(numerical)))))

		if diff/denom > tol {
			numFailed++
			if numFailed <= 5 {
				t.Errorf("dInput[%d]: analytical=%.6f numerical=%.6f relErr=%.4f",
					i, a, numerical, diff/denom)
			}
		}
	}
	if numFailed > 0 {
		t.Fatalf("dInput: %d/%d exceeded tol=%.4f", numFailed, len(data), tol)
	}
	t.Logf("dInput: %d elements passed finite-diff check", len(data))
}

func TestMLABackward_InputValidation(t *testing.T) {
	ctx := context.Background()
	mla, hiddenDim := newTestMLAForBackward(t)

	input, _ := tensor.New[float32]([]int{1, 2, hiddenDim}, nil)
	for i := range input.Data() {
		input.Data()[i] = 0.1
	}
	out, err := mla.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	dOut, _ := tensor.New[float32](out.Shape(), nil)

	// Wrong number of inputs.
	_, err = mla.Backward(ctx, types.FullBackprop, dOut)
	if err == nil {
		t.Error("expected error for 0 inputs, got nil")
	}

	_, err = mla.Backward(ctx, types.FullBackprop, dOut, input, input)
	if err == nil {
		t.Error("expected error for 2 inputs, got nil")
	}
}

