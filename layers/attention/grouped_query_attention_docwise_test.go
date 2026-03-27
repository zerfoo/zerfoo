package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestDocumentWiseRoPE_GQA verifies that setting document boundaries on the
// GQA layer produces different output compared to no boundaries, confirming
// that document-wise RoPE is wired through correctly.
func TestDocumentWiseRoPE_GQA(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 1
	seqLen := 6
	modelDim := 16
	numQueryHeads := 4
	numKeyValueHeads := 2

	makeGQA := func(t *testing.T) *GroupedQueryAttention[float32] {
		t.Helper()
		gqa, err := NewGroupedQueryAttention[float32](
			engine,
			numeric.Float32Ops{},
			modelDim,
			numQueryHeads,
			numKeyValueHeads,
			WithRopeBase[float32](10000.0),
			WithMaxSeqLen[float32](seqLen),
		)
		if err != nil {
			t.Fatalf("failed to construct GQA: %v", err)
		}
		return gqa
	}

	makeInput := func(t *testing.T) *tensor.TensorNumeric[float32] {
		t.Helper()
		inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
		if err != nil {
			t.Fatalf("failed creating input: %v", err)
		}
		for i := range inp.Data() {
			inp.Data()[i] = float32(i%13) / 10.0
		}
		return inp
	}

	// Run without document boundaries.
	gqaBase := makeGQA(t)
	inpBase := makeInput(t)
	outBase, err := gqaBase.Forward(context.Background(), inpBase)
	if err != nil {
		t.Fatalf("GQA Forward (no boundaries) failed: %v", err)
	}

	// Run with document boundaries: new document starts at position 3.
	gqaDoc := makeGQA(t)
	// Copy weights from base so the only difference is document boundaries.
	copyParams(t, gqaBase, gqaDoc)
	gqaDoc.SetDocumentBoundaries([]int{0, 3})

	inpDoc := makeInput(t)
	outDoc, err := gqaDoc.Forward(context.Background(), inpDoc)
	if err != nil {
		t.Fatalf("GQA Forward (with boundaries) failed: %v", err)
	}

	// Outputs must have the same shape.
	if len(outBase.Shape()) != len(outDoc.Shape()) {
		t.Fatalf("shape rank mismatch: %v vs %v", outBase.Shape(), outDoc.Shape())
	}
	for i := range outBase.Shape() {
		if outBase.Shape()[i] != outDoc.Shape()[i] {
			t.Fatalf("shape mismatch at dim %d: %d vs %d", i, outBase.Shape()[i], outDoc.Shape()[i])
		}
	}

	// The outputs must differ because document-wise RoPE resets positions.
	baseData := outBase.Data()
	docData := outDoc.Data()
	allSame := true
	for i := range baseData {
		if baseData[i] != docData[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Fatal("expected output to differ with document boundaries set, but outputs are identical")
	}
}

// TestDocumentWiseRoPE_GQA_NilClearsBoundaries verifies that passing nil
// to SetDocumentBoundaries disables document-wise mode and produces the
// same output as never setting boundaries.
func TestDocumentWiseRoPE_GQA_NilClearsBoundaries(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 1
	seqLen := 6
	modelDim := 16
	numQueryHeads := 4
	numKeyValueHeads := 2

	makeGQA := func(t *testing.T) *GroupedQueryAttention[float32] {
		t.Helper()
		gqa, err := NewGroupedQueryAttention[float32](
			engine,
			numeric.Float32Ops{},
			modelDim,
			numQueryHeads,
			numKeyValueHeads,
			WithRopeBase[float32](10000.0),
			WithMaxSeqLen[float32](seqLen),
		)
		if err != nil {
			t.Fatalf("failed to construct GQA: %v", err)
		}
		return gqa
	}

	makeInput := func(t *testing.T) *tensor.TensorNumeric[float32] {
		t.Helper()
		inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
		if err != nil {
			t.Fatalf("failed creating input: %v", err)
		}
		for i := range inp.Data() {
			inp.Data()[i] = float32(i%13) / 10.0
		}
		return inp
	}

	// Run without boundaries.
	gqaBase := makeGQA(t)
	inpBase := makeInput(t)
	outBase, err := gqaBase.Forward(context.Background(), inpBase)
	if err != nil {
		t.Fatalf("GQA Forward (no boundaries) failed: %v", err)
	}

	// Set boundaries then clear them with nil.
	gqaCleared := makeGQA(t)
	copyParams(t, gqaBase, gqaCleared)
	gqaCleared.SetDocumentBoundaries([]int{0, 3})
	gqaCleared.SetDocumentBoundaries(nil)

	inpCleared := makeInput(t)
	outCleared, err := gqaCleared.Forward(context.Background(), inpCleared)
	if err != nil {
		t.Fatalf("GQA Forward (cleared boundaries) failed: %v", err)
	}

	// Outputs should be identical after clearing boundaries.
	baseData := outBase.Data()
	clearedData := outCleared.Data()
	for i := range baseData {
		if baseData[i] != clearedData[i] {
			t.Fatalf("output differs at index %d after clearing boundaries: %f vs %f", i, baseData[i], clearedData[i])
		}
	}
}

// copyParams copies all weight parameters from src to dst so they produce
// identical results when given the same input (aside from RoPE config).
func copyParams(t *testing.T, src, dst *GroupedQueryAttention[float32]) {
	t.Helper()
	srcParams := src.Parameters()
	dstParams := dst.Parameters()
	if len(srcParams) != len(dstParams) {
		t.Fatalf("parameter count mismatch: %d vs %d", len(srcParams), len(dstParams))
	}
	for i := range srcParams {
		srcData := srcParams[i].Value.Data()
		dstData := dstParams[i].Value.Data()
		copy(dstData, srcData)
	}
}
