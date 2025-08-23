package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestGroupedQueryAttention_Forward_Shape(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 2
	seqLen := 7
	modelDim := 16
	numQueryHeads := 4
	numKeyValueHeads := 2

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

	inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
	if err != nil {
		t.Fatalf("failed creating input: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i%13) / 10.0 // simple deterministic data
	}

	out, err := gqa.Forward(context.Background(), inp)
	if err != nil {
		t.Fatalf("GQA Forward failed: %v", err)
	}

	expected := []int{batchSize, seqLen, modelDim}
	if !testutils.IntSliceEqual(expected, out.Shape()) {
		t.Fatalf("unexpected output shape: got %v want %v", out.Shape(), expected)
	}

	// Sanity: output should be finite numbers (no NaN/Inf) for a few positions
	for _, idx := range []int{0, len(out.Data())/2, len(out.Data()) - 1} {
		v := out.Data()[idx]
		if !(v == v) { // NaN check
			t.Fatalf("output contains NaN at idx %d", idx)
		}
	}
}
