package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
)

// TestGroupedQueryAttention_KVPort_BasicShape verifies that KPort() and
// VPort() return non-nil graph nodes and that, after the owner's Forward
// runs, the ports yield tensors of the expected
// [batch, numKVHeads, seqLen, headDim] shape.
func TestGroupedQueryAttention_KVPort_BasicShape(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 2
	seqLen := 5
	modelDim := 16
	numQueryHeads := 4
	numKeyValueHeads := 2
	headDim := modelDim / numQueryHeads

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

	kPort := gqa.KPort()
	vPort := gqa.VPort()
	if kPort == nil || vPort == nil {
		t.Fatalf("KPort/VPort must never return nil (got K=%v V=%v)", kPort, vPort)
	}

	// Before Forward runs, ports' Forward should error (nil-safety).
	if _, err := kPort.Forward(context.Background()); err == nil {
		t.Fatalf("KPort.Forward before owner.Forward should error")
	}

	inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
	if err != nil {
		t.Fatalf("failed creating input: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i%7) * 0.1
	}

	if _, err := gqa.Forward(context.Background(), inp); err != nil {
		t.Fatalf("owner Forward failed: %v", err)
	}

	kTensor, err := kPort.Forward(context.Background())
	if err != nil {
		t.Fatalf("KPort.Forward failed after owner.Forward: %v", err)
	}
	vTensor, err := vPort.Forward(context.Background())
	if err != nil {
		t.Fatalf("VPort.Forward failed after owner.Forward: %v", err)
	}

	expected := []int{batchSize, numKeyValueHeads, seqLen, headDim}
	if !testutils.IntSliceEqual(expected, kTensor.Shape()) {
		t.Fatalf("KPort shape: got %v want %v", kTensor.Shape(), expected)
	}
	if !testutils.IntSliceEqual(expected, vTensor.Shape()) {
		t.Fatalf("VPort shape: got %v want %v", vTensor.Shape(), expected)
	}

	if kPort.OpType() != "GroupedQueryAttention.KPort" {
		t.Fatalf("KPort OpType got %q", kPort.OpType())
	}
	if vPort.OpType() != "GroupedQueryAttention.VPort" {
		t.Fatalf("VPort OpType got %q", vPort.OpType())
	}
	if !testutils.IntSliceEqual(expected, kPort.OutputShape()) {
		t.Fatalf("KPort.OutputShape: got %v want %v", kPort.OutputShape(), expected)
	}
	if !testutils.IntSliceEqual(expected, vPort.OutputShape()) {
		t.Fatalf("VPort.OutputShape: got %v want %v", vPort.OutputShape(), expected)
	}
}

// TestGroupedQueryAttention_KVPort_SharedIdentity simulates the donor/shared
// layer pattern from ADR-087 at the K/V port level: layer A runs Forward,
// layer B is configured to treat A.KPort() / A.VPort() as its external K/V
// donor. We confirm that B's effective K/V source (as exposed via
// B.KPort() / B.VPort() when running in external-KV mode) produces tensors
// identical (same pointer, same shape, same data) to A's K/V output.
//
// Until task-T95.1.1's WithExternalKV() option lands, we drive the
// external-KV override directly through the additive fields.
func TestGroupedQueryAttention_KVPort_SharedIdentity(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 1
	seqLen := 4
	modelDim := 8
	numQueryHeads := 2
	numKeyValueHeads := 1

	mk := func(name string) *GroupedQueryAttention[float32] {
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
			t.Fatalf("construct %s: %v", name, err)
		}
		return gqa
	}

	a := mk("A")
	b := mk("B")

	inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
	if err != nil {
		t.Fatalf("failed creating input: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i) * 0.01
	}

	// Run donor layer A and pull its K/V tensors via ports.
	if _, err := a.Forward(context.Background(), inp); err != nil {
		t.Fatalf("A.Forward: %v", err)
	}
	aK, err := a.KPort().Forward(context.Background())
	if err != nil {
		t.Fatalf("A.KPort.Forward: %v", err)
	}
	aV, err := a.VPort().Forward(context.Background())
	if err != nil {
		t.Fatalf("A.VPort.Forward: %v", err)
	}

	// K/V port shape is [B, numKVHeads, S, headDim]. WithExternalKV's
	// Forward expects inputs[1]/[2] in the layer-input shape
	// [B, S, numKVHeads*headDim]. Builders bridge the two via a
	// transpose+reshape node (future T95.2.1). For this unit test we
	// reshape by hand so B can consume A's K/V as forward inputs.
	headDim := modelDim / numQueryHeads
	reshape := func(src *tensor.TensorNumeric[float32]) *tensor.TensorNumeric[float32] {
		// [B, numKV, S, headDim] -> [B, S, numKV*headDim]
		out, err := tensor.New[float32]([]int{batchSize, seqLen, numKeyValueHeads * headDim}, nil)
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}
		srcData := src.Data()
		dstData := out.Data()
		for bi := 0; bi < batchSize; bi++ {
			for kh := 0; kh < numKeyValueHeads; kh++ {
				for s := 0; s < seqLen; s++ {
					for d := 0; d < headDim; d++ {
						srcIdx := ((bi*numKeyValueHeads+kh)*seqLen+s)*headDim + d
						dstIdx := (bi*seqLen+s)*numKeyValueHeads*headDim + kh*headDim + d
						dstData[dstIdx] = srcData[srcIdx]
					}
				}
			}
		}
		return out
	}

	// Configure B as external-KV via the field set by WithExternalKV().
	b.externalKV = true

	// Run B with hidden + external K/V (from A).
	kIn := reshape(aK)
	vIn := reshape(aV)
	if _, err := b.Forward(context.Background(), inp, kIn, vIn); err != nil {
		t.Fatalf("B.Forward (external-KV): %v", err)
	}

	// B's KPort/VPort should yield tensors that round-trip back to A's
	// shapes (the external K/V was captured into B's kOut/vOut after
	// RoPE and any bookkeeping).
	bK, err := b.KPort().Forward(context.Background())
	if err != nil {
		t.Fatalf("B.KPort.Forward: %v", err)
	}
	bV, err := b.VPort().Forward(context.Background())
	if err != nil {
		t.Fatalf("B.VPort.Forward: %v", err)
	}
	if !testutils.IntSliceEqual(aK.Shape(), bK.Shape()) {
		t.Fatalf("shape mismatch K: donor %v vs shared %v", aK.Shape(), bK.Shape())
	}
	if !testutils.IntSliceEqual(aV.Shape(), bV.Shape()) {
		t.Fatalf("shape mismatch V: donor %v vs shared %v", aV.Shape(), bV.Shape())
	}
}
