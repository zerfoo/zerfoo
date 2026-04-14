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

	// Run donor layer A.
	if _, err := a.Forward(context.Background(), inp); err != nil {
		t.Fatalf("A.Forward: %v", err)
	}

	// Wire B to consume A's K/V ports as external donor. This emulates
	// what task-T95.1.1's WithExternalKV() option will produce.
	b.externalKV = true
	b.extKPortNode = a.KPort()
	b.extVPortNode = a.VPort()

	// Also run B on its own input to ensure the layer body still works.
	if _, err := b.Forward(context.Background(), inp); err != nil {
		t.Fatalf("B.Forward: %v", err)
	}

	// In external-KV mode, B.KPort()/VPort() must return the donor ports
	// *themselves* (not a fresh adapter), so downstream consumers reach
	// A's K/V without any intermediate hop.
	if b.KPort() != a.KPort() {
		// Note: KPort() constructs a fresh adapter each call for the local
		// case; in external mode we expect the stored donor node reference
		// to be returned verbatim. Since adapters are fresh, we compare
		// A's port through the same external-mode access pattern instead:
		// b.KPort() should === b.extKPortNode.
	}
	if b.KPort() != b.extKPortNode {
		t.Fatalf("B.KPort() in external-KV mode must return the donor port node")
	}
	if b.VPort() != b.extVPortNode {
		t.Fatalf("B.VPort() in external-KV mode must return the donor port node")
	}

	// The tensors reached through B's external ports must be identical
	// (same underlying *TensorNumeric) to A's K/V.
	aK, err := a.KPort().Forward(context.Background())
	if err != nil {
		t.Fatalf("A.KPort.Forward: %v", err)
	}
	aV, err := a.VPort().Forward(context.Background())
	if err != nil {
		t.Fatalf("A.VPort.Forward: %v", err)
	}
	bK, err := b.KPort().Forward(context.Background())
	if err != nil {
		t.Fatalf("B.KPort.Forward: %v", err)
	}
	bV, err := b.VPort().Forward(context.Background())
	if err != nil {
		t.Fatalf("B.VPort.Forward: %v", err)
	}

	if aK != bK {
		t.Fatalf("B's K in external-KV mode must be the same *TensorNumeric as A's K (got different pointers)")
	}
	if aV != bV {
		t.Fatalf("B's V in external-KV mode must be the same *TensorNumeric as A's V (got different pointers)")
	}
	if !testutils.IntSliceEqual(aK.Shape(), bK.Shape()) {
		t.Fatalf("shape mismatch K: %v vs %v", aK.Shape(), bK.Shape())
	}
	if !testutils.IntSliceEqual(aV.Shape(), bV.Shape()) {
		t.Fatalf("shape mismatch V: %v vs %v", aV.Shape(), bV.Shape())
	}
}
