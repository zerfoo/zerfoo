package inference

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
)

// TestKVReuseNode_BridgesDonorToConsumerLayout runs a donor GQA layer and
// routes its K/V via KVReuseNode, verifying that the emitted tensor has the
// [B, S, numKVHeads*headDim] layout expected by a consumer GQA's external-KV
// Forward input and that element ordering matches the explicit
// transpose+reshape reference.
func TestKVReuseNode_BridgesDonorToConsumerLayout(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 2
	seqLen := 4
	modelDim := 8
	numQueryHeads := 4
	numKVHeads := 2
	headDim := modelDim / numQueryHeads

	gqa, err := attention.NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		modelDim,
		numQueryHeads,
		numKVHeads,
		attention.WithRopeBase[float32](10000.0),
		attention.WithMaxSeqLen[float32](seqLen),
	)
	if err != nil {
		t.Fatalf("construct donor GQA: %v", err)
	}

	inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i%11) * 0.017
	}

	ctx := context.Background()
	if _, err := gqa.Forward(ctx, inp); err != nil {
		t.Fatalf("donor Forward: %v", err)
	}

	for _, tc := range []struct {
		name  string
		isKey bool
		port  func() (donor, _ interface{})
	}{
		{name: "K", isKey: true},
		{name: "V", isKey: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var donor = gqa.KPort()
			if !tc.isKey {
				donor = gqa.VPort()
			}

			node, err := NewKVReuseNode[float32](engine, donor, numKVHeads, headDim, tc.isKey)
			if err != nil {
				t.Fatalf("NewKVReuseNode: %v", err)
			}

			out, err := node.Forward(ctx)
			if err != nil {
				t.Fatalf("KVReuseNode.Forward: %v", err)
			}
			wantShape := []int{batchSize, seqLen, numKVHeads * headDim}
			if !testutils.IntSliceEqual(wantShape, out.Shape()) {
				t.Fatalf("output shape: got %v want %v", out.Shape(), wantShape)
			}
			if !testutils.IntSliceEqual(wantShape, node.OutputShape()) {
				t.Fatalf("OutputShape(): got %v want %v", node.OutputShape(), wantShape)
			}

			// Reference: read donor port directly and do the same
			// transpose+reshape by hand.
			src, err := donor.Forward(ctx)
			if err != nil {
				t.Fatalf("donor.Forward: %v", err)
			}
			ref := make([]float32, batchSize*seqLen*numKVHeads*headDim)
			srcData := src.Data()
			for b := 0; b < batchSize; b++ {
				for kh := 0; kh < numKVHeads; kh++ {
					for s := 0; s < seqLen; s++ {
						for d := 0; d < headDim; d++ {
							srcIdx := ((b*numKVHeads+kh)*seqLen+s)*headDim + d
							dstIdx := (b*seqLen+s)*numKVHeads*headDim + kh*headDim + d
							ref[dstIdx] = srcData[srcIdx]
						}
					}
				}
			}
			gotData := out.Data()
			for i, v := range ref {
				if gotData[i] != v {
					t.Fatalf("data mismatch at %d: got %v want %v", i, gotData[i], v)
				}
			}

			// Metadata checks.
			wantOp := "KVReuseNode.K"
			if !tc.isKey {
				wantOp = "KVReuseNode.V"
			}
			if node.OpType() != wantOp {
				t.Fatalf("OpType: got %q want %q", node.OpType(), wantOp)
			}
			attrs := node.Attributes()
			if attrs["num_kv_heads"].(int) != numKVHeads || attrs["head_dim"].(int) != headDim {
				t.Fatalf("Attributes: %v", attrs)
			}
			if node.Parameters() != nil {
				t.Fatalf("Parameters should be nil, got %v", node.Parameters())
			}
		})
	}
}

// TestKVReuseNode_FeedsConsumerExternalKV confirms that the node's output is
// directly consumable as inputs[1]/inputs[2] for a consumer GQA configured
// with WithExternalKV -- the end-to-end contract of ADR-087.
func TestKVReuseNode_FeedsConsumerExternalKV(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 1
	seqLen := 3
	modelDim := 8
	numQueryHeads := 2
	numKVHeads := 1
	headDim := modelDim / numQueryHeads

	mk := func(withExt bool) *attention.GroupedQueryAttention[float32] {
		opts := []attention.GQAOption[float32]{
			attention.WithRopeBase[float32](10000.0),
			attention.WithMaxSeqLen[float32](seqLen),
		}
		if withExt {
			opts = append(opts, attention.WithExternalKV[float32]())
		}
		gqa, err := attention.NewGroupedQueryAttention[float32](
			engine, numeric.Float32Ops{}, modelDim, numQueryHeads, numKVHeads, opts...,
		)
		if err != nil {
			t.Fatalf("construct GQA: %v", err)
		}
		return gqa
	}

	donor := mk(false)
	consumer := mk(true)

	inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i) * 0.013
	}

	ctx := context.Background()
	if _, err := donor.Forward(ctx, inp); err != nil {
		t.Fatalf("donor.Forward: %v", err)
	}

	kNode, err := NewKVReuseNode[float32](engine, donor.KPort(), numKVHeads, headDim, true)
	if err != nil {
		t.Fatalf("NewKVReuseNode K: %v", err)
	}
	vNode, err := NewKVReuseNode[float32](engine, donor.VPort(), numKVHeads, headDim, false)
	if err != nil {
		t.Fatalf("NewKVReuseNode V: %v", err)
	}

	kIn, err := kNode.Forward(ctx)
	if err != nil {
		t.Fatalf("kNode.Forward: %v", err)
	}
	vIn, err := vNode.Forward(ctx)
	if err != nil {
		t.Fatalf("vNode.Forward: %v", err)
	}

	if _, err := consumer.Forward(ctx, inp, kIn, vIn); err != nil {
		t.Fatalf("consumer.Forward (external-KV): %v", err)
	}
}
