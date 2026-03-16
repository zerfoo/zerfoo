package attention

import (
	"context"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// cpuFusedEngine wraps a CPU engine and implements FusedQKNormRoPEProvider
// to return a CPU-resident tensor (no GPUStorage). This simulates the
// scenario where the fused kernel output lacks GPU storage.
type cpuFusedEngine struct {
	compute.Engine[float32]
}

func (e *cpuFusedEngine) GPUFusedQKNormRoPE(
	input *tensor.TensorNumeric[float32],
	weightQ, weightK *tensor.TensorNumeric[float32],
	cosAngles, sinAngles *tensor.TensorNumeric[float32],
	eps float32,
	totalHeads, headDim, numQHeads, halfRotary int,
) (*tensor.TensorNumeric[float32], error) {
	// Return a plain CPU tensor (no GPUStorage), simulating a broken path.
	outElems := totalHeads * headDim
	data := make([]float32, outElems)
	for i := range data {
		data[i] = 1.0
	}
	return tensor.New[float32]([]int{totalHeads, headDim}, data)
}

// Verify the interface is satisfied.
var _ compute.FusedQKNormRoPEProvider[float32] = (*cpuFusedEngine)(nil)

func TestFusedQKNormRoPE_RejectsCPUStorage(t *testing.T) {
	ops := numeric.Float32Ops{}
	inner := compute.NewCPUEngine[float32](ops)
	engine := &cpuFusedEngine{Engine: inner}

	modelDim := 8
	numQueryHeads := 2
	numKVHeads := 2
	headDim := modelDim / numQueryHeads // 4

	gqa, err := NewGroupedQueryAttention[float32](
		engine, ops, modelDim, numQueryHeads, numKVHeads,
		WithMaxSeqLen[float32](16),
	)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention: %v", err)
	}

	// Set QK norm weights to enable the fused path.
	qWeight, _ := tensor.New[float32]([]int{headDim}, []float32{1, 1, 1, 1})
	kWeight, _ := tensor.New[float32]([]int{headDim}, []float32{1, 1, 1, 1})
	gqa.SetQKNormWeights(qWeight, kWeight, 1e-5)

	// Also need to set qNorm/kNorm graph nodes so the unfused path works too.
	qNorm, _ := normalization.NewRMSNorm[float32]("q_norm", engine, ops, headDim)
	kNorm, _ := normalization.NewRMSNorm[float32]("k_norm", engine, ops, headDim)
	gqa.SetQKNorms(qNorm, kNorm)

	// Create a seqLen=1 input to trigger the decode (fused) path.
	batchSize := 1
	inp, _ := tensor.New[float32]([]int{batchSize, 1, modelDim}, nil)
	for i := range inp.Data() {
		inp.Data()[i] = float32(i+1) * 0.1
	}

	_, err = gqa.Forward(context.Background(), inp)
	if err == nil {
		t.Fatal("expected error when fused QK norm+RoPE returns CPU storage, got nil")
	}
	if !strings.Contains(err.Error(), "expected GPUStorage or Float16Storage") {
		t.Fatalf("unexpected error message: %v", err)
	}
}
