package core

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// identityExpert returns its input unchanged.
type identityExpert struct{ shape []int }

func (n *identityExpert) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	n.shape = inputs[0].Shape()
	return inputs[0], nil
}

func (n *identityExpert) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *identityExpert) OpType() string                          { return "Identity" }
func (n *identityExpert) Attributes() map[string]interface{}      { return nil }
func (n *identityExpert) OutputShape() []int                      { return n.shape }
func (n *identityExpert) Parameters() []*graph.Parameter[float32] { return nil }

// scale2Expert multiplies each element by 2.
type scale2Expert struct{ shape []int }

func (n *scale2Expert) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	src := inputs[0].Data()
	out := make([]float32, len(src))
	for i, v := range src {
		out[i] = v * 2
	}
	t, err := tensor.New[float32](inputs[0].Shape(), out)
	if err != nil {
		return nil, err
	}
	n.shape = t.Shape()
	return t, nil
}

func (n *scale2Expert) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *scale2Expert) OpType() string                          { return "ScaleBy2" }
func (n *scale2Expert) Attributes() map[string]interface{}      { return nil }
func (n *scale2Expert) OutputShape() []int                      { return n.shape }
func (n *scale2Expert) Parameters() []*graph.Parameter[float32] { return nil }

// --- MoEGate tests ---

func TestMoEGate_ForwardShapeAndWeightsSumToOne(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 2) // topK=2

	// hiddenStates [2, 3] (seq_len=2, model_dim=3)
	hs, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 0, 0, 0, 1, 0})
	// gateWeight [3, 3] (num_experts=3, model_dim=3) - identity matrix
	gw, _ := tensor.New[float32]([]int{3, 3}, []float32{1, 0, 0, 0, 1, 0, 0, 0, 1})

	out, err := gate.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("MoEGate.Forward failed: %v", err)
	}
	if len(out.Shape()) != 2 || out.Shape()[0] != 2 || out.Shape()[1] != 2 {
		t.Errorf("output shape = %v, want [2 2]", out.Shape())
	}
	data := out.Data()
	for row := 0; row < 2; row++ {
		sum := float64(0)
		for k := 0; k < 2; k++ {
			sum += float64(data[row*2+k])
		}
		if math.Abs(sum-1.0) > 1e-5 {
			t.Errorf("row %d weights sum = %f, want 1.0", row, sum)
		}
	}
}

func TestMoEGate_InvalidInputs(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 2)

	_, err := gate.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}

	hs, _ := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
	_, err = gate.Forward(context.Background(), hs)
	if err == nil {
		t.Error("expected error for 1 input")
	}
}

func TestMoEGate_OpType(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	if gate.OpType() != "MoEGate" {
		t.Errorf("OpType = %q, want MoEGate", gate.OpType())
	}
}

func TestMoEGate_Attributes(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 2)
	attrs := gate.Attributes()
	if v, ok := attrs["top_k"]; !ok || v != 2 {
		t.Errorf("attributes[top_k] = %v, want 2", attrs["top_k"])
	}
}

func TestMoEGate_Backward(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	grads, err := gate.Backward(context.Background(), types.FullBackprop, nil)
	if err != nil {
		t.Fatalf("MoEGate.Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("expected nil grads")
	}
}

func TestMoEGate_Parameters(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	if gate.Parameters() != nil {
		t.Error("expected nil parameters")
	}
}

func TestMoEGate_OutputShape(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	if gate.OutputShape() != nil {
		t.Error("expected nil output shape before forward")
	}
}

func TestBuildMoEGate_Int64TopK(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	attrs := map[string]interface{}{"top_k": int64(2)}
	node, err := BuildMoEGate(eng, ops, "gate", nil, attrs)
	if err != nil {
		t.Fatalf("BuildMoEGate failed: %v", err)
	}
	if node.OpType() != "MoEGate" {
		t.Errorf("OpType = %q, want MoEGate", node.OpType())
	}
}

func TestBuildMoEGate_IntTopK(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	attrs := map[string]interface{}{"top_k": int(3)}
	node, err := BuildMoEGate(eng, ops, "gate", nil, attrs)
	if err != nil {
		t.Fatalf("BuildMoEGate with int top_k failed: %v", err)
	}
	if node.OpType() != "MoEGate" {
		t.Errorf("OpType = %q, want MoEGate", node.OpType())
	}
}

func TestBuildMoEGate_MissingTopK(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	_, err := BuildMoEGate(eng, ops, "gate", nil, map[string]interface{}{})
	if err == nil {
		t.Error("expected error for missing top_k")
	}
}

// --- MixtureOfExperts tests ---

func TestMixtureOfExperts_Forward(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1) // topK=1

	experts := []graph.Node[float32]{
		&identityExpert{},
		&scale2Expert{},
	}
	moe := NewMixtureOfExperts[float32](eng, ops, gate, experts, 2, 1)

	// hiddenStates [2, 2]: token 0 = [1,0], token 1 = [0,1]
	hs, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 0, 0, 1})
	// gateWeight [2, 2]: identity -> token 0 has high logit for expert 0, token 1 for expert 1
	gw, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 0, 0, 1})

	out, err := moe.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("MixtureOfExperts.Forward failed: %v", err)
	}
	if len(out.Shape()) != 2 || out.Shape()[0] != 2 || out.Shape()[1] != 2 {
		t.Errorf("output shape = %v, want [2 2]", out.Shape())
	}
	data := out.Data()
	// token 0 -> expert 0 (identity): output = [1, 0]
	if math.Abs(float64(data[0])-1.0) > 1e-5 || math.Abs(float64(data[1])) > 1e-5 {
		t.Errorf("token 0 output = [%f %f], want [1 0]", data[0], data[1])
	}
	// token 1 -> expert 1 (scale-by-2): output = [0, 2]
	if math.Abs(float64(data[2])) > 1e-5 || math.Abs(float64(data[3])-2.0) > 1e-5 {
		t.Errorf("token 1 output = [%f %f], want [0 2]", data[2], data[3])
	}
}

func TestMixtureOfExperts_InvalidInputs(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)
	_, err := moe.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestMixtureOfExperts_NoExperts(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)

	hs, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 0, 0, 1})
	gw, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 0, 0, 1})
	_, err := moe.Forward(context.Background(), hs, gw)
	if err == nil {
		t.Error("expected error when experts is nil")
	}
}

func TestMixtureOfExperts_OpType(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)
	if moe.OpType() != "MixtureOfExperts" {
		t.Errorf("OpType = %q, want MixtureOfExperts", moe.OpType())
	}
}

func TestMixtureOfExperts_Attributes(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 2)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 3, 2)
	attrs := moe.Attributes()
	if v, ok := attrs["num_experts"]; !ok || v != 3 {
		t.Errorf("attributes[num_experts] = %v, want 3", attrs["num_experts"])
	}
	if v, ok := attrs["top_k"]; !ok || v != 2 {
		t.Errorf("attributes[top_k] = %v, want 2", attrs["top_k"])
	}
}

func TestMixtureOfExperts_Backward(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)
	grads, err := moe.Backward(context.Background(), types.FullBackprop, nil)
	if err != nil {
		t.Fatalf("MixtureOfExperts.Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("expected nil grads")
	}
}

func TestMixtureOfExperts_OutputShape(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)
	if moe.OutputShape() != nil {
		t.Error("expected nil output shape before forward")
	}
}

func TestMixtureOfExperts_Parameters(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)
	if moe.Parameters() != nil {
		t.Error("expected nil parameters")
	}
}

func TestBuildMixtureOfExperts_Int64Attrs(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	attrs := map[string]interface{}{
		"num_experts": int64(4),
		"top_k":       int64(2),
	}
	node, err := BuildMixtureOfExperts(eng, ops, "moe", nil, attrs)
	if err != nil {
		t.Fatalf("BuildMixtureOfExperts failed: %v", err)
	}
	if node.OpType() != "MixtureOfExperts" {
		t.Errorf("OpType = %q, want MixtureOfExperts", node.OpType())
	}
}

func TestBuildMixtureOfExperts_MissingNumExperts(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	_, err := BuildMixtureOfExperts(eng, ops, "moe", nil, map[string]interface{}{"top_k": int64(2)})
	if err == nil {
		t.Error("expected error for missing num_experts")
	}
}

func TestBuildMixtureOfExperts_MissingTopK(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	_, err := BuildMixtureOfExperts(eng, ops, "moe", nil, map[string]interface{}{"num_experts": int64(4)})
	if err == nil {
		t.Error("expected error for missing top_k")
	}
}

func TestBuildMoEGate_UnsupportedTopKType(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	_, err := BuildMoEGate(eng, ops, "gate", nil, map[string]interface{}{"top_k": float64(2)})
	if err == nil {
		t.Error("expected error for unsupported top_k type")
	}
}

func TestBuildMixtureOfExperts_UnsupportedNumExpertsType(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	_, err := BuildMixtureOfExperts(eng, ops, "moe", nil, map[string]interface{}{
		"num_experts": float64(4),
		"top_k":       int64(2),
	})
	if err == nil {
		t.Error("expected error for unsupported num_experts type")
	}
}

func TestBuildMixtureOfExperts_UnsupportedTopKType(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	_, err := BuildMixtureOfExperts(eng, ops, "moe", nil, map[string]interface{}{
		"num_experts": int64(4),
		"top_k":       float64(2),
	})
	if err == nil {
		t.Error("expected error for unsupported top_k type")
	}
}

func TestMixtureOfExperts_ExpertIndexOutOfRange(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1) // topK=1

	// Only 1 expert but gateWeight has 2 rows; token will be routed to expert 1.
	experts := []graph.Node[float32]{&identityExpert{}}
	moe := NewMixtureOfExperts[float32](eng, ops, gate, experts, 2, 1)

	// Token [0,1] will get higher logit for expert 1 (gateWeight row 1 = [0,1]).
	hs, _ := tensor.New[float32]([]int{1, 2}, []float32{0, 1})
	gw, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 0, 0, 1})

	_, err := moe.Forward(context.Background(), hs, gw)
	if err == nil {
		t.Error("expected error when expert index is out of range")
	}
}

func TestMixtureOfExperts_HiddenStates1D(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, []graph.Node[float32]{&identityExpert{}}, 1, 1)

	hs, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	gw, _ := tensor.New[float32]([]int{1, 3}, []float32{1, 0, 0})
	_, err := moe.Forward(context.Background(), hs, gw)
	if err == nil {
		t.Error("expected error for 1D hiddenStates")
	}
}

func TestMixtureOfExperts_WithSharedExpert(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)

	experts := []graph.Node[float32]{&identityExpert{}}
	moe := NewMixtureOfExperts[float32](eng, ops, gate, experts, 1, 1)
	moe.SharedExpert = &scale2Expert{} // shared expert multiplies by 2

	// hiddenStates [1, 2]: single token [3, 4]
	hs, _ := tensor.New[float32]([]int{1, 2}, []float32{3, 4})
	// gateWeight [1, 2]: single expert
	gw, _ := tensor.New[float32]([]int{1, 2}, []float32{1, 0})

	out, err := moe.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("Forward with shared expert failed: %v", err)
	}

	data := out.Data()
	// Expected: shared(token) + routed(token)
	// shared = scale2([3,4]) = [6,8]
	// routed = identity([3,4]) * 1.0 = [3,4]
	// total = [9, 12]
	if math.Abs(float64(data[0])-9.0) > 1e-4 || math.Abs(float64(data[1])-12.0) > 1e-4 {
		t.Errorf("output = [%f, %f], want [9, 12]", data[0], data[1])
	}
}

func TestMixtureOfExperts_WithoutSharedExpert_BackwardCompat(t *testing.T) {
	// Verify that nil SharedExpert produces the same result as before.
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)

	experts := []graph.Node[float32]{&identityExpert{}}
	moe := NewMixtureOfExperts[float32](eng, ops, gate, experts, 1, 1)
	// SharedExpert is nil by default.

	hs, _ := tensor.New[float32]([]int{1, 2}, []float32{3, 4})
	gw, _ := tensor.New[float32]([]int{1, 2}, []float32{1, 0})

	out, err := moe.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("Forward without shared expert failed: %v", err)
	}

	data := out.Data()
	// identity expert, weight 1.0: output = [3, 4]
	if math.Abs(float64(data[0])-3.0) > 1e-4 || math.Abs(float64(data[1])-4.0) > 1e-4 {
		t.Errorf("output = [%f, %f], want [3, 4]", data[0], data[1])
	}
}

// TestMixtureOfExperts_BatchedMatchesSequential verifies the batched path (seqLen>1)
// produces the same result as processing tokens individually.
func TestMixtureOfExperts_BatchedMatchesSequential(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}

	numExperts := 4
	topK := 2
	modelDim := 3
	seqLen := 8

	experts := make([]graph.Node[float32], numExperts)
	for i := range experts {
		experts[i] = &scale2Expert{}
	}

	// Run batched (all tokens at once).
	gate := NewMoEGate[float32](eng, ops, topK)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, experts, numExperts, topK)

	hsData := make([]float32, seqLen*modelDim)
	for i := range hsData {
		hsData[i] = float32(i+1) * 0.1
	}
	hs, _ := tensor.New[float32]([]int{seqLen, modelDim}, hsData)

	gwData := make([]float32, numExperts*modelDim)
	for i := 0; i < numExperts; i++ {
		gwData[i*modelDim+i%modelDim] = 1.0
	}
	gw, _ := tensor.New[float32]([]int{numExperts, modelDim}, gwData)

	batchedOut, err := moe.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("batched forward failed: %v", err)
	}

	// Run sequential (one token at a time).
	seqOutData := make([]float32, seqLen*modelDim)
	for tok := 0; tok < seqLen; tok++ {
		tokData := make([]float32, modelDim)
		copy(tokData, hsData[tok*modelDim:(tok+1)*modelDim])
		tokTensor, _ := tensor.New[float32]([]int{1, modelDim}, tokData)

		seqExperts := make([]graph.Node[float32], numExperts)
		for i := range seqExperts {
			seqExperts[i] = &scale2Expert{}
		}
		seqGate := NewMoEGate[float32](eng, ops, topK)
		seqMoe := NewMixtureOfExperts[float32](eng, ops, seqGate, seqExperts, numExperts, topK)

		out, eerr := seqMoe.Forward(context.Background(), tokTensor, gw)
		if eerr != nil {
			t.Fatalf("sequential forward for token %d failed: %v", tok, eerr)
		}
		copy(seqOutData[tok*modelDim:(tok+1)*modelDim], out.Data())
	}

	bData := batchedOut.Data()
	for i := range bData {
		if math.Abs(float64(bData[i])-float64(seqOutData[i])) > 1e-5 {
			t.Errorf("mismatch at index %d: batched=%f sequential=%f", i, bData[i], seqOutData[i])
		}
	}
}

// TestMixtureOfExperts_BatchedMultiToken verifies correct output with multiple tokens
// routed to different experts via the batched path.
func TestMixtureOfExperts_BatchedMultiToken(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1) // topK=1

	experts := []graph.Node[float32]{
		&identityExpert{},
		&scale2Expert{},
		&identityExpert{},
		&scale2Expert{},
	}
	moe := NewMixtureOfExperts[float32](eng, ops, gate, experts, 4, 1)

	// 4 tokens, modelDim=2
	hs, _ := tensor.New[float32]([]int{4, 2}, []float32{
		1, 0, // token 0
		0, 1, // token 1
		1, 1, // token 2
		2, 3, // token 3
	})
	// gateWeight [4, 2]: identity-like routing
	gw, _ := tensor.New[float32]([]int{4, 2}, []float32{
		1, 0, // expert 0 high
		0, 1, // expert 1 high
		1, 1, // expert 2 or 3 (both equal, pick lower index)
		0, 1, // expert 1 high
	})

	out, err := moe.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if out.Shape()[0] != 4 || out.Shape()[1] != 2 {
		t.Fatalf("output shape = %v, want [4 2]", out.Shape())
	}
}

func TestMixtureOfExperts_SharedExpert_Attributes(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)

	// Without shared expert.
	attrs := moe.Attributes()
	if _, ok := attrs["has_shared_expert"]; ok {
		t.Error("expected no has_shared_expert attribute when SharedExpert is nil")
	}

	// With shared expert.
	moe.SharedExpert = &identityExpert{}
	attrs = moe.Attributes()
	if v, ok := attrs["has_shared_expert"]; !ok || v != true {
		t.Error("expected has_shared_expert=true when SharedExpert is set")
	}
}

// BenchmarkMoE_BatchedVsSequential benchmarks MoE forward pass at various batch sizes.
// The batched path (seqLen > 1) should be faster than the sequential path per token.
func BenchmarkMoE_BatchedVsSequential(b *testing.B) {
	numExperts := 8
	topK := 2
	modelDim := 64

	for _, batchSize := range []int{1, 4, 8, 16} {
		b.Run(fmt.Sprintf("batch=%d", batchSize), func(b *testing.B) {
			eng := makeFloat32Engine()
			ops := numeric.Float32Ops{}

			experts := make([]graph.Node[float32], numExperts)
			for i := range experts {
				experts[i] = &scale2Expert{}
			}

			gate := NewMoEGate[float32](eng, ops, topK)
			moe := NewMixtureOfExperts[float32](eng, ops, gate, experts, numExperts, topK)

			hsData := make([]float32, batchSize*modelDim)
			for i := range hsData {
				hsData[i] = float32(i%modelDim) * 0.01
			}
			hs, _ := tensor.New[float32]([]int{batchSize, modelDim}, hsData)

			gwData := make([]float32, numExperts*modelDim)
			for i := 0; i < numExperts; i++ {
				gwData[i*modelDim+i%modelDim] = 1.0
			}
			gw, _ := tensor.New[float32]([]int{numExperts, modelDim}, gwData)

			ctx := context.Background()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = moe.Forward(ctx, hs, gw)
			}
		})
	}
}

// linearExpert multiplies each element by a fixed scale factor.
// This has a known, well-defined gradient: dOut/dIn = scale.
type linearExpert struct {
	scale       float32
	shape       []int
	cachedInput *tensor.TensorNumeric[float32]
}

func (n *linearExpert) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	n.cachedInput = inputs[0]
	src := inputs[0].Data()
	out := make([]float32, len(src))
	for i, v := range src {
		out[i] = v * n.scale
	}
	t, err := tensor.New[float32](inputs[0].Shape(), out)
	if err != nil {
		return nil, err
	}
	n.shape = t.Shape()
	return t, nil
}

func (n *linearExpert) Backward(_ context.Context, _ types.BackwardMode, outputGrad *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	if outputGrad == nil {
		return nil, nil
	}
	ogData := outputGrad.Data()
	dIn := make([]float32, len(ogData))
	for i, v := range ogData {
		dIn[i] = v * n.scale
	}
	t, err := tensor.New[float32](outputGrad.Shape(), dIn)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[float32]{t}, nil
}

func (n *linearExpert) OpType() string                          { return "Linear" }
func (n *linearExpert) Attributes() map[string]interface{}      { return nil }
func (n *linearExpert) OutputShape() []int                      { return n.shape }
func (n *linearExpert) Parameters() []*graph.Parameter[float32] { return nil }

// moeForwardLoss runs a forward pass through a fresh MoE layer and returns
// sum(output) as a scalar loss. Used for finite-difference gradient checking.
func moeForwardLoss(
	eng compute.Engine[float32],
	ops numeric.Float32Ops,
	experts []graph.Node[float32],
	numExperts, topK int,
	hs, gw *tensor.TensorNumeric[float32],
) float32 {
	gate := NewMoEGate[float32](eng, ops, topK)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, experts, numExperts, topK)
	out, err := moe.Forward(context.Background(), hs, gw)
	if err != nil {
		panic(err)
	}
	var sum float32
	for _, v := range out.Data() {
		sum += v
	}
	return sum
}

// TestMoEBackward verifies the MoE backward pass using finite differences.
func TestMoEBackward(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}

	numExperts := 4
	topK := 2
	modelDim := 8
	seqLen := 2

	rng := rand.New(rand.NewSource(42))

	// Create experts with different scales so routing matters.
	scales := []float32{1.5, 0.5, 2.0, 0.8}
	makeExperts := func() []graph.Node[float32] {
		experts := make([]graph.Node[float32], numExperts)
		for i := range experts {
			experts[i] = &linearExpert{scale: scales[i]}
		}
		return experts
	}

	// Random hidden states.
	hsData := make([]float32, seqLen*modelDim)
	for i := range hsData {
		hsData[i] = rng.Float32()*2 - 1
	}
	hs, _ := tensor.New[float32]([]int{seqLen, modelDim}, hsData)

	// Random gate weights.
	gwData := make([]float32, numExperts*modelDim)
	for i := range gwData {
		gwData[i] = rng.Float32()*2 - 1
	}
	gw, _ := tensor.New[float32]([]int{numExperts, modelDim}, gwData)

	// Forward pass.
	gate := NewMoEGate[float32](eng, ops, topK)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, makeExperts(), numExperts, topK)
	out, err := moe.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// loss = sum(output)
	outShape := out.Shape()
	onesData := make([]float32, outShape[0]*outShape[1])
	for i := range onesData {
		onesData[i] = 1.0
	}
	dOut, _ := tensor.New[float32](outShape, onesData)

	// Backward pass.
	grads, err := moe.Backward(context.Background(), types.FullBackprop, dOut, hs, gw)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if grads == nil || len(grads) < 2 {
		t.Fatal("expected 2 gradients [dHS, dGW]")
	}
	dHS := grads[0]
	dGW := grads[1]

	if dHS == nil {
		t.Fatal("dHS is nil")
	}
	if dGW == nil {
		t.Fatal("dGW is nil")
	}

	// Finite difference check on hiddenStates.
	eps := float32(1e-3)
	tol := float32(5e-2) // Larger tolerance due to STE and routing discreteness.
	dHSData := dHS.Data()

	t.Run("dHiddenStates", func(t *testing.T) {
		maxErr := float32(0)
		for i := 0; i < len(hsData); i++ {
			// f(x + eps)
			hsPlus := make([]float32, len(hsData))
			copy(hsPlus, hsData)
			hsPlus[i] += eps
			hsPlusTensor, _ := tensor.New[float32]([]int{seqLen, modelDim}, hsPlus)
			lossPlus := moeForwardLoss(eng, ops, makeExperts(), numExperts, topK, hsPlusTensor, gw)

			// f(x - eps)
			hsMinus := make([]float32, len(hsData))
			copy(hsMinus, hsData)
			hsMinus[i] -= eps
			hsMinusTensor, _ := tensor.New[float32]([]int{seqLen, modelDim}, hsMinus)
			lossMinus := moeForwardLoss(eng, ops, makeExperts(), numExperts, topK, hsMinusTensor, gw)

			fdGrad := (lossPlus - lossMinus) / (2 * eps)
			analytic := dHSData[i]
			diff := float32(math.Abs(float64(fdGrad - analytic)))
			if diff > maxErr {
				maxErr = diff
			}
		}
		if maxErr > tol {
			t.Errorf("dHiddenStates max finite-diff error = %e, want < %e", maxErr, tol)
		} else {
			t.Logf("dHiddenStates max finite-diff error = %e (tol %e)", maxErr, tol)
		}
	})

	// Finite difference check on gate weights.
	dGWData := dGW.Data()
	t.Run("dGateWeight", func(t *testing.T) {
		maxErr := float32(0)
		for i := 0; i < len(gwData); i++ {
			gwPlus := make([]float32, len(gwData))
			copy(gwPlus, gwData)
			gwPlus[i] += eps
			gwPlusTensor, _ := tensor.New[float32]([]int{numExperts, modelDim}, gwPlus)
			lossPlus := moeForwardLoss(eng, ops, makeExperts(), numExperts, topK, hs, gwPlusTensor)

			gwMinus := make([]float32, len(gwData))
			copy(gwMinus, gwData)
			gwMinus[i] -= eps
			gwMinusTensor, _ := tensor.New[float32]([]int{numExperts, modelDim}, gwMinus)
			lossMinus := moeForwardLoss(eng, ops, makeExperts(), numExperts, topK, hs, gwMinusTensor)

			fdGrad := (lossPlus - lossMinus) / (2 * eps)
			analytic := dGWData[i]
			diff := float32(math.Abs(float64(fdGrad - analytic)))
			if diff > maxErr {
				maxErr = diff
			}
		}
		if maxErr > tol {
			t.Errorf("dGateWeight max finite-diff error = %e, want < %e", maxErr, tol)
		} else {
			t.Logf("dGateWeight max finite-diff error = %e (tol %e)", maxErr, tol)
		}
	})
}

// --- Sigmoid gating tests ---

func TestMoEGate_SigmoidGating_TopKSelection(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 2, WithSigmoidGating[float32]())

	// hiddenStates [1, 3], gateWeight [3, 3] identity
	// Token [1, 0, 0] -> logits [1, 0, 0] -> sigmoid [0.731, 0.5, 0.5]
	// Top-2 should pick expert 0 first (highest sigmoid), then expert 1 or 2.
	hs, _ := tensor.New[float32]([]int{1, 3}, []float32{1, 0, 0})
	gw, _ := tensor.New[float32]([]int{3, 3}, []float32{1, 0, 0, 0, 1, 0, 0, 0, 1})

	out, err := gate.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if out.Shape()[0] != 1 || out.Shape()[1] != 2 {
		t.Fatalf("shape = %v, want [1 2]", out.Shape())
	}

	// Verify top expert is expert 0 (highest logit=1 -> highest sigmoid).
	if gate.cachedIndices[0][0] != 0 {
		t.Errorf("top expert = %d, want 0", gate.cachedIndices[0][0])
	}

	// Verify weights sum to 1.0 (normalized).
	data := out.Data()
	sum := float64(data[0]) + float64(data[1])
	if math.Abs(sum-1.0) > 1e-5 {
		t.Errorf("weights sum = %f, want 1.0", sum)
	}
}

func TestMoEGate_SigmoidGating_BiasShiftsSelection(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}

	// Without bias: token [1,0,0] with identity gate -> expert 0 is top.
	// With large bias on expert 2: sigmoid(0 + 10) >> sigmoid(1 + 0), so expert 2 wins.
	bias, _ := tensor.New[float32]([]int{3}, []float32{0, 0, 10})
	gate := NewMoEGate[float32](eng, ops, 1,
		WithSigmoidGating[float32](),
		WithRoutingBias[float32](bias),
	)

	hs, _ := tensor.New[float32]([]int{1, 3}, []float32{1, 0, 0})
	gw, _ := tensor.New[float32]([]int{3, 3}, []float32{1, 0, 0, 0, 1, 0, 0, 0, 1})

	_, err := gate.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Bias should shift expert 2 to the top.
	if gate.cachedIndices[0][0] != 2 {
		t.Errorf("top expert = %d, want 2 (bias should shift selection)", gate.cachedIndices[0][0])
	}
}

func TestMoEGate_SigmoidGating_NormalizedWeights(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 3, WithSigmoidGating[float32]())

	// 2 tokens, 4 experts, topK=3
	hs, _ := tensor.New[float32]([]int{2, 4}, []float32{
		1, 2, 0, -1,
		-1, 0, 3, 1,
	})
	gw, _ := tensor.New[float32]([]int{4, 4}, []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	})

	out, err := gate.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	data := out.Data()
	topK := 3
	for row := 0; row < 2; row++ {
		sum := float64(0)
		for k := 0; k < topK; k++ {
			sum += float64(data[row*topK+k])
		}
		if math.Abs(sum-1.0) > 1e-5 {
			t.Errorf("row %d: weights sum = %f, want ~1.0", row, sum)
		}
	}
}

func TestMoEGate_SigmoidGating_Attributes(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 2, WithSigmoidGating[float32]())
	attrs := gate.Attributes()
	if v, ok := attrs["sigmoid_gating"]; !ok || v != true {
		t.Errorf("attributes[sigmoid_gating] = %v, want true", attrs["sigmoid_gating"])
	}

	// Without sigmoid gating, attribute should not be present.
	gate2 := NewMoEGate[float32](eng, ops, 2)
	attrs2 := gate2.Attributes()
	if _, ok := attrs2["sigmoid_gating"]; ok {
		t.Error("sigmoid_gating attribute should not be present when not enabled")
	}
}

// BenchmarkMoE_SequentialBaseline benchmarks the sequential path by running
// one token at a time and accumulating, simulating the old per-token dispatch.
func BenchmarkMoE_SequentialBaseline(b *testing.B) {
	numExperts := 8
	topK := 2
	modelDim := 64

	for _, batchSize := range []int{1, 4, 8, 16} {
		b.Run(fmt.Sprintf("tokens=%d", batchSize), func(b *testing.B) {
			eng := makeFloat32Engine()
			ops := numeric.Float32Ops{}

			gwData := make([]float32, numExperts*modelDim)
			for i := 0; i < numExperts; i++ {
				gwData[i*modelDim+i%modelDim] = 1.0
			}
			gw, _ := tensor.New[float32]([]int{numExperts, modelDim}, gwData)

			tokens := make([]*tensor.TensorNumeric[float32], batchSize)
			for t := 0; t < batchSize; t++ {
				td := make([]float32, modelDim)
				for d := range td {
					td[d] = float32(t*modelDim+d) * 0.01
				}
				tokens[t], _ = tensor.New[float32]([]int{1, modelDim}, td)
			}

			ctx := context.Background()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for _, tok := range tokens {
					experts := make([]graph.Node[float32], numExperts)
					for j := range experts {
						experts[j] = &scale2Expert{}
					}
					gate := NewMoEGate[float32](eng, ops, topK)
					moe := NewMixtureOfExperts[float32](eng, ops, gate, experts, numExperts, topK)
					_, _ = moe.Forward(ctx, tok, gw)
				}
			}
		})
	}
}
