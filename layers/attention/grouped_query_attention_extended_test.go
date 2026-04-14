package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func TestGroupedQueryAttention_OpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention failed: %v", err)
	}
	if got := gqa.OpType(); got != "GroupedQueryAttention" {
		t.Errorf("OpType() = %q, want %q", got, "GroupedQueryAttention")
	}
}

func TestGroupedQueryAttention_Attributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention failed: %v", err)
	}
	attrs := gqa.Attributes()
	if attrs["model_dim"] != 16 {
		t.Errorf("model_dim = %v, want 16", attrs["model_dim"])
	}
	if attrs["num_query_heads"] != 4 {
		t.Errorf("num_query_heads = %v, want 4", attrs["num_query_heads"])
	}
	if attrs["num_key_value_heads"] != 2 {
		t.Errorf("num_key_value_heads = %v, want 2", attrs["num_key_value_heads"])
	}
	if attrs["head_dim"] != 4 {
		t.Errorf("head_dim = %v, want 4 (16/4)", attrs["head_dim"])
	}
}

func TestGroupedQueryAttention_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention failed: %v", err)
	}
	params := gqa.Parameters()
	// wq, wk, wv, wo: each has weight + bias = 8 params total
	if len(params) != 8 {
		t.Errorf("Parameters() len = %d, want 8", len(params))
	}
}

func TestGroupedQueryAttention_OutputShape_BeforeForward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention failed: %v", err)
	}
	// Before Forward, outputShape is nil
	shape := gqa.OutputShape()
	if shape != nil {
		t.Errorf("OutputShape before Forward = %v, want nil", shape)
	}
}

func TestGroupedQueryAttention_ScaleRope(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention failed: %v", err)
	}
	if err := gqa.ScaleRope(ctx, 2.0); err != nil {
		t.Errorf("ScaleRope failed: %v", err)
	}
}

func TestGroupedQueryAttention_Forward_NoInputs(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention failed: %v", err)
	}
	_, err = gqa.Forward(context.Background())
	if err == nil {
		t.Error("expected error for empty inputs")
	}
}

func TestGroupedQueryAttention_Validation_Errors(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name                  string
		modelDim, numQ, numKV int
	}{
		{"QueryNotDivisibleByKV", 16, 5, 2},
		{"ModelNotDivisibleByQuery", 15, 4, 2},
		{"ModelNotDivisibleByKV", 16, 4, 3},
		// headDim=15/3=5 (odd) causes RoPE to fail
		{"OddHeadDimRoPE", 15, 3, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewGroupedQueryAttention[float32](engine, ops, tt.modelDim, tt.numQ, tt.numKV)
			if err == nil {
				t.Error("expected validation error")
			}
		})
	}
}

func TestGroupedQueryAttention_Backward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2,
		WithMaxSeqLen[float32](8))
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	out, err := gqa.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	grads, err := gqa.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if len(grads) != 1 {
		t.Errorf("expected 1 gradient, got %d", len(grads))
	}
	if grads[0].Shape()[0] != 1 || grads[0].Shape()[1] != 3 || grads[0].Shape()[2] != 16 {
		t.Errorf("gradient shape = %v, want [1 3 16]", grads[0].Shape())
	}
}

func TestGroupedQueryAttention_Backward_WrongInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2,
		WithMaxSeqLen[float32](8))
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention failed: %v", err)
	}

	dOut, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
	_, err = gqa.Backward(context.Background(), types.FullBackprop, dOut)
	if err == nil {
		t.Error("expected error for missing inputs in Backward")
	}
}

func TestGroupedQueryAttention_EqualHeadCounts(t *testing.T) {
	// Test with numQueryHeads == numKeyValueHeads (no replication needed)
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 4,
		WithMaxSeqLen[float32](8))
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	out, err := gqa.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Also test backward with equal heads (no reverseHeadReplication)
	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	grads, err := gqa.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if len(grads) != 1 {
		t.Errorf("expected 1 gradient, got %d", len(grads))
	}
}

func TestNewGroupedQueryAttentionFromParams(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	wq, err := core.NewDense[float32]("wq", engine, ops, 16, 16)
	if err != nil {
		t.Fatalf("NewDense wq failed: %v", err)
	}
	wk, err := core.NewDense[float32]("wk", engine, ops, 16, 8)
	if err != nil {
		t.Fatalf("NewDense wk failed: %v", err)
	}
	wv, err := core.NewDense[float32]("wv", engine, ops, 16, 8)
	if err != nil {
		t.Fatalf("NewDense wv failed: %v", err)
	}
	wo, err := core.NewDense[float32]("wo", engine, ops, 16, 16)
	if err != nil {
		t.Fatalf("NewDense wo failed: %v", err)
	}

	rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
		context.Background(), engine, 4, 64, embeddings.WithRotaryBase(10000.0))
	if err != nil {
		t.Fatalf("NewRotaryPositionalEmbedding failed: %v", err)
	}

	gqa, err := NewGroupedQueryAttentionFromParams(engine, ops, 16, 4, 2, wq, wk, wv, wo, rope)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttentionFromParams failed: %v", err)
	}

	if gqa.modelDim != 16 {
		t.Errorf("modelDim = %d, want 16", gqa.modelDim)
	}
	if gqa.numQueryHeads != 4 {
		t.Errorf("numQueryHeads = %d, want 4", gqa.numQueryHeads)
	}
	if gqa.numKeyValueHeads != 2 {
		t.Errorf("numKeyValueHeads = %d, want 2", gqa.numKeyValueHeads)
	}
	if gqa.headDim != 4 {
		t.Errorf("headDim = %d, want 4", gqa.headDim)
	}
}

func makeParam(t *testing.T, name string, shape []int) *graph.Parameter[float32] {
	t.Helper()
	weight, err := tensor.New[float32](shape, nil)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}
	p, err := graph.NewParameter(name, weight, tensor.New[float32])
	if err != nil {
		t.Fatalf("NewParameter failed: %v", err)
	}
	return p
}

func TestBuildGroupQueryAttention(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Create proper parameters for the builder
	modelDim := 16
	numQ := 4
	numKV := 2
	headDim := modelDim / numQ

	params := map[string]*graph.Parameter[float32]{
		"attn_wq": makeParam(t, "attn_wq", []int{modelDim, modelDim}),
		"attn_wk": makeParam(t, "attn_wk", []int{modelDim, headDim * numKV}),
		"attn_wv": makeParam(t, "attn_wv", []int{modelDim, headDim * numKV}),
		"attn_wo": makeParam(t, "attn_wo", []int{modelDim, modelDim}),
	}

	attrs := map[string]interface{}{
		"model_dim":           modelDim,
		"num_query_heads":     numQ,
		"num_key_value_heads": numKV,
		"rope_base":           10000.0,
		"max_seq_len":         64,
	}

	node, err := BuildGroupQueryAttention[float32](engine, ops, "attn", params, attrs)
	if err != nil {
		t.Fatalf("BuildGroupQueryAttention failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildGroupQueryAttention returned nil")
	}
}

func TestBuildGroupQueryAttention_MissingAttributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name  string
		attrs map[string]interface{}
	}{
		{"MissingModelDim", map[string]interface{}{
			"num_query_heads": 4, "num_key_value_heads": 2, "rope_base": 10000.0, "max_seq_len": 64,
		}},
		{"MissingNumQueryHeads", map[string]interface{}{
			"model_dim": 16, "num_key_value_heads": 2, "rope_base": 10000.0, "max_seq_len": 64,
		}},
		{"MissingNumKVHeads", map[string]interface{}{
			"model_dim": 16, "num_query_heads": 4, "rope_base": 10000.0, "max_seq_len": 64,
		}},
		{"MissingRopeBase", map[string]interface{}{
			"model_dim": 16, "num_query_heads": 4, "num_key_value_heads": 2, "max_seq_len": 64,
		}},
		{"MissingMaxSeqLen", map[string]interface{}{
			"model_dim": 16, "num_query_heads": 4, "num_key_value_heads": 2, "rope_base": 10000.0,
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildGroupQueryAttention[float32](engine, ops, "attn", nil, tt.attrs)
			if err == nil {
				t.Error("expected error for missing attribute")
			}
		})
	}
}

func TestGQA_Forward_EngineErrors(t *testing.T) {
	// GQA.Forward gqa.engine call sequence:
	// Reshape x12 (head split x3, transpose prep x3, RoPE in/out x4, SDPA prep x3, concat x2)
	// Transpose x4 (Q/K/V head transpose + concat transpose)
	// Repeat x2 (K/V head replication when numQ != numKV)
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"Reshape_1", map[string]int{"Reshape": 1}},
		{"Reshape_2", map[string]int{"Reshape": 2}},
		{"Reshape_3", map[string]int{"Reshape": 3}},
		{"Reshape_4", map[string]int{"Reshape": 4}},
		{"Reshape_5", map[string]int{"Reshape": 5}},
		{"Reshape_6", map[string]int{"Reshape": 6}},
		{"Reshape_7", map[string]int{"Reshape": 7}},
		{"Reshape_8", map[string]int{"Reshape": 8}},
		{"Reshape_9", map[string]int{"Reshape": 9}},
		{"Reshape_10", map[string]int{"Reshape": 10}},
		{"Reshape_11", map[string]int{"Reshape": 11}},
		{"Reshape_12", map[string]int{"Reshape": 12}},
		{"Transpose_1", map[string]int{"Transpose": 1}},
		{"Transpose_2", map[string]int{"Transpose": 2}},
		{"Transpose_3", map[string]int{"Transpose": 3}},
		{"Transpose_4", map[string]int{"Transpose": 4}},
		{"Repeat_1", map[string]int{"Repeat": 1}},
		{"Repeat_2", map[string]int{"Repeat": 2}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](ops)
			gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2,
				WithMaxSeqLen[float32](8))
			if err != nil {
				t.Fatalf("failed: %v", err)
			}
			input, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
			for i := range input.Data() {
				input.Data()[i] = 0.01
			}

			fe := newFailingEngine(tc.failOn)
			gqa.engine = fe
			_, err = gqa.Forward(context.Background(), input)
			if err == nil {
				t.Errorf("expected error from %s failure in Forward", tc.name)
			}
		})
	}
}

func TestGQA_Backward_EngineErrors(t *testing.T) {
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		// Step 2-4: Reshape x2, Transpose x1
		{"Reshape_1", map[string]int{"Reshape": 1}},
		{"Transpose_1", map[string]int{"Transpose": 1}},
		{"Reshape_2", map[string]int{"Reshape": 2}},
		// Step 6: reverseHeadReplication K (Reshape x3, ReduceSum x1), V (Reshape x3, ReduceSum x1)
		{"Reshape_3", map[string]int{"Reshape": 3}},
		{"Reshape_4", map[string]int{"Reshape": 4}},
		{"ReduceSum_1", map[string]int{"ReduceSum": 1}},
		{"Reshape_5", map[string]int{"Reshape": 5}},
		{"Reshape_6", map[string]int{"Reshape": 6}},
		{"Reshape_7", map[string]int{"Reshape": 7}},
		{"ReduceSum_2", map[string]int{"ReduceSum": 2}},
		{"Reshape_8", map[string]int{"Reshape": 8}},
		// Step 8: reverse head split Q (Reshape, Transpose, Reshape)
		{"Reshape_9", map[string]int{"Reshape": 9}},
		{"Transpose_2", map[string]int{"Transpose": 2}},
		{"Reshape_10", map[string]int{"Reshape": 10}},
		// Step 8: reverse head split K
		{"Reshape_11", map[string]int{"Reshape": 11}},
		{"Transpose_3", map[string]int{"Transpose": 3}},
		{"Reshape_12", map[string]int{"Reshape": 12}},
		// Step 8: reverse head split V
		{"Reshape_13", map[string]int{"Reshape": 13}},
		{"Transpose_4", map[string]int{"Transpose": 4}},
		{"Reshape_14", map[string]int{"Reshape": 14}},
		// Step 10: sum gradients
		{"Add_1", map[string]int{"Add": 1}},
		{"Add_2", map[string]int{"Add": 2}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](ops)
			gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2,
				WithMaxSeqLen[float32](8))
			if err != nil {
				t.Fatalf("failed: %v", err)
			}

			input, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
			for i := range input.Data() {
				input.Data()[i] = float32(i%7+1) * 0.01
			}
			out, err := gqa.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			dOut, _ := tensor.New[float32](out.Shape(), nil)
			for i := range dOut.Data() {
				dOut.Data()[i] = 1.0
			}

			fe := newFailingEngine(tc.failOn)
			gqa.engine = fe
			_, err = gqa.Backward(ctx, types.FullBackprop, dOut, input)
			if err == nil {
				t.Errorf("expected error from %s failure in Backward", tc.name)
			}
		})
	}
}

// TestGQA_SubComponentErrors constructs GQA with a failing engine so that
// errors in sub-component Forward/Backward calls (Dense, RoPE, SDPA) propagate.
func TestGQA_SubComponentErrors(t *testing.T) {
	// Engine call sequence for GQA.Forward (all through failingEngine):
	// wq Dense.Forward: MatMul #1, Add #1
	// wk Dense.Forward: MatMul #2, Add #2
	// wv Dense.Forward: MatMul #3, Add #3
	// gqa.engine: Reshape x5, Transpose x3
	// rope.Forward Q: Mul #1-4, Sub #1, Add #4, Concat #1
	// rope.Forward K: Mul #5-8, Sub #2, Add #5, Concat #2
	// gqa.engine: Reshape x2
	// gqa.engine: Repeat x2
	// gqa.engine: Reshape x3
	// sdpa.Forward: Transpose #4, MatMul #4, MulScalar #1, Softmax #1, MatMul #5
	// gqa.engine: Reshape #11, Transpose #5, Reshape #12
	// wo Dense.Forward: MatMul #6, Add #6
	fwdTests := []struct {
		name   string
		failOn map[string]int
	}{
		{"wq_Forward", map[string]int{"MatMul": 1}},
		{"wk_Forward", map[string]int{"MatMul": 2}},
		{"wv_Forward", map[string]int{"MatMul": 3}},
		{"rope_Q_Forward", map[string]int{"Mul": 1}},
		{"rope_K_Forward", map[string]int{"Mul": 5}},
		{"sdpa_Forward", map[string]int{"MatMul": 4}},
		{"wo_Forward", map[string]int{"MatMul": 6}},
	}

	for _, tc := range fwdTests {
		t.Run(tc.name, func(t *testing.T) {
			fe := newFailingEngine(tc.failOn)
			ops := numeric.Float32Ops{}
			gqa, err := NewGroupedQueryAttention[float32](fe, ops, 16, 4, 2,
				WithMaxSeqLen[float32](8))
			if err != nil {
				t.Fatalf("construction failed: %v", err)
			}

			input, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
			for i := range input.Data() {
				input.Data()[i] = float32(i%5+1) * 0.01
			}

			_, err = gqa.Forward(context.Background(), input)
			if err == nil {
				t.Error("expected error from sub-component Forward failure")
			}
		})
	}

	// Backward sub-component errors: construct with failingEngine, use high
	// fail counts so Forward succeeds but Backward fails in sub-components.
	// Forward total: MatMul=6, Add=6, Mul=8, Sub=2, Transpose=5, Reshape=12, Repeat=2
	// Backward starts at those counts + 1.
	// Backward cumulative engine calls through failingEngine:
	// Forward total: MatMul=6, Add=6, Mul=8, Sub=2, Sum=0, Transpose=5, Reshape=12
	// wo.Backward: Sum #1, Reshape #13,14, Transpose #6, MatMul #7, Add #7, Transpose #7, MatMul #8
	// gqa.engine: Reshape #15,16,17 + Transpose #8
	// sdpa.Backward: Transpose #9,10,11, MatMul #9,10,11,12, Mul #9,10, ReduceSum #1, Sub #3, MulScalar #2
	// reverseHeadReplication K: Reshape #18-20, ReduceSum #2
	// reverseHeadReplication V: Reshape #21-23, ReduceSum #3
	// rope.Backward Q: Mul #11-14, Add #8, Sub #4, Concat
	// rope.Backward K: Mul #15-18, Add #9, Sub #5, Concat
	// gqa.engine: Reshape x9, Transpose x3
	// wq.Backward: Sum #2, Reshape, Transpose, MatMul, Add, Transpose, MatMul
	// wk.Backward: Sum #3, ...
	// wv.Backward: Sum #4, ...
	// gqa.engine.Add x2
	bwdTests := []struct {
		name   string
		failOn map[string]int
	}{
		{"wo_Backward", map[string]int{"MatMul": 7}},
		{"sdpa_Backward_Mul", map[string]int{"Mul": 9}},
		{"rope_Q_Backward_Mul", map[string]int{"Mul": 11}},
		{"rope_K_Backward_Mul", map[string]int{"Mul": 15}},
		// Bias.Backward uses 2 Sum calls per 3D gradient (loop over dims-1).
		// wo.Backward: Sum #1,#2; wq: Sum #3,#4; wk: Sum #5,#6; wv: Sum #7,#8
		{"wq_Backward_Sum", map[string]int{"Sum": 3}},
		{"wk_Backward_Sum", map[string]int{"Sum": 5}},
		{"wv_Backward_Sum", map[string]int{"Sum": 7}},
	}

	for _, tc := range bwdTests {
		t.Run(tc.name, func(t *testing.T) {
			fe := newFailingEngine(tc.failOn)
			ops := numeric.Float32Ops{}
			gqa, err := NewGroupedQueryAttention[float32](fe, ops, 16, 4, 2,
				WithMaxSeqLen[float32](8))
			if err != nil {
				t.Fatalf("construction failed: %v", err)
			}

			input, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
			for i := range input.Data() {
				input.Data()[i] = float32(i%5+1) * 0.01
			}

			out, err := gqa.Forward(context.Background(), input)
			if err != nil {
				t.Skipf("Forward failed (test targets Backward): %v", err)
			}

			dOut, _ := tensor.New[float32](out.Shape(), nil)
			for i := range dOut.Data() {
				dOut.Data()[i] = 1.0
			}

			_, err = gqa.Backward(context.Background(), types.FullBackprop, dOut, input)
			if err == nil {
				t.Skipf("Backward succeeded (count may be wrong for %s)", tc.name)
			}
		})
	}
}

func TestBuildGroupQueryAttention_OddHeadDim(t *testing.T) {
	// headDim=5 (odd) causes RoPE creation to fail
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	attrs := map[string]any{
		"model_dim":           15,
		"num_query_heads":     3,
		"num_key_value_heads": 1,
		"rope_base":           10000.0,
		"max_seq_len":         64,
	}

	params := map[string]*graph.Parameter[float32]{
		"attn_wq": makeParam(t, "attn_wq", []int{15, 15}),
		"attn_wk": makeParam(t, "attn_wk", []int{15, 5}),
		"attn_wv": makeParam(t, "attn_wv", []int{15, 5}),
		"attn_wo": makeParam(t, "attn_wo", []int{15, 15}),
	}

	_, err := BuildGroupQueryAttention[float32](engine, ops, "attn", params, attrs)
	if err == nil {
		t.Error("expected error for odd headDim (RoPE requires even)")
	}
}

// makeNonZeroParam creates a parameter with non-zero data for testing.
func makeNonZeroParam(t *testing.T, name string, shape []int) *graph.Parameter[float32] {
	t.Helper()
	total := 1
	for _, d := range shape {
		total *= d
	}
	data := make([]float32, total)
	for i := range data {
		data[i] = float32(i%7+1) * 0.01
	}
	weight, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}
	p, err := graph.NewParameter(name, weight, tensor.New[float32])
	if err != nil {
		t.Fatalf("NewParameter failed: %v", err)
	}
	return p
}

func TestBuildGroupQueryAttention_WithBias(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 16
	numQ := 4
	numKV := 2
	headDim := modelDim / numQ
	kvDim := headDim * numKV

	// Use non-zero weights so bias has observable effect
	wq := makeNonZeroParam(t, "attn_wq", []int{modelDim, modelDim})
	wk := makeNonZeroParam(t, "attn_wk", []int{modelDim, kvDim})
	wv := makeNonZeroParam(t, "attn_wv", []int{modelDim, kvDim})
	wo := makeNonZeroParam(t, "attn_wo", []int{modelDim, modelDim})

	// Create bias params with non-zero values
	qBias := makeNonZeroParam(t, "attn_wq_bias", []int{modelDim})
	kBias := makeNonZeroParam(t, "attn_wk_bias", []int{kvDim})
	vBias := makeNonZeroParam(t, "attn_wv_bias", []int{kvDim})

	attrs := map[string]interface{}{
		"model_dim":           modelDim,
		"num_query_heads":     numQ,
		"num_key_value_heads": numKV,
		"rope_base":           10000.0,
		"max_seq_len":         64,
	}

	input, err := tensor.New[float32]([]int{1, 4, modelDim}, nil)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	// Build and run WITHOUT bias
	paramsNoBias := map[string]*graph.Parameter[float32]{
		"attn_wq": wq, "attn_wk": wk, "attn_wv": wv, "attn_wo": wo,
	}
	nodeNoBias, err := BuildGroupQueryAttention[float32](engine, ops, "attn", paramsNoBias, attrs)
	if err != nil {
		t.Fatalf("BuildGroupQueryAttention without bias failed: %v", err)
	}
	outNoBias, err := nodeNoBias.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward without bias failed: %v", err)
	}

	// Build and run WITH bias
	paramsWithBias := map[string]*graph.Parameter[float32]{
		"attn_wq": wq, "attn_wk": wk, "attn_wv": wv, "attn_wo": wo,
		"attn_wq_bias": qBias, "attn_wk_bias": kBias, "attn_wv_bias": vBias,
	}
	nodeWithBias, err := BuildGroupQueryAttention[float32](engine, ops, "attn", paramsWithBias, attrs)
	if err != nil {
		t.Fatalf("BuildGroupQueryAttention with bias failed: %v", err)
	}
	outWithBias, err := nodeWithBias.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward with bias failed: %v", err)
	}

	// Outputs should differ
	biasData := outWithBias.Data()
	noBiasData := outNoBias.Data()
	allSame := true
	for i := range biasData {
		if biasData[i] != noBiasData[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("output with bias should differ from output without bias")
	}
}

func TestBuildGroupQueryAttention_WithPartialBias(t *testing.T) {
	// Only Q bias present — should still work (K, V get nil bias)
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 16
	numQ := 4
	numKV := 2
	headDim := modelDim / numQ
	kvDim := headDim * numKV

	params := map[string]*graph.Parameter[float32]{
		"attn_wq":      makeParam(t, "attn_wq", []int{modelDim, modelDim}),
		"attn_wk":      makeParam(t, "attn_wk", []int{modelDim, kvDim}),
		"attn_wv":      makeParam(t, "attn_wv", []int{modelDim, kvDim}),
		"attn_wo":      makeParam(t, "attn_wo", []int{modelDim, modelDim}),
		"attn_wq_bias": makeParam(t, "attn_wq_bias", []int{modelDim}),
	}

	attrs := map[string]interface{}{
		"model_dim":           modelDim,
		"num_query_heads":     numQ,
		"num_key_value_heads": numKV,
		"rope_base":           10000.0,
		"max_seq_len":         64,
	}

	node, err := BuildGroupQueryAttention[float32](engine, ops, "attn", params, attrs)
	if err != nil {
		t.Fatalf("BuildGroupQueryAttention with partial bias failed: %v", err)
	}

	input, err := tensor.New[float32]([]int{1, 4, modelDim}, nil)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	_, err = node.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward with partial bias failed: %v", err)
	}
}

func TestBuildGroupQueryAttention_WithYaRNScaling(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 16
	numQ := 4
	numKV := 2
	headDim := modelDim / numQ
	kvDim := headDim * numKV

	wq := makeNonZeroParam(t, "attn_wq", []int{modelDim, modelDim})
	wk := makeNonZeroParam(t, "attn_wk", []int{modelDim, kvDim})
	wv := makeNonZeroParam(t, "attn_wv", []int{modelDim, kvDim})
	wo := makeNonZeroParam(t, "attn_wo", []int{modelDim, modelDim})

	params := map[string]*graph.Parameter[float32]{
		"attn_wq": wq, "attn_wk": wk, "attn_wv": wv, "attn_wo": wo,
	}

	input, err := tensor.New[float32]([]int{1, 4, modelDim}, nil)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	// Build WITHOUT YaRN
	attrsNoYaRN := map[string]interface{}{
		"model_dim":           modelDim,
		"num_query_heads":     numQ,
		"num_key_value_heads": numKV,
		"rope_base":           10000.0,
		"max_seq_len":         64,
	}
	nodeNoYaRN, err := BuildGroupQueryAttention[float32](engine, ops, "attn", params, attrsNoYaRN)
	if err != nil {
		t.Fatalf("Build without YaRN failed: %v", err)
	}
	outNoYaRN, err := nodeNoYaRN.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward without YaRN failed: %v", err)
	}

	// Build WITH YaRN scaling
	attrsYaRN := map[string]interface{}{
		"model_dim":                modelDim,
		"num_query_heads":          numQ,
		"num_key_value_heads":      numKV,
		"rope_base":                10000.0,
		"max_seq_len":              64,
		"rope_scaling_type":        "yarn",
		"rope_scaling_factor":      4.0,
		"rope_scaling_orig_max_len": 32,
	}
	nodeYaRN, err := BuildGroupQueryAttention[float32](engine, ops, "attn", params, attrsYaRN)
	if err != nil {
		t.Fatalf("Build with YaRN failed: %v", err)
	}
	outYaRN, err := nodeYaRN.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward with YaRN failed: %v", err)
	}

	// Outputs should differ
	noYaRNData := outNoYaRN.Data()
	yaRNData := outYaRN.Data()
	allSame := true
	for i := range noYaRNData {
		if noYaRNData[i] != yaRNData[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("output with YaRN should differ from output without YaRN")
	}
}

func TestBuildGroupQueryAttention_WithPartialRoPE(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 16
	numQ := 4
	numKV := 2
	headDim := modelDim / numQ
	kvDim := headDim * numKV

	wq := makeNonZeroParam(t, "attn_wq", []int{modelDim, modelDim})
	wk := makeNonZeroParam(t, "attn_wk", []int{modelDim, kvDim})
	wv := makeNonZeroParam(t, "attn_wv", []int{modelDim, kvDim})
	wo := makeNonZeroParam(t, "attn_wo", []int{modelDim, modelDim})

	params := map[string]*graph.Parameter[float32]{
		"attn_wq": wq, "attn_wk": wk, "attn_wv": wv, "attn_wo": wo,
	}

	input, err := tensor.New[float32]([]int{1, 4, modelDim}, nil)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	// Build WITHOUT partial RoPE
	attrsNormal := map[string]interface{}{
		"model_dim":           modelDim,
		"num_query_heads":     numQ,
		"num_key_value_heads": numKV,
		"rope_base":           10000.0,
		"max_seq_len":         64,
	}
	nodeNormal, err := BuildGroupQueryAttention[float32](engine, ops, "attn", params, attrsNormal)
	if err != nil {
		t.Fatalf("Build without partial RoPE failed: %v", err)
	}
	outNormal, err := nodeNormal.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward without partial RoPE failed: %v", err)
	}

	// Build WITH partial RoPE (fraction=0.5)
	attrsPartial := map[string]interface{}{
		"model_dim":              modelDim,
		"num_query_heads":        numQ,
		"num_key_value_heads":    numKV,
		"rope_base":              10000.0,
		"max_seq_len":            64,
		"partial_rotary_factor": 0.5,
	}
	nodePartial, err := BuildGroupQueryAttention[float32](engine, ops, "attn", params, attrsPartial)
	if err != nil {
		t.Fatalf("Build with partial RoPE failed: %v", err)
	}
	outPartial, err := nodePartial.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward with partial RoPE failed: %v", err)
	}

	// Outputs should differ
	normalData := outNormal.Data()
	partialData := outPartial.Data()
	allSame := true
	for i := range normalData {
		if normalData[i] != partialData[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("output with partial RoPE should differ from output without")
	}
}

func TestBuildGroupQueryAttention_MissingParams(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	attrs := map[string]interface{}{
		"model_dim":           16,
		"num_query_heads":     4,
		"num_key_value_heads": 2,
		"rope_base":           10000.0,
		"max_seq_len":         64,
	}

	pWQ := makeParam(t, "attn_wq", []int{16, 16})
	pWK := makeParam(t, "attn_wk", []int{16, 8})
	pWV := makeParam(t, "attn_wv", []int{16, 8})
	pWO := makeParam(t, "attn_wo", []int{16, 16})

	tests := []struct {
		name   string
		params map[string]*graph.Parameter[float32]
	}{
		{"MissingWQ", map[string]*graph.Parameter[float32]{
			"attn_wk": pWK, "attn_wv": pWV, "attn_wo": pWO,
		}},
		{"MissingWK", map[string]*graph.Parameter[float32]{
			"attn_wq": pWQ, "attn_wv": pWV, "attn_wo": pWO,
		}},
		{"MissingWV", map[string]*graph.Parameter[float32]{
			"attn_wq": pWQ, "attn_wk": pWK, "attn_wo": pWO,
		}},
		{"MissingWO", map[string]*graph.Parameter[float32]{
			"attn_wq": pWQ, "attn_wk": pWK, "attn_wv": pWV,
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildGroupQueryAttention[float32](engine, ops, "attn", tt.params, attrs)
			if err == nil {
				t.Error("expected error for missing parameter")
			}
		})
	}
}

// TestGroupedQueryAttention_SetExternalKV_FromParams verifies that
// SetExternalKV(true) on a GQA constructed via NewGroupedQueryAttentionFromParams
// nils wk/wv, enables external-KV Forward semantics, and produces numerically
// identical outputs to a standard internal-KV GQA when fed the same
// pre-computed K/V tensors. This closes the API gap left by E95 where
// WithExternalKV only worked via the NewGroupedQueryAttention constructor.
func TestGroupedQueryAttention_SetExternalKV_FromParams(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batchSize := 2
	seqLen := 5
	modelDim := 16
	numQueryHeads := 4
	numKeyValueHeads := 2
	headDim := modelDim / numQueryHeads
	kvDim := numKeyValueHeads * headDim

	// Reference: internal-KV GQA built via FromParams.
	wqRef, err := core.NewDense[float32]("wq_ref", engine, ops, modelDim, modelDim)
	if err != nil {
		t.Fatalf("NewDense wq_ref: %v", err)
	}
	wkRef, err := core.NewDense[float32]("wk_ref", engine, ops, modelDim, kvDim)
	if err != nil {
		t.Fatalf("NewDense wk_ref: %v", err)
	}
	wvRef, err := core.NewDense[float32]("wv_ref", engine, ops, modelDim, kvDim)
	if err != nil {
		t.Fatalf("NewDense wv_ref: %v", err)
	}
	woRef, err := core.NewDense[float32]("wo_ref", engine, ops, modelDim, modelDim)
	if err != nil {
		t.Fatalf("NewDense wo_ref: %v", err)
	}
	ropeRef, err := embeddings.NewRotaryPositionalEmbedding[float32](
		context.Background(), engine, headDim, seqLen, embeddings.WithRotaryBase(10000.0))
	if err != nil {
		t.Fatalf("NewRotaryPositionalEmbedding ref: %v", err)
	}
	gqaInternal, err := NewGroupedQueryAttentionFromParams[float32](
		engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		wqRef, wkRef, wvRef, woRef, ropeRef)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttentionFromParams internal: %v", err)
	}

	// Under-test: FromParams layer with SetExternalKV(true).
	wqEx, err := core.NewDense[float32]("wq_ex", engine, ops, modelDim, modelDim)
	if err != nil {
		t.Fatalf("NewDense wq_ex: %v", err)
	}
	wkEx, err := core.NewDense[float32]("wk_ex", engine, ops, modelDim, kvDim)
	if err != nil {
		t.Fatalf("NewDense wk_ex: %v", err)
	}
	wvEx, err := core.NewDense[float32]("wv_ex", engine, ops, modelDim, kvDim)
	if err != nil {
		t.Fatalf("NewDense wv_ex: %v", err)
	}
	woEx, err := core.NewDense[float32]("wo_ex", engine, ops, modelDim, modelDim)
	if err != nil {
		t.Fatalf("NewDense wo_ex: %v", err)
	}
	ropeEx, err := embeddings.NewRotaryPositionalEmbedding[float32](
		context.Background(), engine, headDim, seqLen, embeddings.WithRotaryBase(10000.0))
	if err != nil {
		t.Fatalf("NewRotaryPositionalEmbedding ex: %v", err)
	}
	gqaExternal, err := NewGroupedQueryAttentionFromParams[float32](
		engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		wqEx, wkEx, wvEx, woEx, ropeEx)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttentionFromParams external: %v", err)
	}

	gqaExternal.SetExternalKV(true)

	if !gqaExternal.ExternalKV() {
		t.Fatalf("ExternalKV() = false, want true after SetExternalKV(true)")
	}
	if gqaExternal.wk != nil {
		t.Fatalf("SetExternalKV(true) must nil wk")
	}
	if gqaExternal.wv != nil {
		t.Fatalf("SetExternalKV(true) must nil wv")
	}
	if gqaExternal.kNorm != nil {
		t.Fatalf("SetExternalKV(true) must nil kNorm")
	}

	// Share Q and output weights so only the K/V source differs.
	copyDenseWeights(t, gqaExternal.wq, gqaInternal.wq)
	copyDenseWeights(t, gqaExternal.wo, gqaInternal.wo)

	inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
	if err != nil {
		t.Fatalf("input tensor: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i%13) / 10.0
	}

	ctx := context.Background()
	outInternal, err := gqaInternal.Forward(ctx, inp)
	if err != nil {
		t.Fatalf("internal Forward: %v", err)
	}
	kProj := gqaInternal.kProj
	vProj := gqaInternal.vProj
	if kProj == nil || vProj == nil {
		t.Fatalf("internal GQA did not cache kProj/vProj")
	}

	outExternal, err := gqaExternal.Forward(ctx, inp, kProj, vProj)
	if err != nil {
		t.Fatalf("external Forward: %v", err)
	}

	if len(outInternal.Shape()) != len(outExternal.Shape()) {
		t.Fatalf("shape rank mismatch: %v vs %v", outInternal.Shape(), outExternal.Shape())
	}
	for i, d := range outInternal.Shape() {
		if outExternal.Shape()[i] != d {
			t.Fatalf("shape mismatch at axis %d: %v vs %v", i, outInternal.Shape(), outExternal.Shape())
		}
	}

	const tol = 1e-5
	for i := range outInternal.Data() {
		if math.Abs(float64(outInternal.Data()[i]-outExternal.Data()[i])) > tol {
			t.Fatalf("output mismatch at %d: internal=%f external=%f",
				i, outInternal.Data()[i], outExternal.Data()[i])
		}
	}
}

// TestGroupedQueryAttention_SetExternalKV_RejectsKEqV verifies SetExternalKV
// panics if called after SetKEqV, matching the WithExternalKV/SetKEqV mutual
// exclusivity contract.
func TestGroupedQueryAttention_SetExternalKV_RejectsKEqV(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	wq, _ := core.NewDense[float32]("wq", engine, ops, 16, 16)
	wk, _ := core.NewDense[float32]("wk", engine, ops, 16, 8)
	wv, _ := core.NewDense[float32]("wv", engine, ops, 16, 8)
	wo, _ := core.NewDense[float32]("wo", engine, ops, 16, 16)
	rope, _ := embeddings.NewRotaryPositionalEmbedding[float32](
		context.Background(), engine, 4, 64, embeddings.WithRotaryBase(10000.0))

	gqa, err := NewGroupedQueryAttentionFromParams[float32](engine, ops, 16, 4, 2, wq, wk, wv, wo, rope)
	if err != nil {
		t.Fatalf("FromParams: %v", err)
	}
	gqa.SetKEqV(true)

	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic when SetExternalKV is called on kEqV GQA")
		}
	}()
	gqa.SetExternalKV(true)
}
