package normalization

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func makeGainParam(t *testing.T, name string, dim int) *graph.Parameter[float32] {
	t.Helper()
	data := make([]float32, dim)
	for i := range data {
		data[i] = 1.0
	}
	val, err := tensor.New[float32]([]int{dim}, data)
	if err != nil {
		t.Fatal(err)
	}
	p, err := graph.NewParameter[float32](name, val, tensor.New[float32])
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestBuildRMSNorm(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildRMSNorm[float32](engine, ops, "test", params, attrs)
	if err != nil {
		t.Fatalf("BuildRMSNorm failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildRMSNorm returned nil")
	}
}

func TestBuildRMSNorm_MissingGain(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	_, err := BuildRMSNorm[float32](engine, ops, "test", map[string]*graph.Parameter[float32]{}, map[string]any{"epsilon": float64(1e-6)})
	if err == nil {
		t.Error("expected error for missing gain parameter")
	}
}

func TestBuildRMSNorm_MissingEpsilon(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	_, err := BuildRMSNorm[float32](engine, ops, "test", params, map[string]any{})
	if err == nil {
		t.Error("expected error for missing epsilon")
	}
}

func TestBuildRMSNorm_Float32Epsilon(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	attrs := map[string]any{"epsilon": float32(1e-6)}

	node, err := BuildRMSNorm[float32](engine, ops, "test", params, attrs)
	if err != nil {
		t.Fatalf("BuildRMSNorm with float32 epsilon failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildRMSNorm returned nil")
	}
}

func TestBuildRMSNorm_BadEpsilonType(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	attrs := map[string]any{"epsilon": "bad"}

	_, err := BuildRMSNorm[float32](engine, ops, "test", params, attrs)
	if err == nil {
		t.Error("expected error for bad epsilon type")
	}
}

func TestBuildSimplifiedLayerNormalization(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 1: name_gain
	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSimplifiedLayerNormalization[float32](engine, ops, "test", params, attrs)
	if err != nil {
		t.Fatalf("BuildSimplifiedLayerNormalization failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSimplifiedLayerNormalization_DotWeightPattern(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 2: path-like name with .LayerNorm suffix -> dot.weight
	params := map[string]*graph.Parameter[float32]{
		"model.layers.0.input_layernorm.weight": makeGainParam(t, "w", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSimplifiedLayerNormalization[float32](engine, ops, "/model/layers.0/input_layernorm/LayerNorm", params, attrs)
	if err != nil {
		t.Fatalf("BuildSimplifiedLayerNormalization dot-weight pattern failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSimplifiedLayerNormalization_SimplifiedSuffix(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 2: SimplifiedLayerNormalization suffix -> .weight
	params := map[string]*graph.Parameter[float32]{
		"model.norm.weight": makeGainParam(t, "w", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSimplifiedLayerNormalization[float32](engine, ops, "/model/norm/SimplifiedLayerNormalization", params, attrs)
	if err != nil {
		t.Fatalf("BuildSimplifiedLayerNormalization SimplifiedLayerNormalization suffix failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSimplifiedLayerNormalization_WeightSuffix(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 3: just .weight suffix
	params := map[string]*graph.Parameter[float32]{
		"model.custom.weight": makeGainParam(t, "w", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSimplifiedLayerNormalization[float32](engine, ops, "/model/custom", params, attrs)
	if err != nil {
		t.Fatalf("BuildSimplifiedLayerNormalization weight suffix failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSimplifiedLayerNormalization_LayernormWeight(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 4: .layernorm.weight pattern
	params := map[string]*graph.Parameter[float32]{
		"model.attn.q_norm.layernorm.weight": makeGainParam(t, "w", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSimplifiedLayerNormalization[float32](engine, ops, "/model/attn/q_norm/SimplifiedLayerNormalization", params, attrs)
	if err != nil {
		t.Fatalf("BuildSimplifiedLayerNormalization layernorm.weight failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSimplifiedLayerNormalization_MissingGain(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	_, err := BuildSimplifiedLayerNormalization[float32](engine, ops, "test", map[string]*graph.Parameter[float32]{}, map[string]any{"epsilon": float64(1e-6)})
	if err == nil {
		t.Error("expected error for missing gain")
	}
}

func TestBuildSimplifiedLayerNormalization_MissingEpsilon(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	_, err := BuildSimplifiedLayerNormalization[float32](engine, ops, "test", params, map[string]any{})
	if err == nil {
		t.Error("expected error for missing epsilon")
	}
}

func TestBuildSimplifiedLayerNormalization_Float32Epsilon(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	attrs := map[string]any{"epsilon": float32(1e-6)}

	node, err := BuildSimplifiedLayerNormalization[float32](engine, ops, "test", params, attrs)
	if err != nil {
		t.Fatalf("float32 epsilon failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSimplifiedLayerNormalization_BadEpsilonType(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	_, err := BuildSimplifiedLayerNormalization[float32](engine, ops, "test", params, map[string]any{"epsilon": "bad"})
	if err == nil {
		t.Error("expected error for bad epsilon type")
	}
}

func TestBuildSkipSimplifiedLayerNormalization(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 1: name_gain
	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSkipSimplifiedLayerNormalization[float32](engine, ops, "test", params, attrs)
	if err != nil {
		t.Fatalf("BuildSkipSimplifiedLayerNormalization failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSkipSimplifiedLayerNormalization_SkipLayerNormSuffix(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 2: SkipLayerNorm suffix -> .weight
	params := map[string]*graph.Parameter[float32]{
		"model.layers.0.pre_feedforward_layernorm.weight": makeGainParam(t, "w", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSkipSimplifiedLayerNormalization[float32](engine, ops, "/model/layers.0/pre_feedforward_layernorm/SkipLayerNorm", params, attrs)
	if err != nil {
		t.Fatalf("SkipLayerNorm suffix failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSkipSimplifiedLayerNormalization_SkipSimplifiedSuffix(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 2: SkipSimplifiedLayerNormalization suffix -> .weight
	params := map[string]*graph.Parameter[float32]{
		"model.norm.weight": makeGainParam(t, "w", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSkipSimplifiedLayerNormalization[float32](engine, ops, "/model/norm/SkipSimplifiedLayerNormalization", params, attrs)
	if err != nil {
		t.Fatalf("SkipSimplifiedLayerNormalization suffix failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSkipSimplifiedLayerNormalization_WeightSuffix(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 3: .weight suffix
	params := map[string]*graph.Parameter[float32]{
		"model.custom.weight": makeGainParam(t, "w", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSkipSimplifiedLayerNormalization[float32](engine, ops, "/model/custom", params, attrs)
	if err != nil {
		t.Fatalf("weight suffix failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSkipSimplifiedLayerNormalization_LayernormWeight(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 4: .layernorm.weight with SkipLayerNorm suffix
	params := map[string]*graph.Parameter[float32]{
		"model.norm.layernorm.weight": makeGainParam(t, "w", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSkipSimplifiedLayerNormalization[float32](engine, ops, "/model/norm/SkipLayerNorm", params, attrs)
	if err != nil {
		t.Fatalf("layernorm.weight pattern failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSkipSimplifiedLayerNormalization_LayernormWeightSkipSimplified(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Pattern 4: .layernorm.weight with SkipSimplifiedLayerNormalization suffix
	params := map[string]*graph.Parameter[float32]{
		"model.norm.layernorm.weight": makeGainParam(t, "w", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildSkipSimplifiedLayerNormalization[float32](engine, ops, "/model/norm/SkipSimplifiedLayerNormalization", params, attrs)
	if err != nil {
		t.Fatalf("layernorm.weight SkipSimplified pattern failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSkipSimplifiedLayerNormalization_MissingGain(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	_, err := BuildSkipSimplifiedLayerNormalization[float32](engine, ops, "test", map[string]*graph.Parameter[float32]{}, map[string]any{"epsilon": float64(1e-6)})
	if err == nil {
		t.Error("expected error for missing gain")
	}
}

func TestBuildSkipSimplifiedLayerNormalization_MissingEpsilon(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	_, err := BuildSkipSimplifiedLayerNormalization[float32](engine, ops, "test", params, map[string]any{})
	if err == nil {
		t.Error("expected error for missing epsilon")
	}
}

func TestBuildSkipSimplifiedLayerNormalization_Float32Epsilon(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	node, err := BuildSkipSimplifiedLayerNormalization[float32](engine, ops, "test", params, map[string]any{"epsilon": float32(1e-6)})
	if err != nil {
		t.Fatalf("float32 epsilon failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildSkipSimplifiedLayerNormalization_BadEpsilonType(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_gain": makeGainParam(t, "test_gain", 4),
	}
	_, err := BuildSkipSimplifiedLayerNormalization[float32](engine, ops, "test", params, map[string]any{"epsilon": "bad"})
	if err == nil {
		t.Error("expected error for bad epsilon type")
	}
}

// --- SimplifiedLayerNormalization accessor tests ---

func TestSimplifiedLayerNormalization_OpType(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gain, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	sln, err := NewSimplifiedLayerNormalization(engine, ops, gain, float32(1e-6))
	if err != nil {
		t.Fatal(err)
	}
	if sln.OpType() != "SimplifiedLayerNormalization" {
		t.Errorf("OpType() = %q, want SimplifiedLayerNormalization", sln.OpType())
	}
}

func TestSimplifiedLayerNormalization_Attributes(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gain, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	sln, err := NewSimplifiedLayerNormalization(engine, ops, gain, float32(1e-6))
	if err != nil {
		t.Fatal(err)
	}
	attrs := sln.Attributes()
	if _, ok := attrs["epsilon"]; !ok {
		t.Error("Attributes should contain epsilon")
	}
}

func TestSimplifiedLayerNormalization_OutputShape(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gain, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	sln, err := NewSimplifiedLayerNormalization(engine, ops, gain, float32(1e-6))
	if err != nil {
		t.Fatal(err)
	}

	// Before Forward, OutputShape returns nil (no inputShape cached)
	if sln.OutputShape() != nil {
		t.Error("OutputShape before Forward should be nil")
	}

	input, _ := tensor.New[float32]([]int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	_, err = sln.Forward(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	shape := sln.OutputShape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 4 {
		t.Errorf("OutputShape = %v, want [2 4]", shape)
	}
}

func TestSimplifiedLayerNormalization_Forward_InvalidInputCount(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gain, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	sln, err := NewSimplifiedLayerNormalization(engine, ops, gain, float32(1e-6))
	if err != nil {
		t.Fatal(err)
	}

	_, err = sln.Forward(context.Background())
	if err == nil {
		t.Error("expected error for no inputs")
	}
}

func TestSimplifiedLayerNormalization_Backward_InvalidInputCount(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gain, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	sln, err := NewSimplifiedLayerNormalization(engine, ops, gain, float32(1e-6))
	if err != nil {
		t.Fatal(err)
	}

	grad, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 1, 1, 1})
	_, err = sln.Backward(context.Background(), 0, grad)
	if err == nil {
		t.Error("expected error for no inputs in Backward")
	}
}

func TestSimplifiedLayerNormalization_Backward_NoCacheError(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gain, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	sln, err := NewSimplifiedLayerNormalization(engine, ops, gain, float32(1e-6))
	if err != nil {
		t.Fatal(err)
	}

	// Call Backward without Forward (no cached tensors)
	grad, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 1, 1, 1})
	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	_, err = sln.Backward(context.Background(), 0, grad, input)
	if err == nil {
		t.Error("expected error for backward without forward")
	}
}

// --- SkipSimplifiedLayerNormalization accessor tests ---

func TestSkipSimplifiedLayerNormalization_OpType(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gain, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	skip, err := NewSkipSimplifiedLayerNormalization(engine, ops, gain, float32(1e-6))
	if err != nil {
		t.Fatal(err)
	}
	if skip.OpType() != "SkipSimplifiedLayerNormalization" {
		t.Errorf("OpType() = %q, want SkipSimplifiedLayerNormalization", skip.OpType())
	}
}

func TestSkipSimplifiedLayerNormalization_Attributes(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gain, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	skip, err := NewSkipSimplifiedLayerNormalization(engine, ops, gain, float32(1e-6))
	if err != nil {
		t.Fatal(err)
	}
	attrs := skip.Attributes()
	if _, ok := attrs["epsilon"]; !ok {
		t.Error("Attributes should contain epsilon")
	}
}

func TestSkipSimplifiedLayerNormalization_OutputShape(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gain, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	skip, err := NewSkipSimplifiedLayerNormalization(engine, ops, gain, float32(1e-6))
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.New[float32]([]int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	_, err = skip.Forward(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	shape := skip.OutputShape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 4 {
		t.Errorf("OutputShape = %v, want [2 4]", shape)
	}
}
