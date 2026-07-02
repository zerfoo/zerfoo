package inference

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestTensorLookup_Lookup(t *testing.T) {
	dummy, err := tensor.New[float32]([]int{2, 3}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tensors := map[string]*tensor.TensorNumeric[float32]{
		"model.embed_tokens.weight": dummy,
	}
	tl := newTensorLookup(tensors)

	got, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != dummy {
		t.Fatal("Lookup returned wrong tensor")
	}
}

func TestTensorLookup_LookupMissing(t *testing.T) {
	tensors := map[string]*tensor.TensorNumeric[float32]{}
	tl := newTensorLookup(tensors)

	_, err := tl.Lookup("nonexistent")
	if err == nil {
		t.Fatal("expected error for missing tensor")
	}
	want := `missing tensor "nonexistent"`
	if err.Error() != want {
		t.Fatalf("error = %q, want %q", err.Error(), want)
	}
}

func TestTensorLookup_Optional(t *testing.T) {
	dummy, err := tensor.New[float32]([]int{4}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tensors := map[string]*tensor.TensorNumeric[float32]{
		"lm_head.weight": dummy,
	}
	tl := newTensorLookup(tensors)

	got, ok := tl.Optional("lm_head.weight")
	if !ok {
		t.Fatal("expected ok=true for existing tensor")
	}
	if got != dummy {
		t.Fatal("Optional returned wrong tensor")
	}

	got, ok = tl.Optional("missing")
	if ok {
		t.Fatal("expected ok=false for missing tensor")
	}
	if got != nil {
		t.Fatal("expected nil tensor for missing key")
	}
}

func TestTensorLookup_Has(t *testing.T) {
	dummy, err := tensor.New[float32]([]int{1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tensors := map[string]*tensor.TensorNumeric[float32]{
		"weight": dummy,
	}
	tl := newTensorLookup(tensors)

	if !tl.Has("weight") {
		t.Fatal("expected Has to return true for existing key")
	}
	if tl.Has("missing") {
		t.Fatal("expected Has to return false for missing key")
	}
}

func TestParamWrapper_Wrap(t *testing.T) {
	dummy, err := tensor.New[float32]([]int{3, 4}, nil)
	if err != nil {
		t.Fatal(err)
	}
	pw := newParamWrapper[float32]()

	p := pw.Wrap("layer.0.weight", dummy)
	if p.Name != "layer.0.weight" {
		t.Fatalf("Name = %q, want %q", p.Name, "layer.0.weight")
	}
	if p.Value != dummy {
		t.Fatal("Value does not match input tensor")
	}
	if p.Gradient != nil {
		t.Fatal("expected nil Gradient for newly wrapped parameter")
	}
}

func TestNewEmbeddingNode(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	vocabSize, hiddenDim := 8, 4
	weightData := make([]float32, vocabSize*hiddenDim)
	for i := range weightData {
		weightData[i] = float32(i) * 0.1
	}
	weight, err := tensor.New([]int{vocabSize, hiddenDim}, weightData)
	if err != nil {
		t.Fatal(err)
	}

	node := newEmbeddingNode[float32](engine, weight, 0)
	if node.engine != engine {
		t.Fatal("engine not set")
	}
	if node.weight != weight {
		t.Fatal("weight not set")
	}
	if node.scale != 0 {
		t.Fatalf("scale = %v, want 0", node.scale)
	}
	if node.OpType() != "EmbeddingLookup" {
		t.Fatalf("OpType = %q, want %q", node.OpType(), "EmbeddingLookup")
	}

	// Forward: look up token IDs 2 and 5.
	ids, err := tensor.New([]int{1, 2}, []float32{2, 5})
	if err != nil {
		t.Fatal(err)
	}
	out, err := node.Forward(context.Background(), ids)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if s := out.Shape(); len(s) != 3 || s[0] != 1 || s[1] != 2 || s[2] != hiddenDim {
		t.Fatalf("shape = %v, want [1, 2, %d]", s, hiddenDim)
	}
	// Verify token 2 row: indices 8..11 -> 0.8, 0.9, 1.0, 1.1
	data := out.Data()
	for j := 0; j < hiddenDim; j++ {
		want := float32(2*hiddenDim+j) * 0.1
		if got := data[j]; got != want {
			t.Fatalf("data[%d] = %v, want %v", j, got, want)
		}
	}
}

func TestNewEmbeddingNode_WithScale(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	vocabSize, hiddenDim := 4, 2
	weightData := make([]float32, vocabSize*hiddenDim)
	for i := range weightData {
		weightData[i] = 1.0
	}
	weight, err := tensor.New([]int{vocabSize, hiddenDim}, weightData)
	if err != nil {
		t.Fatal(err)
	}

	scale := float32(2.5)
	node := newEmbeddingNode[float32](engine, weight, scale)
	if node.scale != scale {
		t.Fatalf("scale = %v, want %v", node.scale, scale)
	}

	ids, err := tensor.New([]int{1, 1}, []float32{0})
	if err != nil {
		t.Fatal(err)
	}
	out, err := node.Forward(context.Background(), ids)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	// All weight values are 1.0, scaled by 2.5.
	for i, v := range out.Data() {
		if v != scale {
			t.Fatalf("data[%d] = %v, want %v", i, v, scale)
		}
	}
}

func TestNewLMHeadNode(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	vocabSize, hiddenDim := 4, 3
	weightData := make([]float32, vocabSize*hiddenDim)
	for i := range weightData {
		weightData[i] = float32(i) * 0.01
	}
	weight, err := tensor.New([]int{vocabSize, hiddenDim}, weightData)
	if err != nil {
		t.Fatal(err)
	}

	node := newLMHeadNode[float32](engine, weight, 0)
	if node.engine != engine {
		t.Fatal("engine not set")
	}
	if node.weight != weight {
		t.Fatal("weight not set")
	}
	if node.softcapVal != 0 {
		t.Fatalf("softcapVal = %v, want 0", node.softcapVal)
	}
	if node.OpType() != "LMHead" {
		t.Fatalf("OpType = %q, want %q", node.OpType(), "LMHead")
	}

	// Forward: [1, 1, hiddenDim] -> [1, 1, vocabSize].
	inputData := make([]float32, hiddenDim)
	for i := range inputData {
		inputData[i] = 1.0
	}
	input, err := tensor.New([]int{1, 1, hiddenDim}, inputData)
	if err != nil {
		t.Fatal(err)
	}
	out, err := node.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if s := out.Shape(); len(s) != 3 || s[0] != 1 || s[1] != 1 || s[2] != vocabSize {
		t.Fatalf("shape = %v, want [1, 1, %d]", s, vocabSize)
	}
}

func TestNewLMHeadNode_WithSoftcap(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	vocabSize, hiddenDim := 2, 2
	weight, err := tensor.New([]int{vocabSize, hiddenDim}, []float32{1, 0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}

	softcap := float32(30.0)
	node := newLMHeadNode[float32](engine, weight, softcap)
	if node.softcapVal != softcap {
		t.Fatalf("softcapVal = %v, want %v", node.softcapVal, softcap)
	}

	// Identity-like weight, input [100, 0] should produce logits [100, 0]
	// after softcap: 30 * tanh(100/30) ≈ 30 * ~1.0 ≈ 30.
	input, err := tensor.New([]int{1, 1, hiddenDim}, []float32{100, 0})
	if err != nil {
		t.Fatal(err)
	}
	out, err := node.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	data := out.Data()
	// First logit should be capped near 30.
	if data[0] < 29.0 || data[0] > 30.1 {
		t.Fatalf("softcapped logit[0] = %v, expected near 30", data[0])
	}
	// Second logit is 0, tanh(0)=0, so should be 0.
	if data[1] != 0 {
		t.Fatalf("softcapped logit[1] = %v, want 0", data[1])
	}
}
