package lora

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// testModel is a synthetic model with named Linear layers for testing injection.
type testModel[T tensor.Numeric] struct {
	layers map[string]Layer[T]
	order  []string // preserves insertion order
}

func newTestModel[T tensor.Numeric]() *testModel[T] {
	return &testModel[T]{
		layers: make(map[string]Layer[T]),
	}
}

func (m *testModel[T]) AddLayer(layer Layer[T]) {
	name := layer.Name()
	m.layers[name] = layer
	m.order = append(m.order, name)
}

func (m *testModel[T]) Layers() []Layer[T] {
	result := make([]Layer[T], 0, len(m.order))
	for _, name := range m.order {
		result = append(result, m.layers[name])
	}
	return result
}

func (m *testModel[T]) ReplaceLayer(name string, replacement Layer[T]) error {
	if _, ok := m.layers[name]; !ok {
		return nil
	}
	m.layers[name] = replacement
	return nil
}

// namedStubLinear wraps stubLinear with a name to implement Layer[T].
type namedStubLinear[T tensor.Numeric] struct {
	*stubLinear[T]
	layerName      string
	inputFeatures  int
	outputFeatures int
}

func newNamedStubLinear[T tensor.Numeric](name string, engine compute.Engine[T], dIn, dOut int) (*namedStubLinear[T], error) {
	wData := make([]T, dIn*dOut)
	for i := range wData {
		wData[i] = T(i + 1)
	}
	wTensor, err := tensor.New[T]([]int{dIn, dOut}, wData)
	if err != nil {
		return nil, err
	}
	base, err := newStubLinear[T](engine, wTensor)
	if err != nil {
		return nil, err
	}
	base.weights.Name = name + "_weights"
	return &namedStubLinear[T]{
		stubLinear:     base,
		layerName:      name,
		inputFeatures:  dIn,
		outputFeatures: dOut,
	}, nil
}

func (n *namedStubLinear[T]) Name() string   { return n.layerName }
func (n *namedStubLinear[T]) OpType() string { return "Linear" }
func (n *namedStubLinear[T]) SetName(name string) {
	n.layerName = name
	n.weights.Name = name + "_weights"
}
func (n *namedStubLinear[T]) InputFeatures() int  { return n.inputFeatures }
func (n *namedStubLinear[T]) OutputFeatures() int { return n.outputFeatures }

func TestLoraInject(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dIn, dOut := 512, 512
	rank := 2
	alpha := float32(4.0)

	// Build a synthetic model with 3 Linear layers.
	m := newTestModel[float32]()

	qProj, err := newNamedStubLinear[float32]("q_proj", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create q_proj: %v", err)
	}
	vProj, err := newNamedStubLinear[float32]("v_proj", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create v_proj: %v", err)
	}
	mlp, err := newNamedStubLinear[float32]("mlp", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create mlp: %v", err)
	}

	m.AddLayer(qProj)
	m.AddLayer(vProj)
	m.AddLayer(mlp)

	// Inject LoRA into q_proj and v_proj only.
	err = InjectLoRA[float32](m, rank, alpha, []string{"q_proj", "v_proj"}, engine)
	if err != nil {
		t.Fatalf("InjectLoRA failed: %v", err)
	}

	// Assert: q_proj and v_proj are now LoraLinear.
	layers := m.Layers()
	for _, layer := range layers {
		name := layer.Name()
		switch name {
		case "q_proj", "v_proj":
			if _, ok := layer.(*LoraLinear[float32]); !ok {
				t.Errorf("expected %q to be LoraLinear, got %T", name, layer)
			}
		case "mlp":
			if _, ok := layer.(*LoraLinear[float32]); ok {
				t.Errorf("expected %q to remain unchanged, but it was replaced with LoraLinear", name)
			}
		}
	}

	// Assert: trainable params / total params <= 1%.
	trainable := TrainableParamCount[float32](m)
	total := TotalParamCount[float32](m)
	if total == 0 {
		t.Fatal("total param count is 0")
	}
	ratio := float64(trainable) / float64(total)
	t.Logf("trainable=%d total=%d ratio=%.4f%%", trainable, total, ratio*100)
	if ratio > 0.01 {
		t.Errorf("trainable/total ratio = %.4f, want <= 0.01 (1%%)", ratio)
	}

	// Assert: forward pass still produces output (no crashes).
	ctx := context.Background()
	x, err := tensor.New[float32]([]int{1, dIn}, make([]float32, dIn))
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}
	for _, layer := range m.Layers() {
		out, err := layer.Forward(ctx, x)
		if err != nil {
			t.Fatalf("forward pass for %q failed: %v", layer.(Named).Name(), err)
		}
		if out == nil {
			t.Fatalf("forward pass for %q returned nil", layer.(Named).Name())
		}
	}
}

func TestLoraInject_SuffixMatch(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	m := newTestModel[float32]()
	layer, err := newNamedStubLinear[float32]("layers.0.self_attn.q_proj", engine, 16, 8)
	if err != nil {
		t.Fatalf("failed to create layer: %v", err)
	}
	m.AddLayer(layer)

	err = InjectLoRA[float32](m, 2, 4.0, []string{"q_proj"}, engine)
	if err != nil {
		t.Fatalf("InjectLoRA failed: %v", err)
	}

	replaced := m.Layers()[0]
	if _, ok := replaced.(*LoraLinear[float32]); !ok {
		t.Errorf("expected suffix-matched layer to be LoraLinear, got %T", replaced)
	}
}

func TestLoraInject_NoMatch(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	m := newTestModel[float32]()
	layer, err := newNamedStubLinear[float32]("mlp", engine, 16, 8)
	if err != nil {
		t.Fatalf("failed to create layer: %v", err)
	}
	m.AddLayer(layer)

	err = InjectLoRA[float32](m, 2, 4.0, []string{"q_proj"}, engine)
	if err == nil {
		t.Error("expected error when no layers match, got nil")
	}
}

func TestLoraInject_EmptyTargets(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	m := newTestModel[float32]()

	err := InjectLoRA[float32](m, 2, 4.0, []string{}, engine)
	if err == nil {
		t.Error("expected error for empty targetModules, got nil")
	}
}

func TestTrainableParamCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	m := newTestModel[float32]()
	dIn, dOut, rank := 32, 16, 2

	// Add a regular linear layer (not LoRA).
	regular, err := newNamedStubLinear[float32]("regular", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create layer: %v", err)
	}
	m.AddLayer(regular)

	// Before injection: 0 trainable (no LoRA layers).
	if count := TrainableParamCount[float32](m); count != 0 {
		t.Errorf("trainable param count before injection = %d, want 0", count)
	}

	// Create a LoRA-wrapped layer manually.
	base, err := newNamedStubLinear[float32]("lora_layer", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create base: %v", err)
	}
	loraLayer, err := NewLoraLinear[float32]("lora_layer", base, rank, 4.0, engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create LoraLinear: %v", err)
	}
	m.AddLayer(loraLayer)

	trainable := TrainableParamCount[float32](m)
	// A: rank * dIn = 2*32 = 64, B: dOut * rank = 16*2 = 32, total trainable = 96
	expectedTrainable := rank*dIn + dOut*rank
	if trainable != expectedTrainable {
		t.Errorf("trainable param count = %d, want %d", trainable, expectedTrainable)
	}
}

func TestTotalParamCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	m := newTestModel[float32]()
	dIn, dOut, rank := 32, 16, 2

	regular, err := newNamedStubLinear[float32]("regular", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create layer: %v", err)
	}
	m.AddLayer(regular)

	base, err := newNamedStubLinear[float32]("lora_layer", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create base: %v", err)
	}
	loraLayer, err := NewLoraLinear[float32]("lora_layer", base, rank, 4.0, engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create LoraLinear: %v", err)
	}
	m.AddLayer(loraLayer)

	total := TotalParamCount[float32](m)
	// regular: dIn*dOut = 512
	// lora_layer: A(rank*dIn=64) + B(dOut*rank=32) + base(dIn*dOut=512) = 608
	// Total = 512 + 608 = 1120
	expected := dIn*dOut + (rank*dIn + dOut*rank + dIn*dOut)
	if total != expected {
		t.Errorf("total param count = %d, want %d", total, expected)
	}
}

// Verify LoraLinear satisfies Layer interface.
var _ Layer[float32] = (*LoraLinear[float32])(nil)
var _ graph.Node[float32] = (*LoraLinear[float32])(nil)
