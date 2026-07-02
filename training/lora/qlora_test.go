package lora

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// nf4StubLinear is a float32-specific Linear layer whose weight tensor uses
// NF4Storage under the hood. It simulates a quantized base layer that
// dequantizes on the fly during forward.
type nf4StubLinear struct {
	weights    *graph.Parameter[float32]
	engine     compute.Engine[float32]
	layerName  string
	dIn, dOut  int
	nf4Storage *tensor.NF4Storage
}

func newNF4StubLinear(name string, engine compute.Engine[float32], dIn, dOut int, data []float32) (*nf4StubLinear, error) {
	// Quantize data to NF4.
	nf4 := tensor.NewNF4Storage(data, []int{dIn, dOut})

	// Create weight tensor backed by NF4Storage (NF4Storage implements Storage[float32]).
	wTensor, err := tensor.NewWithStorage[float32]([]int{dIn, dOut}, nf4)
	if err != nil {
		return nil, err
	}
	param, err := graph.NewParameter[float32](name+"_weights", wTensor, tensor.New[float32])
	if err != nil {
		return nil, err
	}

	return &nf4StubLinear{
		weights:    param,
		engine:     engine,
		layerName:  name,
		dIn:        dIn,
		dOut:       dOut,
		nf4Storage: nf4,
	}, nil
}

func (n *nf4StubLinear) OpType() string                     { return "Linear" }
func (n *nf4StubLinear) Attributes() map[string]interface{} { return nil }
func (n *nf4StubLinear) OutputShape() []int                 { return []int{-1, n.dOut} }
func (n *nf4StubLinear) Parameters() []*graph.Parameter[float32] {
	return []*graph.Parameter[float32]{n.weights}
}
func (n *nf4StubLinear) Name() string { return n.layerName }
func (n *nf4StubLinear) SetName(name string) {
	n.layerName = name
	n.weights.Name = name + "_weights"
}

func (n *nf4StubLinear) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	// NF4Storage.Slice() dequantizes on the fly, and the engine uses Data() which calls Slice().
	return n.engine.MatMul(ctx, inputs[0], n.weights.Value)
}

func (n *nf4StubLinear) Backward(ctx context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[float32], inputs ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	// Base weights are frozen in QLoRA -- no gradient accumulation for base weights.
	// Just propagate gradient to input: dx = grad @ W^T
	wT, err := n.engine.Transpose(ctx, n.weights.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dx, err := n.engine.MatMul(ctx, outputGradient, wT)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[float32]{dx}, nil
}

// Verify nf4StubLinear implements Layer[float32].
var _ Layer[float32] = (*nf4StubLinear)(nil)

func TestQLORATrainer(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	// Tiny synthetic model: 2 Linear layers.
	// Simulates a Gemma 3 1B-style model at small scale.
	d := 32
	vocab := 64
	rank := 4
	alpha := float32(8.0)

	// Create base weight data (deterministic).
	w1Data := make([]float32, d*d)
	for i := range w1Data {
		w1Data[i] = float32(i%7-3) * 0.1
	}
	w2Data := make([]float32, d*vocab)
	for i := range w2Data {
		w2Data[i] = float32(i%11-5) * 0.05
	}

	// Save original NF4-quantized weight data for later comparison.
	nf4W1 := tensor.NewNF4Storage(w1Data, []int{d, d})
	nf4W2 := tensor.NewNF4Storage(w2Data, []int{d, vocab})
	origW1 := make([]float32, len(nf4W1.Dequantize()))
	copy(origW1, nf4W1.Dequantize())
	origW2 := make([]float32, len(nf4W2.Dequantize()))
	copy(origW2, nf4W2.Dequantize())

	// Build model with NF4-quantized base weights.
	m := newTestModel[float32]()

	layer1, err := newNF4StubLinear("layer1", engine, d, d, w1Data)
	if err != nil {
		t.Fatalf("failed to create layer1: %v", err)
	}
	layer2, err := newNF4StubLinear("layer2", engine, d, vocab, w2Data)
	if err != nil {
		t.Fatalf("failed to create layer2: %v", err)
	}
	m.AddLayer(layer1)
	m.AddLayer(layer2)

	// Create optimizer.
	opt := optimizer.NewAdamW[float32](engine, 0.001, 0.9, 0.999, 1e-8, 0.0)

	// Create QLoRA trainer -- injects LoRA adapters into both layers.
	trainer, err := NewQLoRATrainer[float32](
		m, rank, alpha,
		[]string{"layer1", "layer2"},
		engine, opt,
	)
	if err != nil {
		t.Fatalf("NewQLoRATrainer failed: %v", err)
	}

	// Verify LoRA was injected.
	for _, layer := range m.Layers() {
		if _, ok := layer.(*LoraLinear[float32]); !ok {
			t.Fatalf("expected layer %q to be LoraLinear, got %T", layer.Name(), layer)
		}
	}

	// Create synthetic training data.
	batchSize := 2
	inputData := make([]float32, batchSize*d)
	for i := range inputData {
		inputData[i] = float32(i%13-6) * 0.1
	}
	input, err := tensor.New[float32]([]int{batchSize, d}, inputData)
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	targetData := make([]float32, batchSize*vocab)
	for i := range targetData {
		targetData[i] = float32(i%9-4) * 0.01
	}
	target, err := tensor.New[float32]([]int{batchSize, vocab}, targetData)
	if err != nil {
		t.Fatalf("failed to create target: %v", err)
	}

	// Run 20 training steps and verify loss decreases.
	var losses []float32
	for step := 0; step < 20; step++ {
		loss, err := trainer.Step(ctx, input, target)
		if err != nil {
			t.Fatalf("Step %d failed: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("Step %d: loss is NaN or Inf: %v", step, loss)
		}
		losses = append(losses, loss)
	}

	// Verify loss decreased: last loss should be less than first loss.
	if losses[len(losses)-1] >= losses[0] {
		t.Errorf("loss did not decrease: first=%f last=%f", losses[0], losses[len(losses)-1])
	}
	t.Logf("loss progression: first=%f last=%f (%.1f%% reduction)",
		losses[0], losses[len(losses)-1],
		(1-float64(losses[len(losses)-1])/float64(losses[0]))*100)

	// Verify only A and B gradients are non-nil after training.
	params := trainer.TrainableParams()
	if len(params) == 0 {
		t.Fatal("no trainable parameters found")
	}
	for _, p := range params {
		if p.Gradient == nil {
			t.Errorf("LoRA parameter %q has nil gradient", p.Name)
		}
	}

	// Verify base weights are unchanged.
	for _, layer := range m.Layers() {
		ll, ok := layer.(*LoraLinear[float32])
		if !ok {
			continue
		}
		baseParams := ll.base.Parameters()
		if len(baseParams) == 0 {
			continue
		}
		baseWeight := baseParams[0].Value
		baseData := baseWeight.Data()

		var origData []float32
		switch ll.Name() {
		case "layer1":
			origData = origW1
		case "layer2":
			origData = origW2
		default:
			continue
		}

		for i, v := range baseData {
			if i < len(origData) && v != origData[i] {
				t.Errorf("base weight %q changed at index %d: was %f, now %f",
					ll.Name(), i, origData[i], v)
				break
			}
		}
	}
}

func TestQLoRATrainer_InvalidArgs(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	m := newTestModel[float32]()
	opt := optimizer.NewAdamW[float32](engine, 0.001, 0.9, 0.999, 1e-8, 0.0)

	// Zero rank.
	_, err := NewQLoRATrainer[float32](m, 0, 1.0, []string{"q_proj"}, engine, opt)
	if err == nil {
		t.Error("expected error for zero rank, got nil")
	}

	// Empty target modules.
	_, err = NewQLoRATrainer[float32](m, 4, 1.0, []string{}, engine, opt)
	if err == nil {
		t.Error("expected error for empty targetModules, got nil")
	}
}
