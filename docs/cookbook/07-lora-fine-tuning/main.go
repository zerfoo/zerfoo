// Recipe 07: Fine-Tuning with LoRA
//
// Inject LoRA (Low-Rank Adaptation) adapters into a model and train on custom
// data. Only the small LoRA matrices are updated -- the base model weights stay
// frozen, making fine-tuning fast and memory-efficient.
//
// This recipe demonstrates the LoRA + training API at the graph/layer level:
//   - Building a model with Linear layers
//   - Injecting LoRA adapters into target layers
//   - Running a forward pass through LoRA-wrapped layers
//   - Saving and loading the LoRA checkpoint
//
// Usage:
//
//	go run ./docs/cookbook/07-lora-fine-tuning/
package main

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/training/lora"
)

// simpleLinear is a minimal Linear layer that satisfies lora.Layer[T]
// (which requires graph.Node[T] + Named).
type simpleLinear struct {
	name    string
	weights *graph.Parameter[float32]
	engine  compute.Engine[float32]
	dIn     int
	dOut    int
}

func newSimpleLinear(name string, engine compute.Engine[float32], dIn, dOut int) (*simpleLinear, error) {
	data := make([]float32, dIn*dOut)
	for i := range data {
		data[i] = rand.Float32()*0.02 - 0.01
	}
	w, err := tensor.New[float32]([]int{dIn, dOut}, data)
	if err != nil {
		return nil, err
	}
	param, err := graph.NewParameter[float32](name+"_weights", w, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	return &simpleLinear{name: name, weights: param, engine: engine, dIn: dIn, dOut: dOut}, nil
}

func (l *simpleLinear) Name() string                            { return l.name }
func (l *simpleLinear) OpType() string                          { return "Linear" }
func (l *simpleLinear) Attributes() map[string]any              { return nil }
func (l *simpleLinear) OutputShape() []int                      { return []int{-1, l.dOut} }
func (l *simpleLinear) Parameters() []*graph.Parameter[float32] { return []*graph.Parameter[float32]{l.weights} }
func (l *simpleLinear) InputFeatures() int                      { return l.dIn }
func (l *simpleLinear) OutputFeatures() int                     { return l.dOut }

func (l *simpleLinear) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return l.engine.MatMul(ctx, inputs[0], l.weights.Value)
}

func (l *simpleLinear) Backward(_ context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return []*tensor.TensorNumeric[float32]{outputGradient}, nil
}

// simpleModel wraps a set of Linear layers for LoRA injection.
type simpleModel struct {
	layers map[string]lora.Layer[float32]
	order  []string
}

func (m *simpleModel) Layers() []lora.Layer[float32] {
	var out []lora.Layer[float32]
	for _, name := range m.order {
		out = append(out, m.layers[name])
	}
	return out
}

func (m *simpleModel) ReplaceLayer(name string, replacement lora.Layer[float32]) error {
	if _, ok := m.layers[name]; !ok {
		return fmt.Errorf("layer %q not found", name)
	}
	m.layers[name] = replacement
	return nil
}

func main() {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Build a small model with attention-like projections.
	qProj, _ := newSimpleLinear("q_proj", engine, 64, 64)
	kProj, _ := newSimpleLinear("k_proj", engine, 64, 64)
	vProj, _ := newSimpleLinear("v_proj", engine, 64, 64)
	oProj, _ := newSimpleLinear("o_proj", engine, 64, 64)

	model := &simpleModel{
		layers: map[string]lora.Layer[float32]{
			"q_proj": qProj, "k_proj": kProj, "v_proj": vProj, "o_proj": oProj,
		},
		order: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
	}

	// Inject LoRA adapters into Q and V projections (rank=8, alpha=16).
	// Only the LoRA A and B matrices are trainable; base weights are frozen.
	err := lora.InjectLoRA[float32](
		model,
		8,     // rank
		16.0,  // alpha
		[]string{"q_proj", "v_proj"},
		engine,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "inject lora: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Injected LoRA into q_proj and v_proj (rank=8, alpha=16)")

	// Count total parameters across all layers.
	var totalParams int
	for _, layer := range model.Layers() {
		for _, p := range layer.Parameters() {
			n := 1
			for _, d := range p.Value.Shape() {
				n *= d
			}
			totalParams += n
		}
	}
	fmt.Printf("Total parameters: %d\n", totalParams)

	// Forward pass with synthetic data through LoRA-wrapped layers.
	ctx := context.Background()
	inputData := make([]float32, 4*64)
	for i := range inputData {
		inputData[i] = rand.Float32()
	}
	input, _ := tensor.New[float32]([]int{4, 64}, inputData)

	out := input
	for _, name := range model.order {
		layer := model.layers[name]
		out, err = layer.Forward(ctx, out)
		if err != nil {
			fmt.Fprintf(os.Stderr, "forward %s: %v\n", name, err)
			os.Exit(1)
		}
	}
	fmt.Printf("Output shape: %v\n", out.Shape())

	// Save the LoRA adapter checkpoint.
	checkpointPath := "lora-adapter.bin"
	if err := lora.SaveAdapter[float32](checkpointPath, model); err != nil {
		fmt.Fprintf(os.Stderr, "save: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Saved LoRA adapter to %s\n", checkpointPath)

	// Clean up the checkpoint file created by this demo.
	os.Remove(checkpointPath)
	fmt.Println("Done.")
}
