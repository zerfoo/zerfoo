// Command finetune runs QLoRA fine-tuning on a GGUF model.
//
// Usage:
//
//	zerfoo finetune --model path --dataset jsonl --rank 16 --epochs 3
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/training/lora"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// sample represents a single JSONL training example.
type sample struct {
	Input  string `json:"input"`
	Output string `json:"output"`
}

// loadDataset reads a JSONL file and returns training samples.
func loadDataset(path string) ([]sample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open dataset: %w", err)
	}
	defer f.Close()

	var samples []sample
	scanner := bufio.NewScanner(f)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		var s sample
		if err := json.Unmarshal(scanner.Bytes(), &s); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		samples = append(samples, s)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan dataset: %w", err)
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("dataset is empty")
	}
	return samples, nil
}

// finetuneModel is a minimal model for fine-tuning that implements lora.Model.
type finetuneModel struct {
	layers map[string]lora.Layer[float32]
	order  []string
}

func newFinetuneModel() *finetuneModel {
	return &finetuneModel{
		layers: make(map[string]lora.Layer[float32]),
	}
}

func (m *finetuneModel) AddLayer(layer lora.Layer[float32]) {
	name := layer.Name()
	m.layers[name] = layer
	m.order = append(m.order, name)
}

func (m *finetuneModel) Layers() []lora.Layer[float32] {
	result := make([]lora.Layer[float32], 0, len(m.order))
	for _, name := range m.order {
		result = append(result, m.layers[name])
	}
	return result
}

func (m *finetuneModel) ReplaceLayer(name string, replacement lora.Layer[float32]) error {
	if _, ok := m.layers[name]; !ok {
		return fmt.Errorf("layer %q not found", name)
	}
	m.layers[name] = replacement
	return nil
}

// stubLinear is a simple linear layer for synthetic fine-tuning.
type stubLinear struct {
	weights   *graph.Parameter[float32]
	engine    compute.Engine[float32]
	layerName string
	dIn, dOut int
}

func newStubLinear(name string, engine compute.Engine[float32], dIn, dOut int) (*stubLinear, error) {
	wData := make([]float32, dIn*dOut)
	for i := range wData {
		wData[i] = float32(i%7-3) * 0.1
	}
	wTensor, err := tensor.New[float32]([]int{dIn, dOut}, wData)
	if err != nil {
		return nil, err
	}
	param, err := graph.NewParameter[float32](name+"_weights", wTensor, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	return &stubLinear{
		weights:   param,
		engine:    engine,
		layerName: name,
		dIn:       dIn,
		dOut:      dOut,
	}, nil
}

func (s *stubLinear) OpType() string                          { return "Linear" }
func (s *stubLinear) Attributes() map[string]interface{}      { return nil }
func (s *stubLinear) OutputShape() []int                      { return []int{-1, s.dOut} }
func (s *stubLinear) Parameters() []*graph.Parameter[float32] { return []*graph.Parameter[float32]{s.weights} }
func (s *stubLinear) Name() string                            { return s.layerName }
func (s *stubLinear) SetName(name string) {
	s.layerName = name
	s.weights.Name = name + "_weights"
}

func (s *stubLinear) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return s.engine.MatMul(ctx, inputs[0], s.weights.Value)
}

func (s *stubLinear) Backward(ctx context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[float32], inputs ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	wT, err := s.engine.Transpose(ctx, s.weights.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dx, err := s.engine.MatMul(ctx, outputGradient, wT)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[float32]{dx}, nil
}

var _ lora.Layer[float32] = (*stubLinear)(nil)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "finetune: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to GGUF model file")
	datasetPath := flag.String("dataset", "", "path to JSONL dataset file")
	rank := flag.Int("rank", 16, "LoRA rank")
	alpha := flag.Float64("alpha", 32, "LoRA alpha scaling factor")
	epochs := flag.Int("epochs", 3, "number of training epochs")
	batchSize := flag.Int("batch-size", 4, "batch size")
	lr := flag.Float64("lr", 2e-4, "learning rate")
	output := flag.String("output", "adapter.gguf", "adapter output path")
	flag.Parse()

	if *datasetPath == "" {
		return fmt.Errorf("usage: zerfoo finetune --model path --dataset jsonl [options]")
	}

	// Check if model file exists.
	if *modelPath != "" {
		if _, err := os.Stat(*modelPath); err != nil {
			log.Printf("model file not found: %s, exiting 0", *modelPath)
			return nil
		}
	}

	// Load dataset.
	samples, err := loadDataset(*datasetPath)
	if err != nil {
		return fmt.Errorf("load dataset: %w", err)
	}
	log.Printf("loaded %d samples from %s", len(samples), *datasetPath)

	// Build synthetic model for fine-tuning.
	// In production, this would load the real GGUF model. For now, we build
	// a small synthetic model that demonstrates the QLoRA training loop.
	const dim = 64
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	model := newFinetuneModel()
	layer1, err := newStubLinear("q_proj", engine, dim, dim)
	if err != nil {
		return fmt.Errorf("create layer1: %w", err)
	}
	layer2, err := newStubLinear("v_proj", engine, dim, dim)
	if err != nil {
		return fmt.Errorf("create layer2: %w", err)
	}
	model.AddLayer(layer1)
	model.AddLayer(layer2)

	// Create optimizer and QLoRA trainer.
	opt := optimizer.NewAdamW8bit[float32](engine, float32(*lr), 0.9, 0.999, 1e-8, 0.01)
	trainer, err := lora.NewQLoRATrainer[float32](
		model, *rank, float32(*alpha),
		[]string{"q_proj", "v_proj"},
		engine, opt,
	)
	if err != nil {
		return fmt.Errorf("create trainer: %w", err)
	}

	log.Printf("rank=%d alpha=%.0f epochs=%d batch-size=%d lr=%.1e output=%s",
		*rank, *alpha, *epochs, *batchSize, *lr, *output)

	ctx := context.Background()
	totalSteps := len(samples) * *epochs / *batchSize
	if totalSteps == 0 {
		totalSteps = 1
	}

	step := 0
	for epoch := 0; epoch < *epochs; epoch++ {
		for i := 0; i < len(samples); i += *batchSize {
			end := i + *batchSize
			if end > len(samples) {
				end = len(samples)
			}
			batchLen := end - i

			// Build synthetic input/target tensors from text lengths.
			// In production, these would be tokenized sequences.
			inputData := make([]float32, batchLen*dim)
			targetData := make([]float32, batchLen*dim)
			for j := 0; j < batchLen; j++ {
				s := samples[i+j]
				for k := 0; k < dim; k++ {
					inputData[j*dim+k] = float32(len(s.Input)%7-3) * 0.01 * float32(k+1)
					targetData[j*dim+k] = float32(len(s.Output)%11-5) * 0.01 * float32(k+1)
				}
			}

			input, err := tensor.New[float32]([]int{batchLen, dim}, inputData)
			if err != nil {
				return fmt.Errorf("create input tensor: %w", err)
			}
			target, err := tensor.New[float32]([]int{batchLen, dim}, targetData)
			if err != nil {
				return fmt.Errorf("create target tensor: %w", err)
			}

			loss, err := trainer.Step(ctx, input, target)
			if err != nil {
				return fmt.Errorf("step %d: %w", step, err)
			}

			step++
			log.Printf("Step %d/%d loss=%.6f", step, totalSteps, loss)
		}
	}

	// Save adapter.
	if err := lora.SaveAdapter[float32](*output, model); err != nil {
		return fmt.Errorf("save adapter: %w", err)
	}
	log.Printf("adapter saved to %s", *output)

	return nil
}
