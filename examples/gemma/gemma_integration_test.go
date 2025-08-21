// package gemma_test provides an integration test for the Gemma model.
package gemma_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Simple tokenizer for demonstration purposes.
func Tokenize(text string) []int {
	// In a real scenario, this would use a proper sentencepiece tokenizer.
	// For this test, we'll just map characters to integers.
	tokens := []int{1} // Start with a BOS token
	for _, r := range text {
		tokens = append(tokens, int(r))
	}
	return tokens
}

// Simple de-tokenizer for demonstration purposes.
func Detokenize(tokens []int) string {
	var sb strings.Builder
	for _, t := range tokens {
		if t == 1 { // BOS
			continue
		}
		sb.WriteRune(rune(t))
	}
	return sb.String()
}

// TestGemmaIntegration runs an end-to-end test of loading and running the Gemma model.
func TestGemmaIntegration(t *testing.T) {
	registry.RegisterAll()
	// 1. Load the ZMF model
	zmfModel, err := model.LoadZMF("../../../gemma3/data/model.zmf")
	if err != nil {
		t.Fatalf("Failed to load ZMF model: %v", err)
	}

	// 2. Build the Zerfoo model from the ZMF graph
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	zerfooGraph, err := model.BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("Failed to build zerfoo model from ZMF: %v", err)
	}

	// 3. Tokenize a sample prompt
	prompt := "Hello, world!"
	inputTokens := Tokenize(prompt)

	// The input to the graph is a float32 tensor of token IDs.
	inputTensor, err := tensor.New[float32]([]int{1, len(inputTokens)}, nil)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	for i, token := range inputTokens {
		if err := inputTensor.Set(float32(token), 0, i); err != nil {
			t.Fatalf("Failed to set input tensor value: %v", err)
		}
	}

	// 4. Execute the model's forward pass
	// The Gemma model has multiple inputs (input_ids, attention_mask, position_ids, etc.)
	// For this integration test, we'll create a minimal set of inputs
	inputs := zerfooGraph.Inputs()
	t.Logf("Model has %d input nodes", len(inputs))
	
	// Create input tensors for all required inputs
	inputTensors := make([]*tensor.TensorNumeric[float32], len(inputs))
	for i, input := range inputs {
		// Create a tensor with the expected shape for each input
		shape := input.OutputShape()
		if len(shape) == 0 {
			shape = []int{1} // Default to scalar if no shape specified
		}
		inputTensors[i], err = tensor.New[float32](shape, nil)
		if err != nil {
			t.Fatalf("Failed to create input tensor %d: %v", i, err)
		}
		
		// Initialize with simple values (zeros for most, sequence for input_ids)
		if i == 0 { // Assume first input is input_ids
			inputTensors[i] = inputTensor // Use our tokenized input
		}
		// Other inputs can remain as zero-initialized tensors
	}

	outputTensor, err := zerfooGraph.Forward(context.Background(), inputTensors...)
	if err != nil {
		t.Fatalf("Model forward pass failed: %v", err)
	}

	// 5. De-tokenize and verify output (simple check)
	// The output of the model is logits. For generation, we'd need to sample from them.
	// For this test, we'll just check the shape of the output.
	// The Gemma model might have a different vocab size, so we check against the actual model's parameter shape.
	outputLayer := zerfooGraph.Output()
	finalLayerWithParams, ok := outputLayer.(interface{ Parameters() []*graph.Parameter[float32] })
	if !ok {
		t.Fatalf("Output layer does not have parameters, cannot determine vocab size.")
	}
	params := finalLayerWithParams.Parameters()
	if len(params) == 0 {
		t.Fatalf("Output layer has no parameters, cannot determine vocab size.")
	}
	vocabSize := params[0].Value.Shape()[0]

	expectedOutputShape := []int{1, len(inputTokens), vocabSize}
	outputShape := outputTensor.Shape()

	if len(outputShape) != 3 || outputShape[0] != expectedOutputShape[0] || outputShape[1] != expectedOutputShape[1] || outputShape[2] != expectedOutputShape[2] {
		t.Errorf("Expected output shape to be %v, but got %v", expectedOutputShape, outputShape)
	}

	fmt.Printf("Gemma integration test passed. Output shape: %v\n", outputShape)
	t.Log("Gemma integration test passed (forward pass executed successfully).")
}
