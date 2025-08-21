// package gemma_test provides an integration test for the Gemma model.
package gemma_test

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestGemmaIntegration runs an end-to-end test of loading and running the Gemma model.
func TestGemmaIntegration(t *testing.T) {
	// 1. Load the ZMF model
	zmfModel, err := model.LoadZMF("../../../gemma3/data/model.zmf")
	if err != nil {
		t.Fatalf("Failed to load ZMF model: %v", err)
	}

	// 2. Build the Zerfoo model from the ZMF graph
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	// Manually create the embedding layer
	embeddingWeightsProto, ok := zmfModel.Graph.Parameters["model.embed_tokens.weight"]
	if !ok {
		t.Fatalf("Embedding weights not found in ZMF model")
	}
	if embeddingWeightsProto == nil {
		t.Fatalf("Embedding weights proto is nil")
	}
	embeddingWeights, err := model.DecodeTensor[float32](embeddingWeightsProto)
	if err != nil {
		t.Fatalf("Failed to decode embedding weights: %v", err)
	}
	embeddingLayer, err := embeddings.NewTokenEmbedding[float32](engine, embeddingWeights.Shape()[0], embeddingWeights.Shape()[1])
	if err != nil {
		t.Fatalf("Failed to create token embedding layer: %v", err)
	}
	embeddingLayer.Parameters()[0].Value = embeddingWeights

	zerfooGraph, err := model.BuildFromZMF[float32](engine, &ops, zmfModel)
	if err != nil {
		t.Fatalf("Failed to build zerfoo model from ZMF: %v", err)
	}

	zerfooModel := model.NewModel(embeddingLayer, zerfooGraph)

	// 3. Create a dummy input tensor (tokenized prompt)
	inputTokens := []int{1, 2, 3, 4}
	inputTensor, err := tensor.New[int]([]int{1, 4}, inputTokens)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	// 4. Execute the model's forward pass
	_, err = zerfooModel.Forward(context.Background(), inputTensor)
	if err != nil {
		t.Fatalf("Model forward pass failed: %v", err)
	}

	// 5. De-tokenize and verify output (placeholder)
	t.Log("Gemma integration test passed (forward pass executed successfully).")
}