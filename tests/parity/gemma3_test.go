package parity_test

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestGemma3ForwardPass loads a ZMF-converted Gemma 3 model and verifies
// that a single forward pass succeeds, produces a valid output shape, and
// contains no NaN or Inf values.
//
// The test is skipped when GEMMA3_ZMF_PATH is not set; it is intended for
// CI environments where the model file is present.
func TestGemma3ForwardPass(t *testing.T) {
	zmfPath := os.Getenv("GEMMA3_ZMF_PATH")
	if zmfPath == "" {
		t.Skip("GEMMA3_ZMF_PATH not set; skipping Gemma 3 forward pass test")
	}

	registry.RegisterAll()

	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	m, err := model.LoadModelFromZMF[float32](eng, ops, zmfPath)
	if err != nil {
		t.Fatalf("LoadModelFromZMF failed: %v", err)
	}
	if m.Graph == nil {
		t.Fatal("model graph is nil")
	}

	// Input: [1, 8] int64 token IDs cast to float32 for the graph.
	// Gemma 3 graphs expect float32 token IDs at the embedding lookup input.
	seqLen := 8
	inputData := make([]float32, seqLen)
	for i := range inputData {
		inputData[i] = float32(i + 1) // token IDs 1..8
	}
	input, err := tensor.New[float32]([]int{1, seqLen}, inputData)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}

	output, err := m.Graph.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Graph.Forward failed: %v", err)
	}
	if output == nil {
		t.Fatal("output tensor is nil")
	}

	outShape := output.Shape()
	t.Logf("Gemma 3 output shape: %v", outShape)

	// Output should be [1, seqLen, vocabSize] where vocabSize >= 256000.
	if len(outShape) < 2 {
		t.Errorf("output rank %d < 2; want at least 2", len(outShape))
	}
	if len(outShape) == 3 {
		if outShape[0] != 1 {
			t.Errorf("output batch dim = %d, want 1", outShape[0])
		}
		if outShape[1] != seqLen {
			t.Errorf("output seq dim = %d, want %d", outShape[1], seqLen)
		}
		if outShape[2] < 256000 {
			t.Errorf("output vocab dim = %d, want >= 256000", outShape[2])
		}
	}

	// Verify no NaN or Inf in the output.
	data := output.Data()
	for i, v := range data {
		f := float64(v)
		if math.IsNaN(f) {
			t.Errorf("output[%d] is NaN", i)
			break
		}
		if math.IsInf(f, 0) {
			t.Errorf("output[%d] is Inf", i)
			break
		}
	}
}

// TestGemma3GreedyDecode runs 5 greedy decode steps starting from a short
// prompt. Each step picks the argmax over the vocab dimension and appends
// the token to the sequence for the next step.
//
// Assertions: no error, no panic, 5 output tokens in [0, vocabSize).
func TestGemma3GreedyDecode(t *testing.T) {
	zmfPath := os.Getenv("GEMMA3_ZMF_PATH")
	if zmfPath == "" {
		t.Skip("GEMMA3_ZMF_PATH not set; skipping Gemma 3 greedy decode test")
	}

	registry.RegisterAll()

	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	m, err := model.LoadModelFromZMF[float32](eng, ops, zmfPath)
	if err != nil {
		t.Fatalf("LoadModelFromZMF failed: %v", err)
	}

	// Start with token IDs [1, 2, 3].
	tokens := []float32{1, 2, 3}
	const steps = 5

	for step := 0; step < steps; step++ {
		seqLen := len(tokens)
		input, err := tensor.New[float32]([]int{1, seqLen}, append([]float32{}, tokens...))
		if err != nil {
			t.Fatalf("step %d: tensor.New failed: %v", step, err)
		}

		output, err := m.Graph.Forward(context.Background(), input)
		if err != nil {
			t.Fatalf("step %d: Graph.Forward failed: %v", step, err)
		}
		if output == nil {
			t.Fatalf("step %d: output tensor is nil", step)
		}

		outShape := output.Shape()
		data := output.Data()

		// For a [1, seqLen, vocabSize] output, pick the last position's argmax.
		var vocabSize int
		var lastPosOffset int
		switch len(outShape) {
		case 3:
			vocabSize = outShape[2]
			lastPosOffset = (seqLen - 1) * vocabSize
		case 2:
			vocabSize = outShape[1]
			lastPosOffset = 0
		default:
			t.Fatalf("step %d: unexpected output rank %d", step, len(outShape))
		}

		if vocabSize == 0 {
			t.Fatalf("step %d: vocabSize is 0", step)
		}

		// Argmax over the last token position.
		bestIdx := 0
		bestVal := data[lastPosOffset]
		for j := 1; j < vocabSize; j++ {
			if data[lastPosOffset+j] > bestVal {
				bestVal = data[lastPosOffset+j]
				bestIdx = j
			}
		}

		if bestIdx < 0 || bestIdx >= vocabSize {
			t.Errorf("step %d: next token %d out of range [0, %d)", step, bestIdx, vocabSize)
		}
		t.Logf("step %d: next token = %d", step, bestIdx)
		tokens = append(tokens, float32(bestIdx))
	}

	if len(tokens) != 3+steps {
		t.Errorf("expected %d tokens after decode, got %d", 3+steps, len(tokens))
	}
}
