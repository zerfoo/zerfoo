package parity_test

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/registry"
	"github.com/zerfoo/ztensor/tensor"
)

// dirRegistry is a mock ModelRegistry that maps model IDs to local directories.
type dirRegistry struct {
	models map[string]*registry.ModelInfo
}

func (r *dirRegistry) Get(modelID string) (*registry.ModelInfo, bool) {
	info, ok := r.models[modelID]
	return info, ok
}

func (r *dirRegistry) Pull(_ context.Context, _ string) (*registry.ModelInfo, error) {
	return nil, nil
}

func (r *dirRegistry) List() []registry.ModelInfo { return nil }
func (r *dirRegistry) Delete(_ string) error      { return nil }

// loadZMFGraph loads a ZMF model and returns the computation graph.
// ZMF loading was removed; this always skips the test.
func loadZMFGraph(t *testing.T, _ string) *graph.Graph[float32] {
	t.Helper()
	t.Skip("ZMF loading is no longer supported")
	return nil
}

// forwardPassConfig holds parameters for a forward pass test.
type forwardPassConfig struct {
	Name         string
	SeqLen       int
	MinVocabSize int
}

// runForwardPassTest runs a single forward pass and validates the output.
func runForwardPassTest(t *testing.T, g *graph.Graph[float32], cfg forwardPassConfig) {
	t.Helper()

	inputData := make([]float32, cfg.SeqLen)
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}
	input, err := tensor.New[float32]([]int{1, cfg.SeqLen}, inputData)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Graph.Forward failed: %v", err)
	}
	if output == nil {
		t.Fatal("output tensor is nil")
	}

	outShape := output.Shape()
	t.Logf("%s output shape: %v", cfg.Name, outShape)

	if len(outShape) < 2 {
		t.Errorf("output rank %d < 2; want at least 2", len(outShape))
	}
	if len(outShape) == 3 {
		if outShape[0] != 1 {
			t.Errorf("output batch dim = %d, want 1", outShape[0])
		}
		if outShape[1] != cfg.SeqLen {
			t.Errorf("output seq dim = %d, want %d", outShape[1], cfg.SeqLen)
		}
		if outShape[2] < cfg.MinVocabSize {
			t.Errorf("output vocab dim = %d, want >= %d", outShape[2], cfg.MinVocabSize)
		}
	}

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

// runGreedyDecodeTest runs N greedy decode steps from an initial token sequence.
func runGreedyDecodeTest(t *testing.T, g *graph.Graph[float32], initTokens []float32, steps int) {
	t.Helper()

	tokens := append([]float32{}, initTokens...)

	for step := range steps {
		seqLen := len(tokens)
		input, err := tensor.New[float32]([]int{1, seqLen}, append([]float32{}, tokens...))
		if err != nil {
			t.Fatalf("step %d: tensor.New failed: %v", step, err)
		}

		output, err := g.Forward(context.Background(), input)
		if err != nil {
			t.Fatalf("step %d: Graph.Forward failed: %v", step, err)
		}
		if output == nil {
			t.Fatalf("step %d: output tensor is nil", step)
		}

		outShape := output.Shape()
		data := output.Data()

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

	expected := len(initTokens) + steps
	if len(tokens) != expected {
		t.Errorf("expected %d tokens after decode, got %d", expected, len(tokens))
	}
}

// modelParityConfig describes a complete parity test suite for a model family.
type modelParityConfig struct {
	// Name is a human-readable label (e.g. "Llama 3").
	Name string
	// ZMFEnvVar is the environment variable for the .zmf file path.
	ZMFEnvVar string
	// ModelDirEnvVar is the environment variable for the model directory.
	ModelDirEnvVar string
	// ModelID is the ID used with inference.Load.
	ModelID string
	// MinVocabSize is the minimum expected vocabulary dimension.
	MinVocabSize int
}

// runModelForwardPass runs the forward pass test for a model family.
func runModelForwardPass(t *testing.T, cfg modelParityConfig) {
	t.Helper()
	zmfPath := envOrSkip(t, cfg.ZMFEnvVar)
	g := loadZMFGraph(t, zmfPath)
	runForwardPassTest(t, g, forwardPassConfig{
		Name:         cfg.Name,
		SeqLen:       8,
		MinVocabSize: cfg.MinVocabSize,
	})
}

// runModelGreedyDecode runs the greedy decode test for a model family.
func runModelGreedyDecode(t *testing.T, cfg modelParityConfig) {
	t.Helper()
	zmfPath := envOrSkip(t, cfg.ZMFEnvVar)
	g := loadZMFGraph(t, zmfPath)
	runGreedyDecodeTest(t, g, []float32{1, 2, 3}, 5)
}

// runModelGeneration runs the generation test suite for a model family.
func runModelGeneration(t *testing.T, cfg modelParityConfig) {
	t.Helper()
	modelDir := modelDirOrSkip(t, cfg.ModelDirEnvVar, cfg.ZMFEnvVar)
	runGenerationTests(t, generationTestConfig{
		ModelID:  cfg.ModelID,
		ModelDir: modelDir,
	})
}

// envOrSkip returns the value of the named env var, or skips the test.
func envOrSkip(t *testing.T, key string) string {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		t.Skipf("%s not set; skipping", key)
	}
	return v
}

// modelDirOrSkip resolves a model directory from env vars, or skips the test.
func modelDirOrSkip(t *testing.T, dirEnvVar, zmfEnvVar string) string {
	t.Helper()
	if d := os.Getenv(dirEnvVar); d != "" {
		return d
	}
	zmfPath := os.Getenv(zmfEnvVar)
	if zmfPath == "" {
		t.Skipf("%s and %s not set; skipping", dirEnvVar, zmfEnvVar)
	}
	return filepath.Dir(zmfPath)
}

// generationTestConfig holds parameters for generation tests via inference API.
type generationTestConfig struct {
	ModelID  string
	ModelDir string
}

// runGenerationTests runs greedy, stream, and chat tests on an inference.Model.
func runGenerationTests(t *testing.T, cfg generationTestConfig) {
	t.Helper()

	reg := &dirRegistry{
		models: map[string]*registry.ModelInfo{
			cfg.ModelID: {ID: cfg.ModelID, Path: cfg.ModelDir},
		},
	}

	mdl, err := inference.Load(cfg.ModelID, inference.WithRegistry(reg))
	if err != nil {
		t.Fatalf("inference.Load failed: %v", err)
	}

	ctx := context.Background()

	t.Run("greedy_deterministic", func(t *testing.T) {
		prompt := "The capital of France is"
		result1, err := mdl.Generate(ctx, prompt,
			inference.WithTemperature(0),
			inference.WithMaxTokens(20),
		)
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}
		if result1 == "" {
			t.Fatal("greedy generation produced empty output")
		}
		t.Logf("greedy output: %q", result1)

		result2, err := mdl.Generate(ctx, prompt,
			inference.WithTemperature(0),
			inference.WithMaxTokens(20),
		)
		if err != nil {
			t.Fatalf("Generate (second) failed: %v", err)
		}
		if result1 != result2 {
			t.Errorf("greedy outputs differ:\n  run1: %q\n  run2: %q", result1, result2)
		}
	})

	t.Run("stream_parity", func(t *testing.T) {
		prompt := "Hello world"
		nonStream, err := mdl.Generate(ctx, prompt,
			inference.WithTemperature(0),
			inference.WithMaxTokens(10),
		)
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}

		var sb strings.Builder
		err = mdl.GenerateStream(ctx, prompt,
			generate.TokenStreamFunc(func(token string, done bool) error {
				if !done {
					sb.WriteString(token)
				}
				return nil
			}),
			inference.WithTemperature(0),
			inference.WithMaxTokens(10),
		)
		if err != nil {
			t.Fatalf("GenerateStream failed: %v", err)
		}

		streamed := sb.String()
		if nonStream != streamed {
			t.Errorf("stream/non-stream mismatch:\n  non-stream: %q\n  stream:     %q",
				nonStream, streamed)
		}
	})

	t.Run("chat", func(t *testing.T) {
		resp, err := mdl.Chat(ctx, []inference.Message{
			{Role: "user", Content: "Say hello in French"},
		}, inference.WithMaxTokens(20))
		if err != nil {
			t.Fatalf("Chat failed: %v", err)
		}
		if resp.Content == "" {
			t.Error("Chat produced empty content")
		}
		if resp.TokensUsed <= 0 {
			t.Error("TokensUsed should be positive")
		}
		t.Logf("chat response: %q (tokens: %d)", resp.Content, resp.TokensUsed)
	})
}
