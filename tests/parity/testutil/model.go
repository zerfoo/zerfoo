package testutil

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/model/registry"
	"github.com/zerfoo/zerfoo/tests/parity/modelset"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// DirRegistry is a mock ModelRegistry that maps model IDs to local directories.
type DirRegistry struct {
	Models map[string]*registry.ModelInfo
}

// Get returns the model info for the given ID, if any.
func (r *DirRegistry) Get(modelID string) (*registry.ModelInfo, bool) {
	info, ok := r.Models[modelID]
	return info, ok
}

// Pull is a no-op for the directory-backed test registry.
func (r *DirRegistry) Pull(_ context.Context, _ string) (*registry.ModelInfo, error) {
	return nil, nil
}

// List returns nil; tests don't enumerate the registry.
func (r *DirRegistry) List() []registry.ModelInfo { return nil }

// Delete is a no-op for the directory-backed test registry.
func (r *DirRegistry) Delete(_ string) error { return nil }

// LoadZMFGraph loads a ZMF model and returns the computation graph.
// ZMF loading was removed; this always skips the test.
func LoadZMFGraph(t *testing.T, _ string) *graph.Graph[float32] {
	t.Helper()
	t.Skip("ZMF loading is no longer supported")
	return nil
}

// ForwardPassConfig holds parameters for a forward pass test.
type ForwardPassConfig struct {
	Name         string
	SeqLen       int
	MinVocabSize int
}

// RunForwardPassTest runs a single forward pass and validates the output.
func RunForwardPassTest(t *testing.T, g *graph.Graph[float32], cfg ForwardPassConfig) {
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

// RunGreedyDecodeTest runs N greedy decode steps from an initial token sequence.
func RunGreedyDecodeTest(t *testing.T, g *graph.Graph[float32], initTokens []float32, steps int) {
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

// ModelParityConfig describes a complete parity test suite for a model family.
type ModelParityConfig struct {
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
	// MatrixRow is the verified-model matrix row key
	// (tests/parity/modelset/model-matrix.json) that pins the exact GGUF
	// filename for this suite. It is REQUIRED: without it the suite would
	// have to scan the model directory, and the models are staged flat, so
	// every suite would silently load the same file. See docs/lore.md L-0018.
	MatrixRow string
}

// RunModelForwardPass runs the forward pass test for a model family.
func RunModelForwardPass(t *testing.T, cfg ModelParityConfig) {
	t.Helper()
	zmfPath := EnvOrSkip(t, cfg.ZMFEnvVar)
	g := LoadZMFGraph(t, zmfPath)
	RunForwardPassTest(t, g, ForwardPassConfig{
		Name:         cfg.Name,
		SeqLen:       8,
		MinVocabSize: cfg.MinVocabSize,
	})
}

// RunModelGreedyDecode runs the greedy decode test for a model family.
func RunModelGreedyDecode(t *testing.T, cfg ModelParityConfig) {
	t.Helper()
	zmfPath := EnvOrSkip(t, cfg.ZMFEnvVar)
	g := LoadZMFGraph(t, zmfPath)
	RunGreedyDecodeTest(t, g, []float32{1, 2, 3}, 5)
}

// RunModelGeneration runs the generation test suite for a model family against
// the exact GGUF file its matrix row pins, after asserting the resolved file's
// identity. It never scans a directory for "some .gguf".
func RunModelGeneration(t *testing.T, cfg ModelParityConfig) {
	t.Helper()
	modelPath := ResolveMatrixModelOrSkip(t, cfg)
	RunGenerationTests(t, GenerationTestConfig{
		ModelID:   cfg.ModelID,
		ModelPath: modelPath,
	})
}

// ResolveMatrixModelOrSkip turns a parity config into the absolute path of the
// one GGUF its matrix row declares, verifying the file's self-reported identity
// and refusing to let two rows claim the same file.
//
// It fails the test (rather than skipping) for anything that would produce a
// dishonest green: a config without a matrix row, an unknown row, an identity
// mismatch, or a duplicate resolution. It skips only when the model is genuinely
// not available on this host.
func ResolveMatrixModelOrSkip(t *testing.T, cfg ModelParityConfig) string {
	t.Helper()

	if cfg.MatrixRow == "" {
		t.Fatalf("%s: ModelParityConfig.MatrixRow is empty; every parity suite must pin "+
			"a matrix row in tests/parity/modelset/model-matrix.json", cfg.Name)
	}

	matrix, err := modelset.Default()
	if err != nil {
		t.Fatalf("load verified-model matrix: %v", err)
	}

	row, err := matrix.Row(cfg.MatrixRow)
	if err != nil {
		t.Fatalf("%s: %v", cfg.Name, err)
	}
	if !row.Staged() {
		t.Skipf("%s: matrix row %q pins no GGUF file (model not staged); "+
			"skipping instead of scanning a directory", cfg.Name, row.Key)
	}

	baseDir, err := matrix.BaseDir(row.Key)
	if err != nil {
		t.Skipf("%s: %v", cfg.Name, err)
	}

	path, err := matrix.Resolve(row.Key, baseDir)
	if err != nil {
		if errors.Is(err, modelset.ErrFileMissing) {
			t.Skipf("%s: %v", cfg.Name, err)
		}
		t.Fatalf("%s: %v", cfg.Name, err)
	}

	AssertModelIdentity(t, row, path)

	return path
}

// AssertModelIdentity verifies that path is the file row declares and that the
// GGUF header agrees, then claims the file for this row process-wide.
func AssertModelIdentity(t *testing.T, row modelset.Row, path string) {
	t.Helper()

	id, err := modelset.Inspect(path)
	if err != nil {
		t.Fatalf("row %q: %v", row.Key, err)
	}
	if err := row.VerifyIdentity(id); err != nil {
		t.Fatalf("row %q identity check failed: %v", row.Key, err)
	}
	if err := modelset.RecordResolution(row.Key, path); err != nil {
		t.Fatalf("row %q: %v", row.Key, err)
	}
	t.Logf("matrix row %q -> %s (general.architecture=%q general.name=%q size=%d)",
		row.Key, id.Path, id.Architecture, id.Name, id.SizeBytes)
}

// EnvOrSkip returns the value of the named env var, or skips the test.
func EnvOrSkip(t *testing.T, key string) string {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		t.Skipf("%s not set; skipping", key)
	}
	return v
}

// ModelDirOrSkip resolves a model directory from env vars, or skips the test.
//
// Deprecated: a directory alone is not enough to identify a model, because the
// flagship GGUFs are staged flat in one directory and inference.Load resolves a
// directory through findGGUF (first .gguf wins). Parity and bench code must use
// ResolveMatrixModelOrSkip, which returns an exact, identity-checked file path.
func ModelDirOrSkip(t *testing.T, dirEnvVar, zmfEnvVar string) string {
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

// GenerationTestConfig holds parameters for generation tests via inference API.
type GenerationTestConfig struct {
	// ModelID labels the model under test.
	ModelID string
	// ModelPath is the absolute path of the exact GGUF file to load. It is a
	// FILE, never a directory: directory-based loading funnels through
	// inference.findGGUF, which returns the first .gguf it finds.
	ModelPath string
}

// RunGenerationTests runs greedy, stream, and chat tests on an inference.Model.
func RunGenerationTests(t *testing.T, cfg GenerationTestConfig) {
	t.Helper()

	if cfg.ModelPath == "" {
		t.Fatal("GenerationTestConfig.ModelPath is empty; parity tests must name an exact GGUF file")
	}

	t.Logf("%s: loading %s", cfg.ModelID, cfg.ModelPath)
	mdl, err := inference.LoadFile(cfg.ModelPath)
	if err != nil {
		t.Fatalf("inference.LoadFile(%s) failed: %v", cfg.ModelPath, err)
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
