package parity_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/inference"
	layerreg "github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/registry"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"
)

// Q4 model tests use separate env vars:
//   GEMMA3_Q4_ZMF_PATH  = path to Q4 quantized model.zmf
//   GEMMA3_Q4_MODEL_DIR = directory containing config.json, tokenizer.json, model.zmf (Q4)

var gemma3Q4Config = testutil.ModelParityConfig{
	Name:           "Gemma 3 Q4",
	ZMFEnvVar:      "GEMMA3_Q4_ZMF_PATH",
	ModelDirEnvVar: "GEMMA3_Q4_MODEL_DIR",
	ModelID:        "gemma-3-q4",
	MinVocabSize:   256000,
}

func TestGemma3Q4ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, gemma3Q4Config)
}

func TestGemma3Q4GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, gemma3Q4Config)
}

func TestGemma3Q4Generation(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, gemma3Q4Config)
}

func BenchmarkGemma3Q4TokPerSec(b *testing.B) {
	layerreg.RegisterAll()

	modelDir := os.Getenv("GEMMA3_Q4_MODEL_DIR")
	if modelDir == "" {
		b.Skip("GEMMA3_Q4_MODEL_DIR not set; skipping")
	}

	reg := &testutil.DirRegistry{
		Models: map[string]*registry.ModelInfo{
			gemma3Q4Config.ModelID: {ID: gemma3Q4Config.ModelID, Path: modelDir},
		},
	}

	mdl, err := inference.Load(gemma3Q4Config.ModelID, inference.WithRegistry(reg))
	if err != nil {
		b.Fatalf("inference.Load failed: %v", err)
	}

	ctx := context.Background()
	prompt := "The meaning of life is"
	maxTokens := 32

	b.ResetTimer()
	for b.Loop() {
		start := time.Now()
		_, err := mdl.Generate(ctx, prompt,
			inference.WithTemperature(0),
			inference.WithMaxTokens(maxTokens),
		)
		elapsed := time.Since(start)
		if err != nil {
			b.Fatalf("Generate failed: %v", err)
		}
		b.ReportMetric(float64(maxTokens)/elapsed.Seconds(), "tok/s")
	}
}

// BenchmarkSpeculativeVsBaseline compares speculative decoding against
// baseline greedy decode. Uses the same model as both draft and target
// (realistic benchmarks require a smaller draft model).
func BenchmarkSpeculativeVsBaseline(b *testing.B) {
	layerreg.RegisterAll()

	modelDir := os.Getenv("GEMMA3_Q4_MODEL_DIR")
	if modelDir == "" {
		b.Skip("GEMMA3_Q4_MODEL_DIR not set; skipping")
	}

	reg := &testutil.DirRegistry{
		Models: map[string]*registry.ModelInfo{
			gemma3Q4Config.ModelID: {ID: gemma3Q4Config.ModelID, Path: modelDir},
		},
	}

	mdl, err := inference.Load(gemma3Q4Config.ModelID, inference.WithRegistry(reg))
	if err != nil {
		b.Fatalf("inference.Load failed: %v", err)
	}

	ctx := context.Background()
	prompt := "The meaning of life is"
	maxTokens := 32

	b.Run("baseline", func(b *testing.B) {
		for b.Loop() {
			start := time.Now()
			_, err := mdl.Generate(ctx, prompt,
				inference.WithTemperature(0),
				inference.WithMaxTokens(maxTokens),
			)
			elapsed := time.Since(start)
			if err != nil {
				b.Fatalf("Generate failed: %v", err)
			}
			b.ReportMetric(float64(maxTokens)/elapsed.Seconds(), "tok/s")
		}
	})

	b.Run("speculative_k4", func(b *testing.B) {
		for b.Loop() {
			start := time.Now()
			_, err := mdl.SpeculativeGenerate(ctx, mdl, prompt, 4,
				inference.WithMaxTokens(maxTokens),
			)
			elapsed := time.Since(start)
			if err != nil {
				b.Fatalf("SpeculativeGenerate failed: %v", err)
			}
			b.ReportMetric(float64(maxTokens)/elapsed.Seconds(), "tok/s")
		}
	})
}
