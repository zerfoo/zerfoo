package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"sort"
	"time"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/tensor"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "What is 2+2?", "Prompt text")
	maxTokens := flag.Int("max-tokens", 20, "Max tokens to generate")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("--model is required")
	}

	// Load GGUF separately to inspect config.
	log.Printf("Loading GGUF from %s...", *modelPath)
	gm, err := inference.LoadGGUF(*modelPath)
	if err != nil {
		log.Fatalf("LoadGGUF: %v", err)
	}
	cfg := gm.Config
	log.Printf("Architecture: %s", cfg.Architecture)
	log.Printf("HiddenSize: %d, NumLayers: %d", cfg.HiddenSize, cfg.NumLayers)
	log.Printf("NumHeads: %d, NumKVHeads: %d, HeadDim: %d", cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	log.Printf("IntermediateSize: %d, VocabSize: %d", cfg.IntermediateSize, cfg.VocabSize)
	log.Printf("RopeTheta: %f, LocalRopeTheta: %f", cfg.RopeTheta, cfg.LocalRopeTheta)
	log.Printf("SlidingWindow: %d, SlidingWindowPattern: %d", cfg.SlidingWindow, cfg.SlidingWindowPattern)
	log.Printf("LogitSoftcap: %f, RMSNormEps: %e", cfg.LogitSoftcap, cfg.RMSNormEps)
	log.Printf("MaxSeqLen: %d", cfg.MaxSeqLen)

	// Print tensor names and shapes.
	log.Printf("Loaded %d tensors:", len(gm.Tensors))
	for name, t := range gm.Tensors {
		log.Printf("  %s: shape=%v", name, t.Shape())
	}

	// Print first few values of embedding weight to verify dequantization.
	if emb, ok := gm.Tensors["model.embed_tokens.weight"]; ok {
		data := emb.Data()
		n := 10
		if len(data) < n {
			n = len(data)
		}
		log.Printf("Embedding weight first %d values: %v", n, data[:n])
	}

	// Dump first 20 dequantized values of a few weight tensors to verify.
	for _, tname := range []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.mlp.gate_proj.weight",
	} {
		if t, ok := gm.Tensors[tname]; ok {
			data := t.Data()
			n := 20
			if len(data) < n {
				n = len(data)
			}
			log.Printf("%s first %d values: %v", tname, n, data[:n])
		}
	}

	log.Printf("Building model...")
	start := time.Now()
	mdl, err := inference.LoadFile(*modelPath)
	if err != nil {
		log.Fatalf("LoadFile: %v", err)
	}
	log.Printf("Model loaded in %v", time.Since(start))

	// --- Diagnostic: single forward pass with token ID 2 (BOS for Gemma) ---
	log.Printf("=== DIAGNOSTIC: single forward pass with BOS token ===")
	diagInput, _ := tensor.New[float32]([]int{1, 1}, []float32{2})
	gen := mdl.Generator()
	diagCtx := generate.WithKVCache(context.Background(), generate.NewKVCache[float32](cfg.NumLayers, 128))
	diagLogits, diagErr := gen.Graph().Forward(diagCtx, diagInput)
	if diagErr != nil {
		log.Printf("Diagnostic forward error: %v", diagErr)
	} else {
		logitData := diagLogits.Data()
		log.Printf("Logits shape: %v, len: %d", diagLogits.Shape(), len(logitData))

		// Find min, max, mean, NaN count.
		var minVal, maxVal float64 = math.Inf(1), math.Inf(-1)
		var sum float64
		nanCount := 0
		for _, v := range logitData {
			f := float64(v)
			if math.IsNaN(f) {
				nanCount++
				continue
			}
			if f < minVal {
				minVal = f
			}
			if f > maxVal {
				maxVal = f
			}
			sum += f
		}
		mean := sum / float64(len(logitData)-nanCount)
		log.Printf("Logits stats: min=%.4f max=%.4f mean=%.6f NaN=%d", minVal, maxVal, mean, nanCount)

		// Top 10 logits.
		type idxVal struct {
			idx int
			val float32
		}
		ivs := make([]idxVal, len(logitData))
		for i, v := range logitData {
			ivs[i] = idxVal{i, v}
		}
		sort.Slice(ivs, func(a, b int) bool { return ivs[a].val > ivs[b].val })
		log.Printf("Top 10 logits:")
		for i := 0; i < 10 && i < len(ivs); i++ {
			log.Printf("  [%d] token=%d logit=%.4f", i, ivs[i].idx, ivs[i].val)
		}
	}
	log.Printf("=== END DIAGNOSTIC ===")

	log.Printf("Prompt: %q", *prompt)
	log.Printf("Generating %d tokens...", *maxTokens)

	genStart := time.Now()
	tokenCount := 0
	err = mdl.GenerateStream(context.Background(), *prompt, generate.TokenStreamFunc(func(token string, done bool) error {
		if !done {
			fmt.Print(token)
			tokenCount++
		}
		return nil
	}), inference.WithMaxTokens(*maxTokens), inference.WithTemperature(0))
	fmt.Println()

	elapsed := time.Since(genStart)
	if err != nil {
		log.Fatalf("Generate: %v", err)
	}
	log.Printf("Generated %d tokens in %v (%.1f tok/s)", tokenCount, elapsed, float64(tokenCount)/elapsed.Seconds())
}
