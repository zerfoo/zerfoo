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
	"github.com/zerfoo/ztensor/tensor"
)

func printFirst(label string, data []float32, n int) {
	if len(data) < n {
		n = len(data)
	}
	log.Printf("%s (first %d): %v", label, n, data[:n])
}

func l2norm(data []float32) float64 {
	var sum float64
	for _, v := range data {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum)
}

func rmsNorm(data []float32, weight []float32, eps float32) []float32 {
	var sumSq float64
	for _, v := range data {
		sumSq += float64(v) * float64(v)
	}
	rms := float32(math.Sqrt(sumSq/float64(len(data)) + float64(eps)))
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = (v / rms) * weight[i]
	}
	return out
}

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "What is 2+2?", "Prompt text")
	maxTokens := flag.Int("max-tokens", 20, "Max tokens to generate")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("--model is required")
	}

	// Load GGUF separately to inspect config and tensors.
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

	// === LAYER-BY-LAYER COMPARISON ===
	log.Printf("=== LAYER-BY-LAYER COMPARISON ===")

	// 1. Embedding lookup for BOS=2
	emb, ok := gm.Tensors["model.embed_tokens.weight"]
	if !ok {
		log.Fatal("missing embedding tensor")
	}
	embData := emb.Data()
	hiddenSize := cfg.HiddenSize
	bosID := 2
	bosEmb := make([]float32, hiddenSize)
	copy(bosEmb, embData[bosID*hiddenSize:(bosID+1)*hiddenSize])
	printFirst("BOS raw embedding", bosEmb, 20)
	log.Printf("BOS embedding L2: %.6f", l2norm(bosEmb))

	// 2. Scale by sqrt(hiddenSize) (Gemma)
	embedScale := float32(math.Sqrt(float64(hiddenSize)))
	scaled := make([]float32, hiddenSize)
	for i, v := range bosEmb {
		scaled[i] = v * embedScale
	}
	printFirst("Scaled embedding", scaled, 20)
	log.Printf("Scaled embedding L2: %.6f", l2norm(scaled))

	// 3. Layer 0 input_layernorm
	lnW, ok := gm.Tensors["model.layers.0.input_layernorm.weight"]
	if !ok {
		log.Fatal("missing layer 0 input_layernorm.weight")
	}
	lnWData := lnW.Data()
	printFirst("LN weight", lnWData, 20)

	normed := rmsNorm(scaled, lnWData, cfg.RMSNormEps)
	printFirst("After RMSNorm", normed, 20)
	log.Printf("Normed L2: %.6f", l2norm(normed))

	// 4. Q/K/V projections (manual MatVec)
	qW, ok := gm.Tensors["model.layers.0.self_attn.q_proj.weight"]
	if !ok {
		log.Fatal("missing q_proj.weight")
	}
	qWData := qW.Data()
	qWShape := qW.Shape()
	log.Printf("Q weight shape: %v", qWShape)

	// Q weight is [outDim, inDim] = [1024, 1152]
	// Q = normed @ W_q^T, or equivalently: for each row r of W_q, Q[r] = dot(normed, W_q[r])
	outDim := qWShape[0]
	inDim := qWShape[1]
	qOut := make([]float32, outDim)
	for r := range outDim {
		var dot float64
		for c := range inDim {
			dot += float64(normed[c]) * float64(qWData[r*inDim+c])
		}
		qOut[r] = float32(dot)
	}
	printFirst("Q projection", qOut, 20)
	log.Printf("Q output L2: %.6f", l2norm(qOut))

	kW, ok := gm.Tensors["model.layers.0.self_attn.k_proj.weight"]
	if ok {
		kWData := kW.Data()
		kShape := kW.Shape()
		kOutDim := kShape[0]
		kInDim := kShape[1]
		kOut := make([]float32, kOutDim)
		for r := range kOutDim {
			var dot float64
			for c := range kInDim {
				dot += float64(normed[c]) * float64(kWData[r*kInDim+c])
			}
			kOut[r] = float32(dot)
		}
		printFirst("K projection", kOut, 20)
	}

	// === FULL MODEL DIAGNOSTIC ===
	log.Printf("=== FULL MODEL DIAGNOSTIC ===")
	log.Printf("Building model...")
	start := time.Now()
	mdl, err := inference.LoadFile(*modelPath)
	if err != nil {
		log.Fatalf("LoadFile: %v", err)
	}
	log.Printf("Model loaded in %v", time.Since(start))

	// Single forward pass with BOS token.
	log.Printf("=== Single forward pass with BOS token ===")
	diagInput, _ := tensor.New[float32]([]int{1, 1}, []float32{2})
	gen := mdl.Generator()
	diagCtx := generate.WithKVCache(context.Background(), generate.NewKVCache[float32](cfg.NumLayers, 128))
	diagLogits, diagErr := gen.Graph().Forward(diagCtx, diagInput)
	if diagErr != nil {
		log.Printf("Diagnostic forward error: %v", diagErr)
	} else {
		logitData := diagLogits.Data()
		log.Printf("Logits shape: %v, len: %d", diagLogits.Shape(), len(logitData))

		var minVal, maxVal = math.Inf(1), math.Inf(-1)
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

	// Generation test
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
