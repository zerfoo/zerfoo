// Command bench_mamba benchmarks Mamba-3 SSM vs Transformer attention decode
// throughput using synthetic FLOPs-based timing estimates. No GPU required.
//
// Mamba-3 SSM has O(1) per-token decode cost (state recurrence), while
// Transformer attention has O(n) cost per token (KV cache scan grows with
// sequence length). This benchmark quantifies the throughput advantage at
// sequence lengths 512, 2048, and 8192.
//
// Usage:
//
//	bench_mamba [--layers 24] [--d-model 2048] [--d-state 16] [--d-inner 4096] [--heads 16] [--head-dim 128] [--gpu-tflops 150]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"time"
)

// ModelConfig holds architecture dimensions shared between Mamba and Transformer.
type ModelConfig struct {
	Layers   int
	DModel   int
	DState   int
	DInner   int
	DConv    int
	Heads    int
	HeadDim  int
	GPUTFlop float64 // peak GPU TFLOPS (FP16)
}

// SeqResult holds benchmark results for one sequence length.
type SeqResult struct {
	SeqLen          int     `json:"seq_len"`
	MambaFLOPs      float64 `json:"mamba_flops"`
	TransformerFLOPs float64 `json:"transformer_flops"`
	MambaTokPerSec  float64 `json:"mamba_tok_per_sec"`
	TransTokPerSec  float64 `json:"transformer_tok_per_sec"`
	Speedup         float64 `json:"speedup"`
}

// BenchReport is the full benchmark output.
type BenchReport struct {
	Config    ModelConfig `json:"config"`
	Results   []SeqResult `json:"results"`
	Commit    string      `json:"commit"`
	Timestamp string      `json:"timestamp"`
}

// mambaDecodeFLOPs estimates FLOPs per token for Mamba-3 decode.
// Per layer: in_proj (2 * d_model * 2*d_inner) + conv1d (d_inner * d_conv) +
// SSM scan (d_inner * d_state * 6) + out_proj (2 * d_inner * d_model).
// This is O(1) in sequence length — the recurrent state is fixed-size.
func mambaDecodeFLOPs(cfg ModelConfig) float64 {
	perLayer := float64(0)
	// Input projection: [d_model] -> [2*d_inner]
	perLayer += 2 * float64(cfg.DModel) * float64(2*cfg.DInner)
	// Conv1d: d_inner channels, kernel size d_conv
	perLayer += float64(cfg.DInner) * float64(cfg.DConv)
	// SSM discretize + scan: ~6 * d_inner * d_state
	perLayer += 6 * float64(cfg.DInner) * float64(cfg.DState)
	// Output projection: [d_inner] -> [d_model]
	perLayer += 2 * float64(cfg.DInner) * float64(cfg.DModel)

	return perLayer * float64(cfg.Layers)
}

// transformerDecodeFLOPs estimates FLOPs per token for Transformer decode
// at a given sequence length. Per layer:
// QKV projection: 3 * 2 * d_model * d_model
// Attention scores: 2 * heads * head_dim * seq_len (O(n) in seq_len)
// Attention @ V: 2 * heads * seq_len * head_dim
// Output projection: 2 * d_model * d_model
// FFN: 2 * 2 * d_model * 4*d_model (SwiGLU style, ~8/3 * d_model but simplified)
func transformerDecodeFLOPs(cfg ModelConfig, seqLen int) float64 {
	d := float64(cfg.DModel)
	h := float64(cfg.Heads)
	hd := float64(cfg.HeadDim)
	s := float64(seqLen)

	perLayer := float64(0)
	// QKV projections
	perLayer += 3 * 2 * d * d
	// Attention: Q*K^T scores + softmax@V
	perLayer += 2 * h * hd * s // Q*K^T
	perLayer += 2 * h * s * hd // attn@V
	// Output projection
	perLayer += 2 * d * d
	// FFN (SwiGLU: gate + up + down, ~3 projections of size d * 4d)
	perLayer += 3 * 2 * d * (4 * d)

	return perLayer * float64(cfg.Layers)
}

// tokPerSec computes throughput from FLOPs per token and GPU TFLOPS.
// Assumes 30% compute utilization (typical for decode, memory-bound).
func tokPerSec(flopsPerToken, gpuTFlops float64) float64 {
	utilization := 0.30
	peakFlops := gpuTFlops * 1e12
	effectiveFlops := peakFlops * utilization
	return effectiveFlops / flopsPerToken
}

// runBenchmark computes throughput estimates for all sequence lengths.
func runBenchmark(cfg ModelConfig) BenchReport {
	seqLens := []int{512, 2048, 8192}
	results := make([]SeqResult, len(seqLens))

	mambaFlops := mambaDecodeFLOPs(cfg)

	for i, seqLen := range seqLens {
		transFlops := transformerDecodeFLOPs(cfg, seqLen)
		mambaTPS := tokPerSec(mambaFlops, cfg.GPUTFlop)
		transTPS := tokPerSec(transFlops, cfg.GPUTFlop)

		results[i] = SeqResult{
			SeqLen:          seqLen,
			MambaFLOPs:      mambaFlops,
			TransformerFLOPs: transFlops,
			MambaTokPerSec:  math.Round(mambaTPS*100) / 100,
			TransTokPerSec:  math.Round(transTPS*100) / 100,
			Speedup:         math.Round(transTPS/mambaTPS*100) / 100, // will invert below
		}
	}

	// Speedup = Mamba tok/s / Transformer tok/s
	for i := range results {
		if results[i].TransTokPerSec > 0 {
			results[i].Speedup = math.Round(results[i].MambaTokPerSec/results[i].TransTokPerSec*100) / 100
		}
	}

	return BenchReport{
		Config:    cfg,
		Results:   results,
		Commit:    gitCommitHash(),
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
}

// printReport prints a formatted table of results.
func printReport(r BenchReport) {
	fmt.Println()
	fmt.Println("=== Mamba-3 vs Transformer Decode Throughput ===")
	fmt.Printf("Layers: %d  DModel: %d  DState: %d  DInner: %d\n",
		r.Config.Layers, r.Config.DModel, r.Config.DState, r.Config.DInner)
	fmt.Printf("Heads: %d  HeadDim: %d  GPU: %.0f TFLOPS (FP16)\n",
		r.Config.Heads, r.Config.HeadDim, r.Config.GPUTFlop)
	fmt.Println()
	fmt.Printf("%-10s %15s %15s %10s\n", "SeqLen", "Mamba tok/s", "Trans tok/s", "Speedup")
	fmt.Println(strings.Repeat("-", 52))
	for _, res := range r.Results {
		fmt.Printf("%-10d %15.2f %15.2f %9.2fx\n",
			res.SeqLen, res.MambaTokPerSec, res.TransTokPerSec, res.Speedup)
	}
	fmt.Println()

	if len(r.Results) > 0 {
		last := r.Results[len(r.Results)-1]
		if last.Speedup >= 2.0 {
			fmt.Printf("PASS: %.2fx Mamba speedup at seq=%d (>= 2x target)\n", last.Speedup, last.SeqLen)
		} else {
			fmt.Printf("NOTE: %.2fx Mamba speedup at seq=%d (target: >= 2x)\n", last.Speedup, last.SeqLen)
		}
	}
}

// gitCommitHash returns the short git commit hash.
func gitCommitHash() string {
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func parseFlags(args []string) (ModelConfig, string, error) {
	fs := flag.NewFlagSet("bench_mamba", flag.ContinueOnError)
	cfg := ModelConfig{DConv: 4}
	var output string

	fs.IntVar(&cfg.Layers, "layers", 24, "number of layers")
	fs.IntVar(&cfg.DModel, "d-model", 2048, "model dimension")
	fs.IntVar(&cfg.DState, "d-state", 16, "SSM state dimension")
	fs.IntVar(&cfg.DInner, "d-inner", 4096, "inner dimension (2*d_model)")
	fs.IntVar(&cfg.Heads, "heads", 16, "number of attention heads (Transformer)")
	fs.IntVar(&cfg.HeadDim, "head-dim", 128, "attention head dimension (Transformer)")
	fs.Float64Var(&cfg.GPUTFlop, "gpu-tflops", 150, "peak GPU TFLOPS (FP16)")
	fs.StringVar(&output, "output", "bench_mamba_results.json", "path for JSON results file")

	if err := fs.Parse(args); err != nil {
		return cfg, output, err
	}
	return cfg, output, nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, output, err := parseFlags(os.Args[1:])
	if err != nil {
		return err
	}

	report := runBenchmark(cfg)
	printReport(report)

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	if err := os.WriteFile(output, data, 0o644); err != nil {
		return fmt.Errorf("write results: %w", err)
	}
	fmt.Printf("Results written to %s\n", output)
	return nil
}
