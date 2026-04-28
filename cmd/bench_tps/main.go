// bench_tps measures tokens-per-second for a local ZMF model.
//
// Usage: bench_tps -model /path/to/model/dir [-prompt "text"] [-tokens 64]
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"strings"
	"sync/atomic"
	"time"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
	layerreg "github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/model/registry"
)

// dirRegistry is a minimal ModelRegistry that returns a fixed directory.
type dirRegistry struct {
	path string
}

func (r *dirRegistry) Get(modelID string) (*registry.ModelInfo, bool) {
	if _, err := os.Stat(r.path); err != nil {
		return nil, false
	}
	return &registry.ModelInfo{
		ID:   modelID,
		Path: r.path,
	}, true
}

func (r *dirRegistry) Pull(_ context.Context, _ string) (*registry.ModelInfo, error) {
	return nil, fmt.Errorf("pull not supported")
}

func (r *dirRegistry) List() []registry.ModelInfo { return nil }

func (r *dirRegistry) Delete(_ string) error { return nil }

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func run() error {
	layerreg.RegisterAll()

	modelDir := flag.String("model", "", "path to model directory (config.json, tokenizer.json, model.zmf)")
	prompt := flag.String("prompt", "The meaning of life is", "prompt text")
	maxTokens := flag.Int("tokens", 64, "max tokens to generate")
	noMmap := flag.Bool("no-mmap", false, "disable memory-mapped loading (mmap is on by default)")
	maxSeqLen := flag.Int("max-seq-len", 0, "override max sequence length / KV cache size (0 = use model default)")
	device := flag.String("device", "cpu", "compute device (cpu, cuda, cuda:0)")
	dtype := flag.String("dtype", "fp32", "compute precision (fp32, fp16, fp8)")
	kvDtype := flag.String("kv-dtype", "fp32", "KV cache dtype (fp32, fp16)")
	temperature := flag.Float64("temp", 0, "sampling temperature (0=greedy)")
	repetitionPenalty := flag.Float64("repetition-penalty", 1.0, "repetition penalty (1.0=disabled)")
	cpuprofile := flag.String("cpuprofile", "", "write CPU profile to file")
	flag.Parse()

	// Auto-disable mmap on CUDA devices. MmapStorage has alignment and
	// device-consistency issues on ARM64/Grace Hopper that produce garbage
	// output. Non-mmap loading reads weights into heap memory which the
	// GPU engine handles correctly. See docs/devlog.md 2026-03-30 entries.
	if strings.HasPrefix(*device, "cuda") && !*noMmap {
		*noMmap = true
	}

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			return fmt.Errorf("cpuprofile: %w", err)
		}
		defer func() { _ = f.Close() }()
		if err := pprof.StartCPUProfile(f); err != nil {
			return fmt.Errorf("cpuprofile start: %w", err)
		}
		defer pprof.StopCPUProfile()
	}

	if *modelDir == "" {
		return fmt.Errorf("usage: bench_tps -model /path/to/model/dir/or/file.gguf")
	}

	fmt.Printf("Loading model from %s (device=%s, dtype=%s, kv-dtype=%s)...\n", *modelDir, *device, *dtype, *kvDtype)
	t0 := time.Now()

	var mdl *inference.Model
	var err error
	opts := []inference.Option{
		inference.WithMmap(!*noMmap),
		inference.WithDevice(*device),
		inference.WithDType(*dtype),
		inference.WithKVDtype(*kvDtype),
	}
	if *maxSeqLen > 0 {
		opts = append(opts, inference.WithMaxSeqLen(*maxSeqLen))
	}
	if strings.HasSuffix(strings.ToLower(*modelDir), ".gguf") {
		mdl, err = inference.LoadFile(*modelDir, opts...)
	} else {
		reg := &dirRegistry{path: *modelDir}
		mdl, err = inference.Load("bench", append(opts, inference.WithRegistry(reg))...)
	}
	if err != nil {
		return fmt.Errorf("load error: %w", err)
	}
	fmt.Printf("Loaded in %.1fs\n", time.Since(t0).Seconds())

	fmt.Printf("Prompt: %q\n", *prompt)
	fmt.Printf("Max tokens: %d\n", *maxTokens)

	// Warm-up run (short).
	fmt.Println("Warm-up...")
	_, _ = mdl.Generate(context.Background(), *prompt, inference.WithMaxTokens(4), inference.WithTemperature(*temperature), inference.WithRepetitionPenalty(*repetitionPenalty))

	// Timed run with streaming to count tokens.
	fmt.Printf("Generating (temp=%.1f)...\n", *temperature)
	var tokenCount atomic.Int64
	var output string
	handler := generate.TokenStreamFunc(func(token string, done bool) error {
		if !done {
			tokenCount.Add(1)
			output += token
		}
		return nil
	})

	t1 := time.Now()
	err = mdl.GenerateStream(context.Background(), *prompt, handler, inference.WithMaxTokens(*maxTokens), inference.WithTemperature(*temperature), inference.WithRepetitionPenalty(*repetitionPenalty))
	elapsed := time.Since(t1)
	if err != nil {
		return fmt.Errorf("generate error: %w", err)
	}

	genTokens := tokenCount.Load()
	tps := float64(genTokens) / elapsed.Seconds()

	fmt.Printf("\n--- Results ---\n")
	fmt.Printf("Output: %s\n", output)
	fmt.Printf("Generated tokens: %d\n", genTokens)
	fmt.Printf("Time: %.3fs\n", elapsed.Seconds())
	fmt.Printf("Throughput: %.2f tok/s\n", tps)

	// Print cuBLAS profiling stats if ZERFOO_PROFILE_CUBLAS=1.
	gpuapi.PrintCUBLASProfile()

	// Print GPU memory pool stats if available.
	if arena := cuda.DefaultArenaPool(); arena != nil {
		hits, misses, resets := arena.HitMissStats()
		used := arena.UsedBytes()
		fmt.Printf("\nGPU Arena: hits=%d misses=%d resets=%d used=%.1f MB\n",
			hits, misses, resets, float64(used)/1e6)
	}
	if pool := cuda.DefaultMemPool(); pool != nil {
		hits, misses, frees := pool.HitMissStats()
		allocs, cachedBytes := pool.Stats()
		fmt.Printf("GPU MemPool (fallback): hits=%d misses=%d frees=%d cached=%d (%.1f MB)\n",
			hits, misses, frees, allocs, float64(cachedBytes)/1e6)
	}
	return nil
}
