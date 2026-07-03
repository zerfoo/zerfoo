// Command bench_train benchmarks PatchTST GPU training on synthetic data.
//
// Usage:
//
//	bench_train [-samples 28000] [-channels 20] [-epochs 10] [-seed 42]
//
// Seeding (T135.5 / plan-gpu-training-hardening.md T4.1): -seed governs BOTH
// weight initialization (via timeseries.SeedWeightInit, called before the
// model is constructed) and synthetic data generation, so two invocations
// with the same -seed are the reproducible baseline the
// ZTENSOR_DETERMINISTIC=1 bitwise-identity proof depends on.
//
// Weight init specifically needed its own seeding hook: timeseries/*.go's
// Xavier/He init and PatchTST's positional embedding call math/rand/v2's
// top-level convenience functions, which v2 deliberately made unseedable (no
// global Seed function exists in math/rand/v2, unlike classic math/rand) --
// so every model construction was non-reproducible regardless of any other
// seeding, until timeseries.SeedWeightInit existed.
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	ts "github.com/zerfoo/zerfoo/timeseries"
)

func main() {
	nSamples := flag.Int("samples", 28000, "number of training samples")
	nChannels := flag.Int("channels", 20, "number of channels per sample")
	epochs := flag.Int("epochs", 10, "training epochs")
	batchSize := flag.Int("batch-size", 64, "batch size")
	lr := flag.Float64("lr", 1e-3, "learning rate")
	seed := flag.Int64("seed", 42, "seed for weight init and synthetic data generation")
	cpuOnly := flag.Bool("cpu", false, "force CPU engine")
	outFile := flag.String("out", "", "write results to file (unbuffered)")
	flag.Parse()

	// Reseed weight init BEFORE constructing the model (timeseries.NewPatchTST
	// below draws from this generator during Xavier/He init and positional-
	// embedding init). Two uint64s from the same seed: NewPCG takes a 128-bit
	// seed; the exact expansion doesn't matter for reproducibility, only that
	// it's a pure function of -seed.
	ts.SeedWeightInit(uint64(*seed), uint64(*seed)<<32|uint64(*seed))

	// If -out specified, tee all output to the file (unbuffered).
	if *outFile != "" {
		f, err := os.Create(*outFile)
		if err != nil {
			log.Fatalf("create output file: %v", err)
		}
		defer f.Close()
		log.SetOutput(io.MultiWriter(os.Stderr, f))
		// Flush after each log line by wrapping the file.
		defer f.Sync()
	}
	// Force line-buffered log output.
	log.SetFlags(log.Ldate | log.Ltime)

	inputLen := 24
	config := ts.PatchTSTConfig{
		InputLength: inputLen,
		PatchLength: 8,
		Stride:      4,
		DModel:      64,
		NHeads:      4,
		NLayers:     2,
		OutputDim:   1,
	}

	ops := numeric.Float32Ops{}
	var engine compute.Engine[float32]

	if !*cpuOnly {
		gpuEng, err := compute.NewGPUEngine[float32](ops)
		if err == nil {
			engine = gpuEng
			defer gpuEng.Close()
			log.Printf("engine: GPU (CUDA)")
		} else {
			log.Printf("GPU not available: %v; falling back to CPU", err)
		}
	}
	if engine == nil {
		engine = compute.NewCPUEngine[float32](ops)
		log.Printf("engine: CPU")
	}

	model, err := ts.NewPatchTST(config, engine, ops)
	if err != nil {
		log.Fatalf("NewPatchTST: %v", err)
	}

	// Generate synthetic data: random walk per channel. A separate,
	// explicitly-seeded generator (not the global source rand.Seed just
	// configured above) so data generation is reproducible independent of
	// how many global-source draws weight init happens to consume.
	rng := rand.New(rand.NewSource(*seed))
	windows := make([][][]float64, *nSamples)
	labels := make([]float64, *nSamples*config.OutputDim)
	for s := 0; s < *nSamples; s++ {
		windows[s] = make([][]float64, *nChannels)
		sum := 0.0
		for c := 0; c < *nChannels; c++ {
			windows[s][c] = make([]float64, inputLen)
			val := rng.Float64()
			for i := 0; i < inputLen; i++ {
				val += rng.NormFloat64() * 0.1
				windows[s][c][i] = val
				sum += val
			}
		}
		labels[s] = sum / float64(*nChannels*inputLen)
	}

	log.Printf("config: %d samples x %d channels x %d input_len x %d epochs (batch=%d, lr=%.1e)",
		*nSamples, *nChannels, inputLen, *epochs, *batchSize, *lr)
	log.Printf("model: dModel=%d nHeads=%d nLayers=%d patchLen=%d stride=%d",
		config.DModel, config.NHeads, config.NLayers, config.PatchLength, config.Stride)

	start := time.Now()
	result, err := model.TrainWindowed(windows, labels, ts.TrainConfig{
		Epochs:    *epochs,
		LR:        *lr,
		GradClip:  1.0,
		BatchSize: *batchSize,
	})
	elapsed := time.Since(start)

	if err != nil {
		log.Fatalf("TrainWindowed: %v", err)
	}

	perEpoch := elapsed / time.Duration(*epochs)
	log.Printf("total: %v (%v/epoch)", elapsed, perEpoch)

	for i, l := range result.LossHistory {
		finite := "ok"
		if math.IsNaN(l) || math.IsInf(l, 0) {
			finite = "NOT FINITE"
		}
		// Print the exact IEEE-754 bit pattern alongside the rounded decimal:
		// a ZTENSOR_DETERMINISTIC double-run bitwise-identity proof must diff
		// bits, not %.6f-rounded text, which would hide any divergence below
		// the printed precision (plan-gpu-training-hardening.md T4.1).
		fmt.Printf("epoch %2d: loss=%.6f bits=0x%016x %s\n", i+1, l, math.Float64bits(l), finite)
	}

	if len(result.LossHistory) >= 2 {
		first := result.LossHistory[0]
		last := result.LossHistory[len(result.LossHistory)-1]
		if last < first {
			log.Printf("convergence: OK (%.6f -> %.6f, %.1f%% reduction)", first, last, (1-last/first)*100)
		} else {
			log.Printf("convergence: FAILED (loss did not decrease: %.6f -> %.6f)", first, last)
		}
	}
}
