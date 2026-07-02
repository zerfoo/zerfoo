// Package benchmark provides inference performance benchmarks.
//
// Run with: go test -bench=. -run=^$ ./tests/benchmark/
//
// Set BENCH_MODEL_PATH to a ZMF model file to benchmark actual inference.
// Without a model, the benchmark measures the framework overhead only.
package benchmark

import (
	"context"
	"os"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func BenchmarkTokPerSec_Decode(b *testing.B) {
	modelPath := os.Getenv("BENCH_MODEL_PATH")
	if modelPath == "" {
		b.Skip("BENCH_MODEL_PATH not set; skipping inference benchmark")
	}

	// Placeholder: when model loading is wired, load and generate here.
	// For now, this benchmark measures the generation loop with a dummy model.
	b.Skip("model loading not yet wired into benchmark harness")
}

func BenchmarkTokPerSec_Prefill(b *testing.B) {
	modelPath := os.Getenv("BENCH_MODEL_PATH")
	if modelPath == "" {
		b.Skip("BENCH_MODEL_PATH not set; skipping inference benchmark")
	}

	b.Skip("model loading not yet wired into benchmark harness")
}

// BenchmarkGenerateLoop measures the raw token generation loop overhead
// without a real model, using a dummy tensor pipeline.
func BenchmarkGenerateLoop(b *testing.B) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	vocabSize := 32000
	seqLen := 128
	hiddenDim := 64

	// Create a dummy "logits" tensor.
	logitsData := make([]float32, vocabSize)
	for i := range logitsData {
		logitsData[i] = float32(i) * 0.0001
	}
	logits, err := tensor.New([]int{1, vocabSize}, logitsData)
	if err != nil {
		b.Fatal(err)
	}

	for b.Loop() {
		// Simulate token generation loop.
		for step := range seqLen {
			// Argmax over logits (simulated).
			data := logits.Data()
			maxIdx := 0
			maxVal := data[0]
			for j := 1; j < len(data); j++ {
				if data[j] > maxVal {
					maxVal = data[j]
					maxIdx = j
				}
			}
			_ = maxIdx

			// Simulate a MatMul (hidden_dim x hidden_dim).
			a := make([]float32, hiddenDim*hiddenDim)
			bm := make([]float32, hiddenDim*hiddenDim)
			c := make([]float32, hiddenDim*hiddenDim)
			for j := range a {
				a[j] = float32(j%7-3) * 0.01
				bm[j] = float32(j%5-2) * 0.01
			}
			aTensor, _ := tensor.New([]int{hiddenDim, hiddenDim}, a)
			bTensor, _ := tensor.New([]int{hiddenDim, hiddenDim}, bm)
			_, _ = engine.MatMul(ctx, aTensor, bTensor)
			_ = c
			_ = step
		}
	}

	elapsed := b.Elapsed()
	toksPerSec := float64(b.N*seqLen) / elapsed.Seconds()
	b.ReportMetric(toksPerSec, "tok/s")
}

// BenchmarkKVCacheUpdate measures KV cache append performance.
func BenchmarkKVCacheUpdate(b *testing.B) {
	maxSeqLen := 2048
	numLayers := 32
	dim := 32 * 64 // numHeads * headDim

	cache := generate.NewKVCache[float32](numLayers, maxSeqLen)

	newK, _ := tensor.New([]int{1, 1, dim}, make([]float32, dim))
	newV, _ := tensor.New([]int{1, 1, dim}, make([]float32, dim))

	for b.Loop() {
		cache.Reset()
		for step := range maxSeqLen {
			_ = cache.Update(0, newK, newV)
			_ = step
		}
	}

	elapsed := b.Elapsed()
	updatesPerSec := float64(b.N*maxSeqLen) / elapsed.Seconds()
	b.ReportMetric(updatesPerSec, "updates/s")
}

// BenchmarkMemoryAllocs measures allocations during a simulated generation loop.
func BenchmarkMemoryAllocs(b *testing.B) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	dim := 64
	steps := 32

	a, _ := tensor.New([]int{dim, dim}, make([]float32, dim*dim))
	bm, _ := tensor.New([]int{dim, dim}, make([]float32, dim*dim))

	b.ReportAllocs()

	for b.Loop() {
		for range steps {
			_, _ = engine.MatMul(ctx, a, bm)
		}
	}
	elapsed := b.Elapsed()
	allocsPerToken := float64(b.N*steps) / elapsed.Seconds()
	b.ReportMetric(allocsPerToken, "ops/s")
}
