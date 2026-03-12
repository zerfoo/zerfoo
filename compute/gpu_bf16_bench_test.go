package compute

import (
	"context"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func benchBF16MatMul(b *testing.B, eng Engine[float16.BFloat16], size int) {
	ctx := context.Background()
	n := size * size

	aData := make([]float16.BFloat16, n)
	bData := make([]float16.BFloat16, n)

	for i := range aData {
		aData[i] = float16.BFloat16FromFloat32(float32(i%100) * 0.01)
		bData[i] = float16.BFloat16FromFloat32(float32(i%100) * 0.01)
	}

	a, _ := tensor.New[float16.BFloat16]([]int{size, size}, aData)
	bt, _ := tensor.New[float16.BFloat16]([]int{size, size}, bData)

	b.ResetTimer()

	for range b.N {
		_, _ = eng.MatMul(ctx, a, bt)
	}
}

func BenchmarkBF16MatMul_GPU_128(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	ops := numeric.BFloat16Ops{}
	eng, err := NewGPUEngine[float16.BFloat16](ops)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = eng.Close() }()

	benchBF16MatMul(b, eng, 128)
}

func BenchmarkBF16MatMul_GPU_512(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	ops := numeric.BFloat16Ops{}
	eng, err := NewGPUEngine[float16.BFloat16](ops)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = eng.Close() }()

	benchBF16MatMul(b, eng, 512)
}

func BenchmarkBF16MatMul_GPU_1024(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	ops := numeric.BFloat16Ops{}
	eng, err := NewGPUEngine[float16.BFloat16](ops)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = eng.Close() }()

	benchBF16MatMul(b, eng, 1024)
}

func BenchmarkBF16MatMul_GPU_2048(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	ops := numeric.BFloat16Ops{}
	eng, err := NewGPUEngine[float16.BFloat16](ops)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = eng.Close() }()

	benchBF16MatMul(b, eng, 2048)
}
