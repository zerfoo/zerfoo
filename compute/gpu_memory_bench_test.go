package compute

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// BenchmarkAlloc_Discrete benchmarks standard cudaMalloc.
func BenchmarkAlloc_Discrete_1MB(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	benchDiscreteAlloc(b, 1<<20/4) // 1 MB of float32
}

func BenchmarkAlloc_Discrete_16MB(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	benchDiscreteAlloc(b, 16<<20/4)
}

func BenchmarkAlloc_Discrete_64MB(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	benchDiscreteAlloc(b, 64<<20/4)
}

// BenchmarkAlloc_Managed benchmarks cudaMallocManaged (zero-copy on DGX Spark).
func BenchmarkAlloc_Managed_1MB(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	benchManagedAlloc(b, 1<<20/4)
}

func BenchmarkAlloc_Managed_16MB(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	benchManagedAlloc(b, 16<<20/4)
}

func BenchmarkAlloc_Managed_64MB(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	benchManagedAlloc(b, 64<<20/4)
}

// BenchmarkMatMul_Managed benchmarks MatMul using managed memory tensors.
func BenchmarkMatMul_Managed_512(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	benchManagedMatMul(b, 512)
}

func BenchmarkMatMul_Managed_1024(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	benchManagedMatMul(b, 1024)
}

func benchDiscreteAlloc(b *testing.B, elems int) {
	b.ResetTimer()
	for range b.N {
		gs, err := tensor.NewGPUStorage[float32](elems)
		if err != nil {
			b.Fatal(err)
		}
		_ = gs.Free()
	}
}

func benchManagedAlloc(b *testing.B, elems int) {
	pool := gpuapi.NewCUDAMemPool()

	b.ResetTimer()
	for range b.N {
		gs, err := tensor.NewManagedGPUStorage[float32](pool, elems)
		if err != nil {
			b.Fatal(err)
		}
		_ = gs.Free()
	}
}

func benchManagedMatMul(b *testing.B, size int) {
	ops := numeric.Float32Ops{}
	eng, err := NewGPUEngine[float32](ops)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()
	n := size * size

	aData := make([]float32, n)
	bData := make([]float32, n)
	for i := range aData {
		aData[i] = float32(i%100) * 0.01
		bData[i] = float32(i%100) * 0.01
	}

	a, _ := tensor.New[float32]([]int{size, size}, aData)
	bt, _ := tensor.New[float32]([]int{size, size}, bData)

	b.ResetTimer()
	for range b.N {
		_, _ = eng.MatMul(ctx, a, bt)
	}
}
