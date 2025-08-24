package compute

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func newEngineF32() *CPUEngine[float32] {
	return NewCPUEngine[float32](numeric.Float32Ops{})
}

func allocF32(shape []int) *tensor.TensorNumeric[float32] {
	t, err := tensor.New[float32](shape, nil)
	if err != nil {
		panic(err)
	}
	return t
}

func fillUniform(e *CPUEngine[float32], t *tensor.TensorNumeric[float32], min, max float32) {
	if err := e.RandomUniform(context.Background(), t, min, max); err != nil {
		panic(err)
	}
}

func BenchmarkCPUEngineMatMul(b *testing.B) {
	cases := []struct{ m, k, n int }{
		{64, 64, 64},
		{128, 128, 128},
	}
	for _, c := range cases {
		name := fmt.Sprintf("%dx%dx%d", c.m, c.k, c.n)
		b.Run(name, func(b *testing.B) {
			ctx := context.Background()
			e := newEngineF32()
			a := allocF32([]int{c.m, c.k})
			bb := allocF32([]int{c.k, c.n})
			fillUniform(e, a, -1, 1)
			fillUniform(e, bb, -1, 1)

			b.ResetTimer()
			var out *tensor.TensorNumeric[float32]
			for i := 0; i < b.N; i++ {
				var err error
				out, err = e.MatMul(ctx, a, bb)
				if err != nil {
					b.Fatalf("MatMul error: %v", err)
				}
			}
			_ = out
		})
	}
}

func BenchmarkCPUEngineAdd(b *testing.B) {
	ctx := context.Background()
	e := newEngineF32()
	a := allocF32([]int{1024, 1024})
	bb := allocF32([]int{1024, 1024})
	fillUniform(e, a, -1, 1)
	fillUniform(e, bb, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.Add(ctx, a, bb); err != nil {
			b.Fatalf("Add error: %v", err)
		}
	}
}

func BenchmarkCPUEngineMul(b *testing.B) {
	ctx := context.Background()
	e := newEngineF32()
	a := allocF32([]int{1024, 1024})
	bb := allocF32([]int{1024, 1024})
	fillUniform(e, a, -1, 1)
	fillUniform(e, bb, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.Mul(ctx, a, bb); err != nil {
			b.Fatalf("Mul error: %v", err)
		}
	}
}

func BenchmarkCPUEngineDiv(b *testing.B) {
	ctx := context.Background()
	e := newEngineF32()
	a := allocF32([]int{1024, 1024})
	bb := allocF32([]int{1024, 1024})
	fillUniform(e, a, -1, 1)
	// Avoid zeros in divisor
	fillUniform(e, bb, 0.5, 1.5)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.Div(ctx, a, bb); err != nil {
			b.Fatalf("Div error: %v", err)
		}
	}
}

func BenchmarkCPUEngineTranspose(b *testing.B) {
	ctx := context.Background()
	e := newEngineF32()
	a := allocF32([]int{1024, 1024})
	fillUniform(e, a, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.Transpose(ctx, a, nil); err != nil { // default 2D transpose
			b.Fatalf("Transpose error: %v", err)
		}
	}
}

func BenchmarkCPUEngineSum(b *testing.B) {
	ctx := context.Background()
	e := newEngineF32()
	a := allocF32([]int{1024, 1024})
	fillUniform(e, a, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.Sum(ctx, a, 1, false); err != nil { // reduce across columns
			b.Fatalf("Sum error: %v", err)
		}
	}
}

func BenchmarkCPUEngineSoftmax(b *testing.B) {
	ctx := context.Background()
	e := newEngineF32()
	a := allocF32([]int{512, 512})
	fillUniform(e, a, -1, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.Softmax(ctx, a, 1); err != nil { // softmax over last axis
			b.Fatalf("Softmax error: %v", err)
		}
	}
}
