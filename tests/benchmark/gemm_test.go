package benchmark

import (
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/internal/xblas"
)

func BenchmarkGEMM(b *testing.B) {
	sizes := []int{128, 256, 512, 1024, 2048}

	for _, n := range sizes {
		a := make([]float32, n*n)
		bm := make([]float32, n*n)
		c := make([]float32, n*n)
		for i := range a {
			a[i] = float32(i%7-3) * 0.01
			bm[i] = float32(i%5-2) * 0.01
		}

		b.Run(fmt.Sprintf("%dx%d", n, n), func(b *testing.B) {
			for b.Loop() {
				// Zero output.
				for i := range c {
					c[i] = 0
				}
				xblas.GemmF32(n, n, n, a, bm, c)
			}
			elapsed := b.Elapsed()
			flops := 2.0 * float64(n) * float64(n) * float64(n) * float64(b.N)
			gflops := flops / elapsed.Seconds() / 1e9
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}
