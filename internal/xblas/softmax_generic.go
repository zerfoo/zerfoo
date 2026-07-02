//go:build !arm64

package xblas

import (
	"math"
	"unsafe"
)

func SoftmaxF32(data *float32, n int) {
	s := unsafe.Slice(data, n)
	mx := s[0]
	for i := 1; i < n; i++ {
		if s[i] > mx {
			mx = s[i]
		}
	}
	var sum float32
	for i := range s {
		s[i] = float32(math.Exp(float64(s[i] - mx)))
		sum += s[i]
	}
	inv := 1.0 / sum
	for i := range s {
		s[i] *= inv
	}
}
