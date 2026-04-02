// Package dsp provides digital signal processing primitives.
package dsp

import (
	"math"
	"math/cmplx"
)

// Coefficients computes the Discrete Fourier Transform of data using an
// iterative Cooley-Tukey radix-2 FFT. If len(data) is not a power of 2 the
// input is zero-padded to the next power of 2 before transformation.
//
// The returned slice has length N (the padded length) and contains the full
// complex spectrum.  For a real-valued input the positive-frequency
// coefficients are at indices 0..N/2 (inclusive).
func Coefficients(data []float64) []complex128 {
	n := len(data)
	if n == 0 {
		return nil
	}

	// Determine padded length (next power of 2).
	m := nextPow2(n)

	// Copy data into complex buffer, zero-padding if necessary.
	buf := make([]complex128, m)
	for i := 0; i < n; i++ {
		buf[i] = complex(data[i], 0)
	}

	// Bit-reversal permutation.
	bitReverse(buf)

	// Butterfly stages.
	for s := 2; s <= m; s <<= 1 {
		half := s >> 1
		wm := cmplx.Exp(complex(0, -2*math.Pi/float64(s))) // twiddle base
		for k := 0; k < m; k += s {
			w := complex(1, 0)
			for j := 0; j < half; j++ {
				t := w * buf[k+j+half]
				u := buf[k+j]
				buf[k+j] = u + t
				buf[k+j+half] = u - t
				w *= wm
			}
		}
	}

	return buf
}

// nextPow2 returns the smallest power of 2 >= n.
func nextPow2(n int) int {
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// bitReverse performs an in-place bit-reversal permutation on buf.
// len(buf) must be a power of 2.
func bitReverse(buf []complex128) {
	n := len(buf)
	bits := 0
	for v := n; v > 1; v >>= 1 {
		bits++
	}
	for i := 0; i < n; i++ {
		j := reverseBits(i, bits)
		if i < j {
			buf[i], buf[j] = buf[j], buf[i]
		}
	}
}

// reverseBits reverses the lowest `width` bits of v.
func reverseBits(v, width int) int {
	r := 0
	for i := 0; i < width; i++ {
		r = (r << 1) | (v & 1)
		v >>= 1
	}
	return r
}
