package timeseries

import "math"

// geluScalar computes the GELU approximation for a single value using the
// tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
func geluScalar[T ~float32 | ~float64](x T) T {
	xf := float64(x)
	inner := math.Sqrt(2/math.Pi) * (xf + 0.044715*xf*xf*xf)
	return T(0.5 * xf * (1 + math.Tanh(inner)))
}

// geluDeriv computes the derivative of the GELU tanh approximation.
func geluDeriv[T ~float32 | ~float64](x T) T {
	xf := float64(x)
	c := math.Sqrt(2.0 / math.Pi)
	inner := c * (xf + 0.044715*xf*xf*xf)
	tanh := math.Tanh(inner)
	dInner := c * (1 + 3*0.044715*xf*xf)
	return T(0.5*(1+tanh) + 0.5*xf*(1-tanh*tanh)*dInner)
}

// copyMatrix creates a deep copy of a 2D float64 slice.
func copyMatrix(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = make([]float64, len(x[i]))
		copy(out[i], x[i])
	}
	return out
}

// softmaxF64 computes softmax over a 1D float64 slice with numerical stability.
func softmaxF64(x []float64) []float64 {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
