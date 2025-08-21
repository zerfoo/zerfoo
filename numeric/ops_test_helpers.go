package numeric

import (
	"math"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
)

// ArithmeticTestCase represents a test case for arithmetic operations.
type ArithmeticTestCase[T any] struct {
	name           string
	a, b, expected T
}

// UnaryTestCase represents a test case for unary operations.
type UnaryTestCase[T any] struct {
	name     string
	x        T
	expected T
}

// SumTestCase represents a test case for sum operations.
type SumTestCase[T any] struct {
	name     string
	s        []T
	expected float32
	epsilon  float32
}

// LeakyReLUTestCase represents a test case for LeakyReLU operations.
type LeakyReLUTestCase[T any] struct {
	name     string
	x        T
	alpha    float64
	expected float32
	epsilon  float32
}

// TestArithmeticOp tests a binary arithmetic operation.
func TestArithmeticOp[T any](t *testing.T, opName string, op func(T, T) T, equal func(T, T) bool, tests []ArithmeticTestCase[T]) {
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := op(tt.a, tt.b)
			if !equal(result, tt.expected) {
				t.Errorf("%s(%v, %v): expected %v, got %v", opName, tt.a, tt.b, tt.expected, result)
			}
		})
	}
}

// TestUnaryOp tests a unary operation.
func TestUnaryOp[T any](t *testing.T, opName string, op func(T) T, equal func(T, T) bool, tests []UnaryTestCase[T]) {
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := op(tt.x)
			if !equal(result, tt.expected) {
				t.Errorf("%s(%v): expected %v, got %v", opName, tt.x, tt.expected, result)
			}
		})
	}
}

// TestSumOp tests sum operations with epsilon tolerance.
func TestSumOp[T any](t *testing.T, op func([]T) T, toFloat32 func(T) float32, tests []SumTestCase[T]) {
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := op(tt.s)
			resultFloat := toFloat32(result)
			if math.Abs(float64(resultFloat-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("Sum(%v): expected %v, got %v", tt.s, tt.expected, resultFloat)
			}
		})
	}
}

// TestLeakyReLUOp tests LeakyReLU operations with epsilon tolerance.
func TestLeakyReLUOp[T any](t *testing.T, opName string, op func(T, float64) T, toFloat32 func(T) float32, tests []LeakyReLUTestCase[T]) {
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := op(tt.x, tt.alpha)
			resultFloat := toFloat32(result)
			if math.Abs(float64(resultFloat-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("%s(%v, %v): expected %v, got %v", opName, tt.x, tt.alpha, tt.expected, resultFloat)
			}
		})
	}
}

// Float8TestData provides common test data for float8 operations.
func Float8TestData() struct {
	Add           []ArithmeticTestCase[float8.Float8]
	Mul           []ArithmeticTestCase[float8.Float8]
	Div           []ArithmeticTestCase[float8.Float8]
	Tanh          []UnaryTestCase[float8.Float8]
	Sigmoid       []UnaryTestCase[float8.Float8]
	LeakyReLU     []LeakyReLUTestCase[float8.Float8]
	LeakyReLUGrad []LeakyReLUTestCase[float8.Float8]
	Sum           []SumTestCase[float8.Float8]
} {
	return struct {
		Add           []ArithmeticTestCase[float8.Float8]
		Mul           []ArithmeticTestCase[float8.Float8]
		Div           []ArithmeticTestCase[float8.Float8]
		Tanh          []UnaryTestCase[float8.Float8]
		Sigmoid       []UnaryTestCase[float8.Float8]
		LeakyReLU     []LeakyReLUTestCase[float8.Float8]
		LeakyReLUGrad []LeakyReLUTestCase[float8.Float8]
		Sum           []SumTestCase[float8.Float8]
	}{
		Add: []ArithmeticTestCase[float8.Float8]{
			{"positive numbers", float8.ToFloat8(1.0), float8.ToFloat8(2.0), float8.ToFloat8(3.0)},
			{"negative numbers", float8.ToFloat8(-1.0), float8.ToFloat8(-2.0), float8.ToFloat8(-3.0)},
			{"mixed numbers", float8.ToFloat8(1.0), float8.ToFloat8(-2.0), float8.ToFloat8(-1.0)},
			{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.0), float8.ToFloat8(0.0)},
		},
		Mul: []ArithmeticTestCase[float8.Float8]{
			{"positive numbers", float8.ToFloat8(2.0), float8.ToFloat8(3.0), float8.ToFloat8(6.0)},
			{"negative numbers", float8.ToFloat8(-2.0), float8.ToFloat8(-3.0), float8.ToFloat8(6.0)},
			{"mixed numbers", float8.ToFloat8(2.0), float8.ToFloat8(-3.0), float8.ToFloat8(-6.0)},
			{"zero", float8.ToFloat8(0.0), float8.ToFloat8(5.0), float8.ToFloat8(0.0)},
		},
		Div: []ArithmeticTestCase[float8.Float8]{
			{"positive numbers", float8.ToFloat8(6.0), float8.ToFloat8(3.0), float8.ToFloat8(2.0)},
			{"negative numbers", float8.ToFloat8(-6.0), float8.ToFloat8(-3.0), float8.ToFloat8(2.0)},
			{"mixed numbers", float8.ToFloat8(6.0), float8.ToFloat8(-3.0), float8.ToFloat8(-2.0)},
			{"divide by one", float8.ToFloat8(5.0), float8.ToFloat8(1.0), float8.ToFloat8(5.0)},
			{"zero dividend", float8.ToFloat8(0.0), float8.ToFloat8(5.0), float8.ToFloat8(0.0)},
		},
		Tanh: []UnaryTestCase[float8.Float8]{
			{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.0)},
			{"positive", float8.ToFloat8(1.0), float8.ToFloat8(float32(math.Tanh(1.0)))},
			{"negative", float8.ToFloat8(-1.0), float8.ToFloat8(float32(math.Tanh(-1.0)))},
			{"large positive", float8.ToFloat8(100.0), float8.ToFloat8(float32(math.Tanh(100.0)))},
			{"large negative", float8.ToFloat8(-100.0), float8.ToFloat8(float32(math.Tanh(-100.0)))},
		},
		Sigmoid: []UnaryTestCase[float8.Float8]{
			{"zero", float8.ToFloat8(0.0), float8.ToFloat8(0.5)},
			{"positive", float8.ToFloat8(1.0), float8.ToFloat8(float32(1.0 / (1.0 + math.Exp(-1.0))))},
			{"negative", float8.ToFloat8(-1.0), float8.ToFloat8(float32(1.0 / (1.0 + math.Exp(1.0))))},
			{"large positive", float8.ToFloat8(100.0), float8.ToFloat8(float32(1.0 / (1.0 + math.Exp(-100.0))))},
			{"large negative", float8.ToFloat8(-100.0), float8.ToFloat8(float32(1.0 / (1.0 + math.Exp(100.0))))},
		},
		LeakyReLU: []LeakyReLUTestCase[float8.Float8]{
			{"positive", float8.ToFloat8(2.0), 0.1, 2.0, 0.1},
			{"negative", float8.ToFloat8(-2.0), 0.1, -0.2, 0.1},
			{"zero", float8.ToFloat8(0.0), 0.1, 0.0, 0.1},
			{"negative with different alpha", float8.ToFloat8(-1.0), 0.2, -0.2, 0.1},
		},
		LeakyReLUGrad: []LeakyReLUTestCase[float8.Float8]{
			{"positive", float8.ToFloat8(2.0), 0.1, 1.0, 0.1},
			{"negative", float8.ToFloat8(-2.0), 0.1, 0.1, 0.1},
			{"zero", float8.ToFloat8(0.0), 0.1, 0.1, 0.1},
			{"negative with different alpha", float8.ToFloat8(-1.0), 0.2, 0.2, 0.1},
		},
		Sum: []SumTestCase[float8.Float8]{
			{"empty slice", []float8.Float8{}, 0.0, 0.1},
			{"single element", []float8.Float8{float8.ToFloat8(2.5)}, 2.5, 0.1},
			{"multiple positive", []float8.Float8{float8.ToFloat8(1.0), float8.ToFloat8(2.0), float8.ToFloat8(3.0)}, 6.0, 0.1},
			{"mixed signs", []float8.Float8{float8.ToFloat8(1.0), float8.ToFloat8(-2.0), float8.ToFloat8(3.0)}, 2.0, 0.1},
			{"all zeros", []float8.Float8{float8.ToFloat8(0.0), float8.ToFloat8(0.0), float8.ToFloat8(0.0)}, 0.0, 0.1},
		},
	}
}

// Float16TestData provides common test data for float16 operations.
func Float16TestData() struct {
	Add           []ArithmeticTestCase[float16.Float16]
	Mul           []ArithmeticTestCase[float16.Float16]
	Div           []ArithmeticTestCase[float16.Float16]
	Tanh          []UnaryTestCase[float16.Float16]
	Sigmoid       []UnaryTestCase[float16.Float16]
	LeakyReLU     []LeakyReLUTestCase[float16.Float16]
	LeakyReLUGrad []LeakyReLUTestCase[float16.Float16]
	Sum           []SumTestCase[float16.Float16]
} {
	return struct {
		Add           []ArithmeticTestCase[float16.Float16]
		Mul           []ArithmeticTestCase[float16.Float16]
		Div           []ArithmeticTestCase[float16.Float16]
		Tanh          []UnaryTestCase[float16.Float16]
		Sigmoid       []UnaryTestCase[float16.Float16]
		LeakyReLU     []LeakyReLUTestCase[float16.Float16]
		LeakyReLUGrad []LeakyReLUTestCase[float16.Float16]
		Sum           []SumTestCase[float16.Float16]
	}{
		Add: []ArithmeticTestCase[float16.Float16]{
			{"positive numbers", float16.FromFloat32(1.0), float16.FromFloat32(2.0), float16.FromFloat32(3.0)},
			{"negative numbers", float16.FromFloat32(-1.0), float16.FromFloat32(-2.0), float16.FromFloat32(-3.0)},
			{"mixed numbers", float16.FromFloat32(1.0), float16.FromFloat32(-2.0), float16.FromFloat32(-1.0)},
			{"zero", float16.FromFloat32(0.0), float16.FromFloat32(0.0), float16.FromFloat32(0.0)},
		},
		Mul: []ArithmeticTestCase[float16.Float16]{
			{"positive numbers", float16.FromFloat32(2.0), float16.FromFloat32(3.0), float16.FromFloat32(6.0)},
			{"negative numbers", float16.FromFloat32(-2.0), float16.FromFloat32(-3.0), float16.FromFloat32(6.0)},
			{"mixed numbers", float16.FromFloat32(2.0), float16.FromFloat32(-3.0), float16.FromFloat32(-6.0)},
			{"zero", float16.FromFloat32(0.0), float16.FromFloat32(5.0), float16.FromFloat32(0.0)},
		},
		Div: []ArithmeticTestCase[float16.Float16]{
			{"positive numbers", float16.FromFloat32(6.0), float16.FromFloat32(3.0), float16.FromFloat32(2.0)},
			{"negative numbers", float16.FromFloat32(-6.0), float16.FromFloat32(-3.0), float16.FromFloat32(2.0)},
			{"mixed numbers", float16.FromFloat32(6.0), float16.FromFloat32(-3.0), float16.FromFloat32(-2.0)},
			{"divide by one", float16.FromFloat32(5.0), float16.FromFloat32(1.0), float16.FromFloat32(5.0)},
			{"zero dividend", float16.FromFloat32(0.0), float16.FromFloat32(5.0), float16.FromFloat32(0.0)},
		},
		Tanh: []UnaryTestCase[float16.Float16]{
			{"zero", float16.FromFloat32(0.0), float16.FromFloat32(0.0)},
			{"positive", float16.FromFloat32(1.0), float16.FromFloat32(float32(math.Tanh(1.0)))},
			{"negative", float16.FromFloat32(-1.0), float16.FromFloat32(float32(math.Tanh(-1.0)))},
			{"large positive", float16.FromFloat32(100.0), float16.FromFloat32(float32(math.Tanh(100.0)))},
			{"large negative", float16.FromFloat32(-100.0), float16.FromFloat32(float32(math.Tanh(-100.0)))},
		},
		Sigmoid: []UnaryTestCase[float16.Float16]{
			{"zero", float16.FromFloat32(0.0), float16.FromFloat32(0.5)},
			{"positive", float16.FromFloat32(1.0), float16.FromFloat32(float32(1.0 / (1.0 + math.Exp(-1.0))))},
			{"negative", float16.FromFloat32(-1.0), float16.FromFloat32(float32(1.0 / (1.0 + math.Exp(1.0))))},
			{"large positive", float16.FromFloat32(100.0), float16.FromFloat32(1.0)},
			{"large negative", float16.FromFloat32(-100.0), float16.FromFloat32(0.0)},
		},
		LeakyReLU: []LeakyReLUTestCase[float16.Float16]{
			{"positive", float16.FromFloat32(2.0), 0.1, 2.0, 0.01},
			{"negative", float16.FromFloat32(-2.0), 0.1, -0.2, 0.01},
			{"zero", float16.FromFloat32(0), 0.1, 0.0, 0.01},
			{"negative with different alpha", float16.FromFloat32(-1.0), 0.2, -0.2, 0.01},
		},
		LeakyReLUGrad: []LeakyReLUTestCase[float16.Float16]{
			{"positive", float16.FromFloat32(2.0), 0.1, 1.0, 0.01},
			{"negative", float16.FromFloat32(-2.0), 0.1, 0.1, 0.01},
			{"zero", float16.FromFloat32(0), 0.1, 0.1, 0.01},
			{"negative with different alpha", float16.FromFloat32(-1.0), 0.2, 0.2, 0.01},
		},
		Sum: []SumTestCase[float16.Float16]{
			{"empty slice", []float16.Float16{}, 0.0, 0.01},
			{"single element", []float16.Float16{float16.FromFloat32(2.5)}, 2.5, 0.01},
			{"multiple positive", []float16.Float16{float16.FromFloat32(1.0), float16.FromFloat32(2.0), float16.FromFloat32(3.0)}, 6.0, 0.01},
			{"mixed signs", []float16.Float16{float16.FromFloat32(1.0), float16.FromFloat32(-2.0), float16.FromFloat32(3.0)}, 2.0, 0.01},
			{"all zeros", []float16.Float16{float16.FromFloat32(0), float16.FromFloat32(0), float16.FromFloat32(0)}, 0.0, 0.01},
		},
	}
}
