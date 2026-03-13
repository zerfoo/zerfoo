package kernels

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// assignFunc forces a compile-time type check on a function value.
func assignFunc[T any](fn T) T { return fn }

// TestElementwiseParitySignatures verifies that the purego kernel wrappers
// have the same function signatures as the CGo versions.
func TestElementwiseParitySignatures(t *testing.T) {
	type binaryFn = func(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error
	type scalarFn = func(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int, s unsafe.Pointer) error
	type unaryFn = func(a, c unsafe.Pointer, n int, s unsafe.Pointer) error
	type broadcastFn = func(a, b, c unsafe.Pointer, saRow, saCol, sbRow, sbCol, M, D int, s unsafe.Pointer) error
	type broadcast4DFn = func(a, b, c unsafe.Pointer, d0, d1, d2, d3, sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3 int, s unsafe.Pointer) error

	tests := []struct {
		name string
		fn   any
	}{
		// binary ops
		{"Add", assignFunc[binaryFn](Add)},
		{"Sub", assignFunc[binaryFn](Sub)},
		{"Mul", assignFunc[binaryFn](Mul)},
		{"Div", assignFunc[binaryFn](Div)},
		{"Pow", assignFunc[binaryFn](Pow)},
		// scalar ops
		{"AddScalar", assignFunc[scalarFn](AddScalar)},
		{"SubScalar", assignFunc[scalarFn](SubScalar)},
		{"MulScalar", assignFunc[scalarFn](MulScalar)},
		{"DivScalar", assignFunc[scalarFn](DivScalar)},
		{"PowScalar", assignFunc[scalarFn](PowScalar)},
		// unary ops
		{"Exp", assignFunc[unaryFn](Exp)},
		{"Log", assignFunc[unaryFn](Log)},
		{"Sqrt", assignFunc[unaryFn](Sqrt)},
		{"Rsqrt", assignFunc[unaryFn](Rsqrt)},
		{"Tanh", assignFunc[unaryFn](Tanh)},
		// TanhPrime has a different signature (3 pointers)
		{"TanhPrime", assignFunc[func(a, upstream, c unsafe.Pointer, n int, s unsafe.Pointer) error](TanhPrime)},
		// special ops
		{"Fill", assignFunc[func(data unsafe.Pointer, value float32, n int, s unsafe.Pointer) error](Fill)},
		{"SumAxis", assignFunc[func(input, output unsafe.Pointer, outer, inner, axisSize int, s unsafe.Pointer) error](SumAxis)},
		{"Softmax", assignFunc[func(input, output unsafe.Pointer, outer, inner, axisSize int, s unsafe.Pointer) error](Softmax)},
		// broadcast ops
		{"AddBroadcast", assignFunc[broadcastFn](AddBroadcast)},
		{"SubBroadcast", assignFunc[broadcastFn](SubBroadcast)},
		{"MulBroadcast", assignFunc[broadcastFn](MulBroadcast)},
		{"DivBroadcast", assignFunc[broadcastFn](DivBroadcast)},
		// broadcast 4D ops
		{"AddBroadcast4D", assignFunc[broadcast4DFn](AddBroadcast4D)},
		{"SubBroadcast4D", assignFunc[broadcast4DFn](SubBroadcast4D)},
		{"MulBroadcast4D", assignFunc[broadcast4DFn](MulBroadcast4D)},
		{"DivBroadcast4D", assignFunc[broadcast4DFn](DivBroadcast4D)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.fn == nil {
				t.Errorf("%s function is nil", tt.name)
			}
		})
	}
}

// TestElementwiseParityKernelLibSymbols verifies that openKernelLib resolves
// all elementwise symbols when CUDA is available.
func TestElementwiseParityKernelLibSymbols(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available, skipping symbol resolution test")
	}

	k, err := openKernelLib()
	if err != nil {
		t.Fatalf("openKernelLib: %v", err)
	}
	if k == nil {
		t.Fatal("openKernelLib returned nil without error")
	}

	symbols := []struct {
		name string
		ptr  uintptr
	}{
		// binary
		{"launchAdd", k.launchAdd},
		{"launchSub", k.launchSub},
		{"launchMul", k.launchMul},
		{"launchDiv", k.launchDiv},
		{"launchPow", k.launchPow},
		// scalar
		{"launchAddScalar", k.launchAddScalar},
		{"launchMulScalar", k.launchMulScalar},
		{"launchDivScalar", k.launchDivScalar},
		{"launchSubScalar", k.launchSubScalar},
		{"launchPowScalar", k.launchPowScalar},
		// unary
		{"launchExp", k.launchExp},
		{"launchLog", k.launchLog},
		{"launchSqrt", k.launchSqrt},
		{"launchRsqrt", k.launchRsqrt},
		{"launchTanh", k.launchTanh},
		{"launchTanhPrime", k.launchTanhPrime},
		// special
		{"launchFill", k.launchFill},
		{"launchSumAxis", k.launchSumAxis},
		{"launchSoftmax", k.launchSoftmax},
		// broadcast
		{"launchAddBroadcast", k.launchAddBroadcast},
		{"launchSubBroadcast", k.launchSubBroadcast},
		{"launchMulBroadcast", k.launchMulBroadcast},
		{"launchDivBroadcast", k.launchDivBroadcast},
		// broadcast 4D
		{"launchAddBroadcast4D", k.launchAddBroadcast4D},
		{"launchSubBroadcast4D", k.launchSubBroadcast4D},
		{"launchMulBroadcast4D", k.launchMulBroadcast4D},
		{"launchDivBroadcast4D", k.launchDivBroadcast4D},
	}

	for _, s := range symbols {
		t.Run(s.name, func(t *testing.T) {
			if s.ptr == 0 {
				t.Errorf("symbol %s not resolved (pointer is 0)", s.name)
			}
		})
	}
}

// TestElementwiseParityGracefulWithoutCUDA verifies that all kernel functions
// return an error (not panic) when CUDA is not available.
func TestElementwiseParityGracefulWithoutCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("CUDA available, skipping graceful-failure tests")
	}

	tests := []struct {
		name string
		fn   func() error
	}{
		// binary
		{"Add", func() error { return Add(nil, nil, nil, 1, nil) }},
		{"Sub", func() error { return Sub(nil, nil, nil, 1, nil) }},
		{"Mul", func() error { return Mul(nil, nil, nil, 1, nil) }},
		{"Div", func() error { return Div(nil, nil, nil, 1, nil) }},
		{"Pow", func() error { return Pow(nil, nil, nil, 1, nil) }},
		// scalar
		{"AddScalar", func() error { return AddScalar(nil, 1.0, nil, 1, nil) }},
		{"SubScalar", func() error { return SubScalar(nil, 1.0, nil, 1, nil) }},
		{"MulScalar", func() error { return MulScalar(nil, 1.0, nil, 1, nil) }},
		{"DivScalar", func() error { return DivScalar(nil, 1.0, nil, 1, nil) }},
		{"PowScalar", func() error { return PowScalar(nil, 1.0, nil, 1, nil) }},
		// unary
		{"Exp", func() error { return Exp(nil, nil, 1, nil) }},
		{"Log", func() error { return Log(nil, nil, 1, nil) }},
		{"Sqrt", func() error { return Sqrt(nil, nil, 1, nil) }},
		{"Rsqrt", func() error { return Rsqrt(nil, nil, 1, nil) }},
		{"Tanh", func() error { return Tanh(nil, nil, 1, nil) }},
		{"TanhPrime", func() error { return TanhPrime(nil, nil, nil, 1, nil) }},
		// special
		{"Fill", func() error { return Fill(nil, 1.0, 1, nil) }},
		{"SumAxis", func() error { return SumAxis(nil, nil, 1, 1, 1, nil) }},
		{"Softmax", func() error { return Softmax(nil, nil, 1, 1, 1, nil) }},
		// broadcast
		{"AddBroadcast", func() error { return AddBroadcast(nil, nil, nil, 1, 1, 1, 1, 1, 1, nil) }},
		{"SubBroadcast", func() error { return SubBroadcast(nil, nil, nil, 1, 1, 1, 1, 1, 1, nil) }},
		{"MulBroadcast", func() error { return MulBroadcast(nil, nil, nil, 1, 1, 1, 1, 1, 1, nil) }},
		{"DivBroadcast", func() error { return DivBroadcast(nil, nil, nil, 1, 1, 1, 1, 1, 1, nil) }},
		// broadcast 4D
		{"AddBroadcast4D", func() error { return AddBroadcast4D(nil, nil, nil, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, nil) }},
		{"SubBroadcast4D", func() error { return SubBroadcast4D(nil, nil, nil, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, nil) }},
		{"MulBroadcast4D", func() error { return MulBroadcast4D(nil, nil, nil, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, nil) }},
		{"DivBroadcast4D", func() error { return DivBroadcast4D(nil, nil, nil, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, nil) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.fn()
			if err == nil {
				t.Errorf("%s should return error without CUDA", tt.name)
			}
		})
	}
}
