package kernels

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// TestFP8OpsParitySignatures verifies that the FP8 purego kernel wrappers
// have the expected function signatures (compile-time check).
func TestFP8OpsParitySignatures(t *testing.T) {
	tests := []struct {
		name string
		fn   any
	}{
		{"DequantFP8E4M3ToFP16", assignFunc[func(input, output unsafe.Pointer, scale float32, n int, s unsafe.Pointer) error](DequantFP8E4M3ToFP16)},
		{"FP8Add", assignFunc[func(a, b, c unsafe.Pointer, scaleA, scaleB float32, n int, s unsafe.Pointer) error](FP8Add)},
		{"FP8Mul", assignFunc[func(a, b, c unsafe.Pointer, scaleA, scaleB float32, n int, s unsafe.Pointer) error](FP8Mul)},
		{"FP8RMSNorm", assignFunc[func(input, weight, output unsafe.Pointer, scale, eps float32, rows, D int, s unsafe.Pointer) error](FP8RMSNorm)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.fn == nil {
				t.Errorf("%s function is nil", tt.name)
			}
		})
	}
}

// TestFP8OpsKernelLibSymbols verifies that openKernelLib resolves all FP8
// symbols when CUDA is available.
func TestFP8OpsKernelLibSymbols(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
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
		{"launchDequantFP8E4M3ToFP16", k.launchDequantFP8E4M3ToFP16},
		{"launchFP8Add", k.launchFP8Add},
		{"launchFP8Mul", k.launchFP8Mul},
		{"launchFP8RMSNorm", k.launchFP8RMSNorm},
	}

	for _, s := range symbols {
		t.Run(s.name, func(t *testing.T) {
			if s.ptr == 0 {
				t.Errorf("symbol %s not resolved (pointer is 0)", s.name)
			}
		})
	}
}

// TestFP8OpsGracefulWithoutCUDA verifies that all FP8 kernel functions
// return an error (not panic) when CUDA is not available.
func TestFP8OpsGracefulWithoutCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("CUDA available, skipping graceful-failure tests")
	}

	tests := []struct {
		name string
		fn   func() error
	}{
		{"DequantFP8E4M3ToFP16", func() error { return DequantFP8E4M3ToFP16(nil, nil, 1.0, 1, nil) }},
		{"FP8Add", func() error { return FP8Add(nil, nil, nil, 1.0, 1.0, 1, nil) }},
		{"FP8Mul", func() error { return FP8Mul(nil, nil, nil, 1.0, 1.0, 1, nil) }},
		{"FP8RMSNorm", func() error { return FP8RMSNorm(nil, nil, nil, 1.0, 1e-5, 1, 1, nil) }},
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
