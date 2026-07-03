package kernels

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// TestFP16SignaturesCompile verifies that the FP16 purego wrappers have
// the expected function signatures (compile-time check).
func TestFP16SignaturesCompile(t *testing.T) {
	type binaryFn = func(a, b, c unsafe.Pointer, n int, s unsafe.Pointer) error

	tests := []struct {
		name string
		fn   any
	}{
		{"AddFP16", assignFunc[binaryFn](AddFP16)},
		{"SubFP16", assignFunc[binaryFn](SubFP16)},
		{"MulFP16", assignFunc[binaryFn](MulFP16)},
		{"DivFP16", assignFunc[binaryFn](DivFP16)},
		{"RMSNormFP16", assignFunc[func(input, weight, output unsafe.Pointer, eps float32, rows, D int, s unsafe.Pointer) error](RMSNormFP16)},
		{"ScaledSoftmaxFP16", assignFunc[func(input, output unsafe.Pointer, outer, inner, axisSize int, scale float32, stream unsafe.Pointer) error](ScaledSoftmaxFP16)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.fn == nil {
				t.Errorf("%s function is nil", tt.name)
			}
		})
	}
}

// TestFP16GracefulWithoutCUDA verifies that all FP16 kernel functions
// return an error (not panic) when CUDA is not available.
func TestFP16GracefulWithoutCUDA(t *testing.T) {
	// When CUDA IS available, klib() is non-nil and these wrappers launch the
	// real FP16 kernels with the nil device pointers below -- a null-pointer
	// kernel launch that triggers an illegal memory access and poisons the
	// CUDA context for every subsequent test in the package (zerfoo#922).
	// Every other *GracefulWithoutCUDA test guards on this for the same reason;
	// this one must too. The graceful path is only meaningful when CUDA is
	// absent.
	if cuda.Available() {
		t.Skip("CUDA available, skipping graceful-failure test")
	}

	tests := []struct {
		name string
		fn   func() error
	}{
		{"AddFP16", func() error { return AddFP16(nil, nil, nil, 1, nil) }},
		{"SubFP16", func() error { return SubFP16(nil, nil, nil, 1, nil) }},
		{"MulFP16", func() error { return MulFP16(nil, nil, nil, 1, nil) }},
		{"DivFP16", func() error { return DivFP16(nil, nil, nil, 1, nil) }},
		{"RMSNormFP16", func() error { return RMSNormFP16(nil, nil, nil, 1e-5, 1, 1, nil) }},
		{"ScaledSoftmaxFP16", func() error { return ScaledSoftmaxFP16(nil, nil, 1, 1, 1, 1.0, nil) }},
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
