//go:build cuda

package codegen

import (
	"fmt"
	"unsafe"
)

// MegakernelRunner manages a compiled megakernel .so and its GPU resources.
// This is a stub for the CGo build path where purego-based dlopen is not used.
type MegakernelRunner struct {
	soHandle    uintptr
	launchFn    uintptr
	workspace   unsafe.Pointer
	frozenPtrs  unsafe.Pointer
	frozenBufs  []unsafe.Pointer
	layout      WorkspaceLayout
	outputShape []int
}

var errStub = fmt.Errorf("megakernel runner not supported in cgo build")

// LoadMegakernel is a stub that returns an error in the CGo build.
func LoadMegakernel(_ string) (*MegakernelRunner, error) {
	return nil, errStub
}

// PrepareWorkspace is a stub that returns an error in the CGo build.
func (r *MegakernelRunner) PrepareWorkspace(_ MegakernelConfig, _ [][]float32) error {
	return errStub
}

// OutputShape returns nil in the CGo build stub.
func (r *MegakernelRunner) OutputShape() []int {
	return nil
}

// Launch is a stub that returns an error in the CGo build.
func (r *MegakernelRunner) Launch(_ []float32, _ int) ([]float32, error) {
	return nil, errStub
}

// Close is a stub that returns nil in the CGo build.
func (r *MegakernelRunner) Close() error {
	return nil
}
