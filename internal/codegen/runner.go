//go:build !cuda

package codegen

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// MegakernelRunner manages a compiled megakernel .so and its GPU resources.
type MegakernelRunner struct {
	soHandle   uintptr        // dlopen handle
	launchFn   uintptr        // dlsym'd launch_megakernel
	workspace  unsafe.Pointer // GPU workspace buffer
	frozenPtrs unsafe.Pointer // GPU array of float* pointers to frozen data
	frozenBufs []unsafe.Pointer
	layout     WorkspaceLayout
	outputShape []int
}

// LoadMegakernel opens a compiled megakernel .so and resolves the launch symbol.
func LoadMegakernel(soPath string) (*MegakernelRunner, error) {
	handle, err := cuda.DlopenPath(soPath)
	if err != nil {
		return nil, fmt.Errorf("load megakernel: %w", err)
	}
	launchFn, err := cuda.Dlsym(handle, "launch_megakernel")
	if err != nil {
		return nil, fmt.Errorf("resolve launch_megakernel: %w", err)
	}
	return &MegakernelRunner{soHandle: handle, launchFn: launchFn}, nil
}

// PrepareWorkspace allocates GPU memory for the workspace and frozen slots.
// frozenData provides the float32 data for each frozen slot, indexed by
// position in cfg.FrozenSlots (not by slot index).
func (r *MegakernelRunner) PrepareWorkspace(cfg MegakernelConfig, frozenData [][]float32) error {
	r.layout = ComputeWorkspaceLayout(cfg)

	// Allocate workspace buffer.
	wsBytes := r.layout.TotalSize * 4
	if wsBytes > 0 {
		var err error
		r.workspace, err = cuda.Malloc(wsBytes)
		if err != nil {
			return fmt.Errorf("alloc workspace (%d bytes): %w", wsBytes, err)
		}
	}

	nFrozen := len(cfg.FrozenSlots)
	if nFrozen == 0 {
		return nil
	}
	if len(frozenData) != nFrozen {
		return fmt.Errorf("frozenData length %d != FrozenSlots length %d", len(frozenData), nFrozen)
	}

	// Allocate and upload each frozen slot to GPU.
	r.frozenBufs = make([]unsafe.Pointer, nFrozen)
	hostPtrs := make([]uintptr, nFrozen)
	for i, data := range frozenData {
		if len(data) == 0 {
			continue
		}
		buf, err := cuda.Malloc(len(data) * 4)
		if err != nil {
			return fmt.Errorf("alloc frozen slot %d: %w", i, err)
		}
		if err := cuda.Memcpy(buf, unsafe.Pointer(&data[0]), len(data)*4, cuda.MemcpyHostToDevice); err != nil {
			return fmt.Errorf("upload frozen slot %d: %w", i, err)
		}
		r.frozenBufs[i] = buf
		hostPtrs[i] = uintptr(buf)
	}

	// Upload the pointer array to GPU (8 bytes per pointer on 64-bit).
	ptrArrayBytes := nFrozen * 8
	var err error
	r.frozenPtrs, err = cuda.Malloc(ptrArrayBytes)
	if err != nil {
		return fmt.Errorf("alloc frozen ptr array: %w", err)
	}
	if err := cuda.Memcpy(r.frozenPtrs, unsafe.Pointer(&hostPtrs[0]), ptrArrayBytes, cuda.MemcpyHostToDevice); err != nil {
		return fmt.Errorf("upload frozen ptr array: %w", err)
	}

	// Store output shape for callers.
	if cfg.OutputSlot < len(cfg.SlotShapes) {
		r.outputShape = cfg.SlotShapes[cfg.OutputSlot]
	}

	return nil
}

// OutputShape returns the shape of the megakernel output slot.
func (r *MegakernelRunner) OutputShape() []int {
	return r.outputShape
}

// Launch runs the megakernel with input data and returns the output.
func (r *MegakernelRunner) Launch(inputData []float32, pos int) ([]float32, error) {
	// Copy input to workspace at InputOffset.
	if len(inputData) > 0 {
		dstPtr := unsafe.Add(r.workspace, r.layout.InputOffset*4)
		if err := cuda.Memcpy(dstPtr, unsafe.Pointer(&inputData[0]), len(inputData)*4, cuda.MemcpyHostToDevice); err != nil {
			return nil, fmt.Errorf("upload input: %w", err)
		}
	}

	// Launch the megakernel.
	ret := cuda.Ccall(r.launchFn,
		uintptr(r.workspace),
		uintptr(r.frozenPtrs),
		uintptr(pos),
		uintptr(r.layout.TotalSize),
	)
	if ret != 0 {
		return nil, fmt.Errorf("megakernel launch failed: cuda error %d", ret)
	}

	// Copy output from workspace at OutputOffset.
	output := make([]float32, r.layout.OutputSize)
	srcPtr := unsafe.Add(r.workspace, r.layout.OutputOffset*4)
	if err := cuda.Memcpy(unsafe.Pointer(&output[0]), srcPtr, r.layout.OutputSize*4, cuda.MemcpyDeviceToHost); err != nil {
		return nil, fmt.Errorf("download output: %w", err)
	}

	return output, nil
}

// Close releases all GPU resources.
func (r *MegakernelRunner) Close() error {
	for _, buf := range r.frozenBufs {
		if buf != nil {
			_ = cuda.Free(buf)
		}
	}
	if r.frozenPtrs != nil {
		_ = cuda.Free(r.frozenPtrs)
	}
	if r.workspace != nil {
		_ = cuda.Free(r.workspace)
	}
	return nil
}
