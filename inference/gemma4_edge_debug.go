package inference

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/zerfoo/ztensor/tensor"
)

// gemma4EdgeDebug is enabled by ZERFOO_GQA_DEBUG=1 (same gate as the GQA
// instrumentation in layers/attention) and emits storage-state lines at
// strategic points in the gemma4-edge prefill pipeline. Used to localize the
// CUDA illegal memory access investigated in E98.T98.2.1.
var gemma4EdgeDebug = os.Getenv("ZERFOO_GQA_DEBUG") == "1"

func gemma4EdgeDebugTensor[T tensor.Numeric](tag string, t *tensor.TensorNumeric[T]) {
	if !gemma4EdgeDebug || t == nil {
		return
	}
	storageType := fmt.Sprintf("%T", t.GetStorage())
	devPtr := unsafe.Pointer(nil)
	devLen := 0
	probe := "skip"
	if gs, ok := t.GetStorage().(*tensor.GPUStorage[T]); ok {
		devPtr = gs.Ptr()
		devLen = gs.Len()
		// Force-sync probe: TrySlice attempts a D2H cudaMemcpy which both
		// flushes pending kernels and surfaces any sticky CUDA error. This
		// localizes async kernel faults to the stage that produced them.
		if _, err := gs.TrySlice(); err != nil {
			probe = "FAIL: " + err.Error()
		} else {
			probe = "ok"
		}
	}
	fmt.Fprintf(os.Stderr, "[GE_DBG] %s shape=%v storage=%s gpuPtr=%p gpuLen=%d sync=%s\n",
		tag, t.Shape(), storageType, devPtr, devLen, probe)
}
