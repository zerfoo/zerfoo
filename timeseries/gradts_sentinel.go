package timeseries

import (
	"fmt"
	"reflect"

	"github.com/zerfoo/ztensor/tensor"
)

// verifyGradTsAliasing is the strengthened gradTs sentinel extracted from
// trainGPU for testability (T2.3). It panics if the cached gradTs slice no
// longer aliases the live grads.allParamTensors() backing storage.
//
// v3 fix (fix/v3-storage-identity-sentinel): the previous revision compared
// unsafe.Pointer(&Data()[0]) which is a false positive for GPU-backed
// tensors — GPUStorage.Slice() (backing Data()) allocates a fresh host
// buffer on every call via a D2H copy, so two consecutive .Data() calls on
// the same tensor return different base addresses. The actual aliasing
// invariant we care about is Storage identity: does the cached gradTs
// wrapper point at the same backing Storage object as grads.X? That
// determines whether AdamW reads the same data backward wrote. We compare
// via reflect.ValueOf(...).Pointer() which is safe for pointer-typed
// interface values and works uniformly for CPUStorage / GPUStorage / any
// future storage kind. See .claude/scratch/v3-storage-kind-result.md and
// ztensor/compute/gpu_kernels.go:121 (makeGPUResult.SetStorage flip).
//
// Checks performed, in order:
//   - len(grads.allParamTensors()) == len(gradTs)
//   - per-index wrapper identity (fast pre-filter)
//   - per-index Storage identity via reflect.Pointer
func verifyGradTsAliasing(grads *gpuGrads, gradTs []*tensor.TensorNumeric[float32]) {
	liveGrads := grads.allParamTensors()
	if len(liveGrads) != len(gradTs) {
		panic(fmt.Sprintf("patchtst gpu sentinel: len(grads.allParamTensors())=%d != len(gradTs)=%d", len(liveGrads), len(gradTs)))
	}
	for i := range liveGrads {
		lg := liveGrads[i]
		gt := gradTs[i]
		// Wrapper identity check: catches the case where gradTs[i] was
		// reassigned to a different *TensorNumeric wrapper entirely (e.g.
		// arena realloc leaving the gpuGrads field updated but the cached
		// flat slice stale).
		if lg != gt {
			panic(fmt.Sprintf("patchtst gpu sentinel: wrapper mismatch at index %d: grads.allParamTensors()[i]=%p, gradTs[i]=%p", i, lg, gt))
		}
		// Storage identity check: catches the case where the wrapper is
		// retained but its backing Storage was swapped mid-batch (e.g.
		// makeGPUResult.SetStorage flipping CPUStorage -> GPUStorage, see
		// ztensor/compute/gpu_kernels.go:121). reflect.Pointer is safe for
		// pointer-typed interface values and uniform across storage kinds.
		lgStorage := lg.GetStorage()
		gtStorage := gt.GetStorage()
		lsp := reflect.ValueOf(lgStorage).Pointer()
		gsp := reflect.ValueOf(gtStorage).Pointer()
		if lsp != gsp {
			panic(fmt.Sprintf(
				"patchtst gpu sentinel: storage identity mismatch at index %d\n"+
					"  grads.allParamTensors()[i] wrapper: %p, Storage ptr: 0x%x, kind: %T\n"+
					"  gradTs[i]                   wrapper: %p, Storage ptr: 0x%x, kind: %T\n"+
					"This means the tensor wrapper identity held but the underlying "+
					"Storage was swapped (e.g. makeGPUResult.SetStorage CPU->GPU flip). "+
					"AdamW below would read stale gradients. Fix the upstream aliasing "+
					"before re-enabling training.",
				i, lg, lsp, lgStorage, gt, gsp, gtStorage))
		}
	}
}
