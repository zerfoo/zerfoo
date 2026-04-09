package timeseries

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/tensor"
)

// verifyGradTsAliasing is the strengthened gradTs sentinel extracted from
// trainGPU for testability (T2.3). It panics if the cached gradTs slice no
// longer aliases the live grads.allParamTensors() backing storage. Behavior
// is identical to the former inline block at patchtst_gpu_train.go ~L1089.
//
// Checks performed, in order:
//   - len(grads.allParamTensors()) == len(gradTs)
//   - per-index Data() length equality
//   - per-index unsafe.Pointer equality of &Data()[0] (skipped when both
//     sides have zero length)
func verifyGradTsAliasing(grads *gpuGrads, gradTs []*tensor.TensorNumeric[float32]) {
	liveGrads := grads.allParamTensors()
	if len(liveGrads) != len(gradTs) {
		panic(fmt.Sprintf("patchtst gpu sentinel: len(grads.allParamTensors())=%d != len(gradTs)=%d", len(liveGrads), len(gradTs)))
	}
	for i := range liveGrads {
		lg := liveGrads[i]
		gt := gradTs[i]
		lgData := lg.Data()
		gtData := gt.Data()
		if len(lgData) != len(gtData) {
			panic(fmt.Sprintf("patchtst gpu sentinel: len mismatch at index %d: grads.allParamTensors()[i].Data() len=%d, gradTs[i].Data() len=%d (tensor wrappers: live=%p cached=%p)",
				i, len(lgData), len(gtData), lg, gt))
		}
		if len(lgData) == 0 {
			// Both zero-length: no backing element to compare. Skip.
			continue
		}
		lgPtr := unsafe.Pointer(&lgData[0])
		gtPtr := unsafe.Pointer(&gtData[0])
		if lgPtr != gtPtr {
			sampleLive := lgData[:4:4]
			sampleCached := gtData[:4:4]
			panic(fmt.Sprintf(
				"patchtst gpu sentinel: gradient backing-slice mismatch at index %d\n"+
					"  grads.allParamTensors()[i] wrapper: %p, Data()[0] ptr: %#x, Data()[:4]: %v\n"+
					"  gradTs[i]                   wrapper: %p, Data()[0] ptr: %#x, Data()[:4]: %v\n"+
					"This means the sentinel wrapper-identity check passed but the underlying "+
					"backing array diverged (arena realloc / SetData / etc). AdamW below would "+
					"read stale gradients. Fix the upstream aliasing before re-enabling training.",
				i, lg, uintptr(lgPtr), sampleLive, gt, uintptr(gtPtr), sampleCached))
		}
	}
}
