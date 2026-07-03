package attention

import (
	"context"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// This file is the ADR-091 fixture for zerfoo#870 (docs/lore.md L-0006).
//
// FusedSDPA's flash kernels used to allocate their scratch/output GPU
// buffers per call: tryFlashDecode's split-KV scratch was freed via a bare
// defer, and tryFlashForward's output had no per-call free at all (relying
// on Go's GC finalizer to reclaim it once unreferenced). Either way, a CUDA
// graph bakes in the literal device address a captured kernel launch reads
// or writes, so freeing that address between replays -- via defer or via a
// finalizer that fires whenever GC next runs -- hands it to an unrelated
// later allocation while the graph keeps replaying kernels against it. Wolf's
// CrossAsset training crashed with "an illegal memory access was
// encountered" around replay #141 of 511 under training.CaptureReplayRunner
// this way.
//
// TestFusedSDPAForwardReplayStableScratch drives the same
// BeginCapture/EndCapture/ReplayGraph state machine CaptureReplayRunner uses
// (compute.GraphCapturer), scoped to a single SDPA node instead of a whole
// training step, and replays 511 times -- matching the issue's exact
// reproduction scale -- to prove the persistent-scratch fix (flash_scratch.go)
// survives it: every replay must succeed, the output buffer's device address
// must never move, and the output must still match the CPU reference after
// the last replay.
//
// GPU-only: without CUDA, engine construction fails and the test skips.
// CUDA-graph capture has no CPU analogue to assert against, so there is no
// CPU-runnable equivalent for the capture/replay assertions themselves; the
// CPU-runnable bailout/fallback wiring is already covered by
// TestTryFlashForwardFallback and TestSDPAFlashFallbackParity in flash_test.go.
func TestFusedSDPAForwardReplayStableScratch(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}

	engine, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Skipf("NewGPUEngine: %v", err)
	}

	var engIface compute.Engine[float32] = engine
	gc, ok := engIface.(compute.GraphCapturer)
	if !ok {
		t.Skip("engine does not implement compute.GraphCapturer")
	}

	const (
		batchHeads = 2
		seqLen     = 32 // >= tryFlashForward's short-sequence bailout threshold (32).
		headDim    = 64
	)
	n := batchHeads * seqLen * headDim

	hostQ := make([]float32, n)
	hostK := make([]float32, n)
	hostV := make([]float32, n)
	for i := range hostQ {
		hostQ[i] = float32(i%7-3) * 0.1
		hostK[i] = float32(i%5-2) * 0.1
		hostV[i] = float32(i%11-5) * 0.1
	}
	want := naiveSoftmaxAttention(hostQ, hostK, hostV, batchHeads, seqLen, headDim)

	shape := []int{batchHeads, seqLen, headDim}
	q := newDeviceTensor(t, hostQ, shape)
	k := newDeviceTensor(t, hostK, shape)
	v := newDeviceTensor(t, hostV, shape)

	sdpa := NewScaledDotProductAttention[float32](engine, headDim)

	ctx := context.Background()

	// Warmup: an eager (uncaptured) call, exactly like CaptureReplayRunner's
	// warmup steps, so the persistent scratch buffer is allocated BEFORE
	// capture begins. ztensor's own capture contract requires this: "All
	// tensors used during capture must be pre-allocated. Allocations during
	// capture (cudaMalloc) will fail with error 901" (compute.GraphCapturer
	// doc, ztensor compute/engine.go).
	warmOut, err := sdpa.Forward(ctx, q, k, v, nil)
	if err != nil {
		t.Fatalf("warmup Forward: %v", err)
	}
	if warmOut == nil {
		t.Fatal("warmup Forward bailed out of the flash path unexpectedly (check seqLen/headDim thresholds)")
	}
	warmStore, ok := warmOut.GetStorage().(*tensor.GPUStorage[float32])
	if !ok {
		t.Fatalf("warmup output storage is %T, want *GPUStorage[float32]", warmOut.GetStorage())
	}
	prePtr := warmStore.Ptr()

	// Capture: record one more forward pass into a CUDA graph. No new
	// allocation should happen here -- the scratch buffer is already sized
	// from the warmup call above.
	if err := gc.BeginCapture(); err != nil {
		t.Fatalf("BeginCapture: %v", err)
	}
	capOut, capErr := sdpa.Forward(ctx, q, k, v, nil)
	handle, endErr := gc.EndCapture()
	if capErr != nil {
		if endErr == nil {
			_ = gc.DestroyGraph(handle)
		}
		t.Fatalf("captured Forward: %v", capErr)
	}
	if endErr != nil {
		t.Fatalf("EndCapture: %v", endErr)
	}
	defer func() { _ = gc.DestroyGraph(handle) }()

	capStore, ok := capOut.GetStorage().(*tensor.GPUStorage[float32])
	if !ok {
		t.Fatalf("captured output storage is %T, want *GPUStorage[float32]", capOut.GetStorage())
	}
	if capStore.Ptr() != prePtr {
		t.Fatalf("output buffer reallocated between warmup and capture: %p != %p", capStore.Ptr(), prePtr)
	}

	// Replay 511 times -- the exact scale zerfoo#870 crashed at (~#141/511).
	// ReplayGraph synchronizes the stream (training/capture_replay.go relies
	// on this too), so a replay-time illegal memory access surfaces here as
	// an error, not a later flaky failure.
	const replays = 511
	for i := 0; i < replays; i++ {
		if err := gc.ReplayGraph(handle); err != nil {
			t.Fatalf("ReplayGraph failed at replay #%d/%d: %v", i+1, replays, err)
		}
	}

	if sdpa.flashFwdOut.storage.Ptr() != prePtr {
		t.Fatalf("persistent scratch buffer moved after %d replays: %p != %p", replays, sdpa.flashFwdOut.storage.Ptr(), prePtr)
	}

	got, err := capStore.TrySlice()
	if err != nil {
		t.Fatalf("TrySlice: %v", err)
	}
	const tol = 1e-4
	mismatches := 0
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d] = %f, want %f (diff %e)", i, got[i], want[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches after %d replays: %d / %d", replays, mismatches, len(got))
	}
}

// TestFlashDecodeScratchReusedAcrossCalls covers the decode (split-KV) side
// of the same fix. Decode's Synchronize call on the launch stream (a
// genuine host-blocking call, needed so eager/inference callers see a
// completed result) is illegal mid-CUDA-graph-capture, so decode-under-
// capture is a separate, pre-existing, documented limitation (see
// flash_decode.go's tryFlashDecode doc comment and the #865->#870->#878
// cluster in docs/lore.md L-0006) that this fix does not newly introduce or
// claim to close -- decode is an inference-time (KV-cache) path that
// training.CaptureReplayRunner's training walk does not exercise. What this
// test DOES verify on real hardware: repeated EAGER calls (the only mode
// decode actually runs in) reuse the SAME persistent output/scratch buffers
// instead of reallocating every call, and correctness holds throughout --
// i.e. the scratch-lifetime half of the zerfoo#870 fix applies to decode
// too, independent of the capture-compatibility question.
func TestFlashDecodeScratchReusedAcrossCalls(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}

	engine, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Skipf("NewGPUEngine: %v", err)
	}

	const (
		batch      = 2
		numQHeads  = 4
		numKVHeads = 4
		headDim    = 64
		kvLen      = 512 // > chunkSize (256): exercises the multi-split combine.
	)
	numBH := batch * numQHeads
	kvDim := numKVHeads * headDim

	hostQ := make([]float32, numBH*headDim)
	hostK := make([]float32, batch*kvLen*kvDim)
	hostV := make([]float32, batch*kvLen*kvDim)
	for i := range hostQ {
		hostQ[i] = float32(i%7-3) * 0.1
	}
	for i := range hostK {
		hostK[i] = float32(i%5-2) * 0.1
		hostV[i] = float32(i%11-5) * 0.1
	}
	want := naiveDecodeRef(hostQ, hostK, hostV, batch, numQHeads, numKVHeads, kvLen, headDim)

	q := newDeviceTensor(t, hostQ, []int{numBH, 1, headDim})
	k := newDeviceTensor(t, hostK, []int{batch, kvLen, kvDim})
	v := newDeviceTensor(t, hostV, []int{batch, kvLen, kvDim})

	sdpa := NewScaledDotProductAttention[float32](engine, headDim, WithHeadCounts[float32](numQHeads, numKVHeads))

	ctx := context.Background()

	var (
		outPtr, partialOPtr, partialLSEPtr unsafe.Pointer
	)

	const calls = 20
	for i := 0; i < calls; i++ {
		out, err := sdpa.Forward(ctx, q, k, v, nil)
		if err != nil {
			t.Fatalf("Forward call #%d: %v", i, err)
		}
		if out == nil {
			t.Fatalf("Forward call #%d bailed out of the flash decode path unexpectedly", i)
		}

		outStore, ok := out.GetStorage().(*tensor.GPUStorage[float32])
		if !ok {
			t.Fatalf("output storage is %T, want *GPUStorage[float32]", out.GetStorage())
		}

		if i == 0 {
			outPtr = outStore.Ptr()
			partialOPtr = sdpa.flashDecPartialO.storage.Ptr()
			partialLSEPtr = sdpa.flashDecPartialLSE.storage.Ptr()
		} else {
			if outStore.Ptr() != outPtr {
				t.Fatalf("call #%d: output buffer moved: %p != %p", i, outStore.Ptr(), outPtr)
			}
			if sdpa.flashDecPartialO.storage.Ptr() != partialOPtr {
				t.Fatalf("call #%d: partialO scratch moved: %p != %p", i, sdpa.flashDecPartialO.storage.Ptr(), partialOPtr)
			}
			if sdpa.flashDecPartialLSE.storage.Ptr() != partialLSEPtr {
				t.Fatalf("call #%d: partialLSE scratch moved: %p != %p", i, sdpa.flashDecPartialLSE.storage.Ptr(), partialLSEPtr)
			}
		}

		got, err := outStore.TrySlice()
		if err != nil {
			t.Fatalf("call #%d: TrySlice: %v", i, err)
		}
		const tol = 1e-3
		mismatches := 0
		for j := range got {
			if diff := math.Abs(float64(got[j] - want[j])); diff > tol {
				if mismatches < 5 {
					t.Errorf("call #%d: output[%d] = %f, want %f (diff %e)", i, j, got[j], want[j], diff)
				}
				mismatches++
			}
		}
		if mismatches > 0 {
			t.Errorf("call #%d: total mismatches: %d / %d", i, mismatches, len(got))
		}
	}
}

func newDeviceTensor(t *testing.T, host []float32, shape []int) *tensor.TensorNumeric[float32] {
	t.Helper()
	store, err := tensor.NewGPUStorageFromSlice(host)
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice: %v", err)
	}
	tt, err := tensor.NewWithStorage[float32](shape, store)
	if err != nil {
		t.Fatalf("NewWithStorage: %v", err)
	}
	return tt
}
