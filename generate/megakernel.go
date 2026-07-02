package generate

import (
	"context"
	"log/slog"
	"os"
	"sync/atomic"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/internal/codegen"
	"github.com/zerfoo/ztensor/tensor"
)

// tryCompileMegakernel attempts to compile and load a megakernel for the
// given ExecutionPlan. On success, it sets the megakernel function on the
// plan so that subsequent Run() calls use the fused kernel. On failure, it
// silently falls back to the per-instruction execution path.
//
// This function is safe to call from a goroutine. The megakernel function
// is set atomically via SetMegakernelFn.
func tryCompileMegakernel[T tensor.Numeric](plan *graph.ExecutionPlan[T], ready *atomic.Bool) {
	instructions := plan.Instructions()
	if len(instructions) == 0 {
		return
	}

	// Check if all ops are supported by the code generator.
	unsupported := codegen.CheckSupport(instructions)
	if len(unsupported) > 0 {
		slog.Debug("megakernel: unsupported ops", "count", len(unsupported), "ops", unsupported)
		return
	}

	// Build megakernel config from the plan.
	frozenSlots := plan.FrozenSlots()
	frozenMeta := make([]codegen.FrozenSlotMeta, len(frozenSlots))
	for i, f := range frozenSlots {
		frozenMeta[i] = codegen.FrozenSlotMeta{SlotIdx: f.SlotIdx}
	}

	slotShapes := plan.SlotShapes()

	// Detect KV cache ops and extract dimensions.
	hasKV := detectKVCacheOps(instructions)
	var numKVLayers int
	if hasKV {
		numKVLayers, _, _ = extractKVCacheDims(instructions, slotShapes)
	}

	cfg := codegen.MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   slotShapes,
		FrozenSlots:  frozenMeta,
		InputSlots:   plan.InputSlots(),
		OutputSlot:   plan.OutputSlot(),
		NumKVLayers:  numKVLayers,
	}

	// Emit CUDA source.
	source, err := codegen.EmitMegakernel(cfg)
	if err != nil {
		slog.Debug("megakernel: emit failed", "error", err)
		return
	}

	// Compile to .so (cached by source hash).
	cacheDir := os.TempDir()
	soPath, err := codegen.CachedCompile(source, cacheDir, "megakernel")
	if err != nil {
		slog.Debug("megakernel: compile failed", "error", err)
		return
	}

	// Load the compiled .so.
	runner, err := codegen.LoadMegakernel(soPath)
	if err != nil {
		slog.Debug("megakernel: load failed", "error", err)
		return
	}

	// Extract frozen slot data for GPU upload.
	frozenData := make([][]float32, len(frozenSlots))
	for i, f := range frozenSlots {
		if f.Data != nil {
			raw := f.Data.Data()
			f32 := make([]float32, len(raw))
			for j, v := range raw {
				f32[j] = float32(v)
			}
			frozenData[i] = f32
		}
	}

	// Allocate GPU workspace and upload weights.
	if err := runner.PrepareWorkspace(cfg, frozenData); err != nil {
		slog.Debug("megakernel: workspace preparation failed", "error", err)
		_ = runner.Close()
		return
	}

	// Allocate GPU KV cache and wire device pointers to runner.
	var kvCache *GPUKVCache
	if hasKV && numKVLayers > 0 {
		_, numHeads, headDim := extractKVCacheDims(instructions, slotShapes)
		if numHeads > 0 && headDim > 0 {
			alloc := cudaAllocator{}
			const defaultMaxSeqLen = 512
			var kvErr error
			kvCache, kvErr = NewGPUKVCache(alloc, numKVLayers, defaultMaxSeqLen, numHeads, headDim)
			if kvErr != nil {
				slog.Debug("megakernel: failed to allocate GPU KV cache", "error", kvErr)
				_ = runner.Close()
				return
			}
			kPtrs, vPtrs, ptrErr := kvCache.DevicePointerArrays()
			if ptrErr != nil {
				slog.Debug("megakernel: failed to get KV device pointer arrays", "error", ptrErr)
				_ = kvCache.Close()
				_ = runner.Close()
				return
			}
			runner.SetKVCache(kPtrs, vPtrs)
		}
	}

	outputShape := runner.OutputShape()

	// Set the megakernel function on the plan.
	plan.SetMegakernelFn(func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
		if len(inputs) == 0 {
			return nil, nil
		}

		// Extract input data as float32.
		inputRaw := inputs[0].Data()
		inputF32 := make([]float32, len(inputRaw))
		for i, v := range inputRaw {
			inputF32[i] = float32(v)
		}

		// Wire pos from KV cache sequence length (not hardcoded).
		pos := 0
		if kvCache != nil {
			pos = kvCache.SeqLen()
		}

		outputF32, err := runner.Launch(inputF32, pos)
		if err != nil {
			return nil, err
		}

		// Convert output back to T and wrap in tensor.
		outputT := make([]T, len(outputF32))
		for i, v := range outputF32 {
			outputT[i] = T(v)
		}

		shape := outputShape
		if shape == nil {
			shape = []int{1, 1, len(outputF32)}
		}

		return tensor.New(shape, outputT)
	})

	if ready != nil {
		ready.Store(true)
	}
	slog.Info("megakernel: compiled and loaded", "instructions", len(instructions), "soPath", soPath)
}
