package generate

import (
	"context"
	"log"
	"os"
	"sync/atomic"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/internal/codegen"
	"github.com/zerfoo/zerfoo/tensor"
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
		return
	}

	// Build megakernel config from the plan.
	frozenSlots := plan.FrozenSlots()
	frozenMeta := make([]codegen.FrozenSlotMeta, len(frozenSlots))
	for i, f := range frozenSlots {
		frozenMeta[i] = codegen.FrozenSlotMeta{SlotIdx: f.SlotIdx}
	}

	cfg := codegen.MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   plan.SlotShapes(),
		FrozenSlots:  frozenMeta,
		InputSlots:   plan.InputSlots(),
		OutputSlot:   plan.OutputSlot(),
	}

	// Emit CUDA source.
	source, err := codegen.EmitMegakernel(cfg)
	if err != nil {
		return
	}

	// Compile to .so (cached by source hash).
	cacheDir := os.TempDir()
	soPath, err := codegen.CachedCompile(source, cacheDir, "megakernel")
	if err != nil {
		return
	}

	// Load the compiled .so.
	runner, err := codegen.LoadMegakernel(soPath)
	if err != nil {
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
		_ = runner.Close()
		return
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

		// Launch kernel.
		pos := 0 // position for rotary embeddings (TODO: wire from KV cache)
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
	log.Printf("megakernel: compiled and loaded (%d instructions, %s)", len(instructions), soPath)
}
