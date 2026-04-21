package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// transposeWeight2D transposes a 2D weight tensor while preserving its underlying
// storage type. For quantized formats (Q4, Q4_K, Q5_0, Q5_K, Q6_K) it performs
// a virtual transpose (swapping shape dimensions) so that quantized blocks stay
// intact and fused dequant+GEMV kernels can be used at inference time.
//
// On the GPU path, Q8 weights are dequantized to F32 for cuBLAS SGEMM. Float16
// and FP8 E4M3 weights are dequantized, transposed, and re-encoded to preserve
// their native storage format.
//
// On the CPU path, all quantized formats (Q4, Q8, Q4_K, Q5_0, Q5_K, Q6_K) use
// virtual transpose. Float16 and FP8 are dequantized, transposed, and re-encoded.
//
// For all other storage types, engine.Transpose is used as the fallback.
func transposeWeight2D(engine compute.Engine[float32], isGPUEngine bool, name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	s := t.GetStorage()

	if isGPUEngine {
		// GPU path: virtual transpose for quantized weights that have fused
		// dequant+GEMV kernels.
		switch any(s).(type) {
		case *tensor.Q4Storage, *tensor.Q4KStorage, *tensor.Q5_0Storage, *tensor.Q5KStorage, *tensor.Q6KStorage:
			shape := t.Shape()
			if len(shape) == 2 {
				return tensor.NewWithStorage[float32]([]int{shape[1], shape[0]}, s)
			}
		}

		shape := t.Shape()
		if len(shape) == 2 {
			rows, cols := shape[0], shape[1]

			// Q8: dequantize to F32 for cuBLAS SGEMM.
			if qs, ok := any(s).(*tensor.Q8Storage); ok {
				f32 := make([]float32, qs.Len())
				qs.Dequantize(f32)
				transposed := make([]float32, len(f32))
				for r := range rows {
					for c := range cols {
						transposed[c*rows+r] = f32[r*cols+c]
					}
				}
				return tensor.New([]int{cols, rows}, transposed)
			}

			// Float16: dequantize, transpose, re-encode to preserve Float16Storage.
			// Without this, engine.Transpose produces F32 storage and FP16 weights
			// lose their native format, doubling memory and missing the FP16 MatMul path.
			if fs, ok := any(s).(*tensor.Float16Storage); ok {
				f32 := fs.Slice()
				transposed := make([]float32, len(f32))
				for r := range rows {
					for c := range cols {
						transposed[c*rows+r] = f32[r*cols+c]
					}
				}
				fp16 := tensor.NewFloat16StorageFromF32(transposed)
				return tensor.NewWithStorage[float32]([]int{cols, rows}, fp16)
			}

			// BFloat16: dequantize, transpose, re-encode to preserve BFloat16Storage.
			// Same rationale as Float16: without this, engine.Transpose produces F32
			// storage and the BF16 MatMul path is never invoked. For Gemma 4 edge
			// Q4_K_M GGUFs, `model.ple_model_proj.weight` is stored as BF16 and was
			// falling through to the dequant-to-F32 fallback, producing degenerate
			// decode output (T99.2.2 H13).
			if fs, ok := any(s).(*tensor.BFloat16Storage); ok {
				f32 := fs.Slice()
				transposed := make([]float32, len(f32))
				for r := range rows {
					for c := range cols {
						transposed[c*rows+r] = f32[r*cols+c]
					}
				}
				bf16 := tensor.NewBFloat16Storage(transposed)
				return tensor.NewWithStorage[float32]([]int{cols, rows}, bf16)
			}

			// FP8 E4M3: dequantize, transpose, re-quantize to preserve FP8E4M3Storage.
			// Without this, engine.Transpose produces F32 storage and the FP8 MatMul
			// path is never invoked, causing degenerate output from double quantization
			// (FP8->F32->FP16 in the generic fp16MatMul fallback).
			if fs, ok := any(s).(*tensor.FP8E4M3Storage); ok {
				f32 := fs.Slice()
				transposed := make([]float32, len(f32))
				for r := range rows {
					for c := range cols {
						transposed[c*rows+r] = f32[r*cols+c]
					}
				}
				fp8 := tensor.NewFP8E4M3Storage(transposed)
				return tensor.NewWithStorage[float32]([]int{cols, rows}, fp8)
			}
		}

		tr, err := engine.Transpose(context.Background(), t, []int{1, 0})
		if err != nil {
			return nil, fmt.Errorf("transpose %s: %w", name, err)
		}
		return tr, nil
	}

	// CPU path: virtual transpose for all quantized storage types.
	switch any(s).(type) {
	case *tensor.Q4Storage, *tensor.Q8Storage, *tensor.Q4KStorage, *tensor.Q5_0Storage, *tensor.Q5KStorage, *tensor.Q6KStorage:
		shape := t.Shape()
		if len(shape) == 2 {
			return tensor.NewWithStorage[float32]([]int{shape[1], shape[0]}, s)
		}
	}

	// Float16: dequantize, transpose, re-encode to preserve compact storage.
	if fs, ok := any(s).(*tensor.Float16Storage); ok {
		shape := t.Shape()
		if len(shape) == 2 {
			f32 := fs.Slice()
			rows, cols := shape[0], shape[1]
			transposed := make([]float32, len(f32))
			for r := range rows {
				for c := range cols {
					transposed[c*rows+r] = f32[r*cols+c]
				}
			}
			fp16 := tensor.NewFloat16StorageFromF32(transposed)
			return tensor.NewWithStorage[float32]([]int{cols, rows}, fp16)
		}
	}

	// BFloat16: dequantize, transpose, re-encode to preserve BFloat16Storage.
	// Without this, engine.Transpose produces F32 storage and the BF16 MatMul
	// path is never invoked. Gemma 4 edge Q4_K_M GGUFs store
	// `model.ple_model_proj.weight` as BF16; the F32 fallback caused degenerate
	// decode output (T99.2.2 H13).
	if fs, ok := any(s).(*tensor.BFloat16Storage); ok {
		shape := t.Shape()
		if len(shape) == 2 {
			f32 := fs.Slice()
			rows, cols := shape[0], shape[1]
			transposed := make([]float32, len(f32))
			for r := range rows {
				for c := range cols {
					transposed[c*rows+r] = f32[r*cols+c]
				}
			}
			bf16 := tensor.NewBFloat16Storage(transposed)
			return tensor.NewWithStorage[float32]([]int{cols, rows}, bf16)
		}
	}

	// FP8 E4M3: dequantize, transpose, re-quantize to preserve FP8E4M3Storage.
	if fs, ok := any(s).(*tensor.FP8E4M3Storage); ok {
		shape := t.Shape()
		if len(shape) == 2 {
			f32 := fs.Slice()
			rows, cols := shape[0], shape[1]
			transposed := make([]float32, len(f32))
			for r := range rows {
				for c := range cols {
					transposed[c*rows+r] = f32[r*cols+c]
				}
			}
			fp8 := tensor.NewFP8E4M3Storage(transposed)
			return tensor.NewWithStorage[float32]([]int{cols, rows}, fp8)
		}
	}

	tr, err := engine.Transpose(context.Background(), t, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("transpose %s: %w", name, err)
	}
	return tr, nil
}
