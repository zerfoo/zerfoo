package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"strings"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/tensor"
)

// LoadTensors reads tensor data from a parsed GGUF file and returns them as
// float32 tensors keyed by name. Quantized tensors (Q4_0, Q8_0) are stored
// using their native quantized storage types for memory efficiency.
func LoadTensors(f *File, r io.ReadSeeker) (map[string]*tensor.TensorNumeric[float32], error) {
	result := make(map[string]*tensor.TensorNumeric[float32], len(f.Tensors))

	for i := range f.Tensors {
		ti := &f.Tensors[i]

		// Compute number of elements.
		numElements := 1
		for _, d := range ti.Dimensions {
			numElements *= int(d)
		}

		// Convert dimensions to int for tensor API.
		// GGUF stores dimensions in GGML order (innermost-first: ne[0]=columns,
		// ne[1]=rows). Reverse to match PyTorch convention (outermost-first:
		// shape[0]=rows, shape[1]=columns).
		shape := make([]int, len(ti.Dimensions))
		for j, d := range ti.Dimensions {
			shape[len(ti.Dimensions)-1-j] = int(d)
		}

		// Compute byte size of this tensor's data.
		dataSize, err := tensorByteSize(ti.Type, numElements)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}

		// Seek to tensor data.
		offset := f.DataOffset + int64(ti.Offset)
		if _, err := r.Seek(offset, io.SeekStart); err != nil {
			return nil, fmt.Errorf("tensor %q: seek to offset %d: %w", ti.Name, offset, err)
		}

		// Read raw data.
		raw := make([]byte, dataSize)
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, fmt.Errorf("tensor %q: read %d bytes: %w", ti.Name, dataSize, err)
		}

		// Decode based on type.
		t, err := decodeTensor(ti.Type, shape, numElements, raw)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}
		result[ti.Name] = t
	}

	return result, nil
}

// tensorByteSize returns the number of bytes needed for a tensor of the given type and element count.
func tensorByteSize(typ GGMLType, numElements int) (int, error) {
	switch typ {
	case GGMLTypeF32:
		return numElements * 4, nil
	case GGMLTypeF16:
		return numElements * 2, nil
	case GGMLTypeQ4_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 18, nil // 2 bytes scale + 16 bytes data
	case GGMLTypeQ8_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 34, nil // 2 bytes fp16 scale + 32 bytes int8 (GGUF format)
	case GGMLTypeQ5_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 22, nil // 2 bytes fp16 scale + 4 bytes high bits + 16 bytes low nibbles
	case GGMLTypeQ4_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 144, nil // 4 bytes header + 12 bytes scales + 128 bytes data
	case GGMLTypeQ5_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 176, nil // 4 bytes header + 12 bytes scales + 128 bytes ql + 32 bytes qh
	case GGMLTypeQ6_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 210, nil // 128 bytes ql + 64 bytes qh + 16 bytes scales + 2 bytes d
	case GGMLTypeBF16:
		return numElements * 2, nil
	default:
		return 0, fmt.Errorf("unsupported GGML type %d", typ)
	}
}

// decodeTensor converts raw bytes into a tensor based on GGML type.
func decodeTensor(typ GGMLType, shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	switch typ {
	case GGMLTypeF32:
		return decodeF32Tensor(shape, numElements, raw)
	case GGMLTypeF16:
		return decodeF16Tensor(shape, numElements, raw)
	case GGMLTypeQ4_0:
		return decodeQ4Tensor(shape, numElements, raw)
	case GGMLTypeQ8_0:
		return decodeQ8Tensor(shape, numElements, raw)
	case GGMLTypeQ5_0:
		return decodeQ5_0Tensor(shape, numElements, raw)
	case GGMLTypeQ4_K:
		return decodeQ4KTensor(shape, numElements, raw)
	case GGMLTypeQ5_K:
		return decodeQ5KTensor(shape, numElements, raw)
	case GGMLTypeQ6_K:
		return decodeQ6KTensor(shape, numElements, raw)
	case GGMLTypeBF16:
		return decodeBF16Tensor(shape, numElements, raw)
	default:
		return nil, fmt.Errorf("unsupported GGML type %d", typ)
	}
}

func decodeF32Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	data := make([]float32, numElements)
	for i := range numElements {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
	}
	return tensor.New[float32](shape, data)
}

func decodeF16Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	if len(raw) < numElements*2 {
		return nil, fmt.Errorf("F16 decode: need %d bytes, got %d", numElements*2, len(raw))
	}
	fp16 := tensor.NewFloat16StorageFromRaw(raw[:numElements*2], numElements)
	return tensor.NewWithStorage[float32](shape, fp16)
}

func decodeQ4Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	q4, err := tensor.NewQ4StorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q4_0 decode: %w", err)
	}
	return tensor.NewWithStorage[float32](shape, q4)
}

func decodeQ4KTensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	q4k, err := tensor.NewQ4KStorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q4_K decode: %w", err)
	}
	// Re-quantize Q4_K to Q4_0. The Q4_K fused GEMV kernel exists but native
	// Q4_K is ~20% slower than Q4_0 GEMV (120 vs 149 tok/s on GB10). The Q4_0
	// path trades per-sub-block 6-bit scale precision for throughput.
	// TODO: optimize Q4_K GEMV to match Q4_0 speed, then use native Q4_K.
	f32 := make([]float32, numElements)
	q4k.Dequantize(f32)
	q4 := tensor.QuantizeQ4(f32)
	return tensor.NewWithStorage[float32](shape, q4)
}

func decodeQ5KTensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	q5k, err := tensor.NewQ5KStorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q5_K decode: %w", err)
	}
	// Native Q5_K storage with fused GemvQ5KF32 dispatch.
	return tensor.NewWithStorage[float32](shape, q5k)
}

func decodeQ6KTensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	q6k, err := tensor.NewQ6KStorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q6_K decode: %w", err)
	}
	// Use native Q6_K storage. The GPU engine dispatches to fused GemvQ6KF32
	// for batch=1 decode. PreUploadFrozenWeights skips Q6KStorage (preserves
	// quantized format). UploadWeights handles GPU pre-upload of raw bytes.
	return tensor.NewWithStorage[float32](shape, q6k)
}

// decodeQ5_0Tensor decodes Q5_0 blocks and re-quantizes to Q4_0 for fast GEMV.
// Q5_0 format: 32 elements per block, 22 bytes per block.
// Layout: 2 bytes fp16 scale + 4 bytes high bits + 16 bytes low 4-bit values.
// Each byte in qs contains two 4-bit values: the low nibble maps to the first
// half of the block (positions 0-15) and the high nibble maps to the second
// half (positions 16-31). This matches llama.cpp's dequantize_row_q5_0.
func decodeQ5_0Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	q5_0, err := tensor.NewQ5_0StorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q5_0 decode: %w", err)
	}
	// Native Q5_0 storage with fused GemvQ5_0F32 dispatch.
	// Eliminates the lossy Q5_0→Q4_0 re-quantization that caused garbled output.
	return tensor.NewWithStorage[float32](shape, q5_0)
}

func decodeQ8Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	// GGUF Q8_0 format: 34 bytes per block (2-byte fp16 scale + 32 int8 quants).
	// Zerfoo Q8Storage uses float32 scales. Convert.
	const ggufQ8BlockBytes = 34
	nBlocks := (numElements + 31) / 32

	scales := make([]float32, nBlocks)
	quants := make([]int8, nBlocks*32)

	for bi := range nBlocks {
		off := bi * ggufQ8BlockBytes
		f16Bits := binary.LittleEndian.Uint16(raw[off : off+2])
		scales[bi] = float16.FromBits(f16Bits).ToFloat32()
		for j := range 32 {
			quants[bi*32+j] = int8(raw[off+2+j])
		}
	}

	q8, err := tensor.NewQ8StorageFromBlocks(scales, quants, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q8_0 decode: %w", err)
	}
	return tensor.NewWithStorage[float32](shape, q8)
}

func decodeBF16Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	if len(raw) < numElements*2 {
		return nil, fmt.Errorf("BF16 decode: need %d bytes, got %d", numElements*2, len(raw))
	}
	u16 := make([]uint16, numElements)
	for i := range numElements {
		u16[i] = uint16(raw[i*2]) | uint16(raw[i*2+1])<<8
	}
	bf16 := tensor.NewBFloat16StorageFromRaw(u16)
	return tensor.NewWithStorage[float32](shape, bf16)
}

// QuantizeToFP8E4M3 converts all tensors in the map from their current storage
// to FP8 E4M3 format with per-tensor absmax scaling. This reduces memory to
// 1 byte per element (1/4 of F32) at the cost of reduced precision.
// The tensors are modified in place — the returned map is the same object.
func QuantizeToFP8E4M3(tensors map[string]*tensor.TensorNumeric[float32]) (map[string]*tensor.TensorNumeric[float32], error) {
	for name, t := range tensors {
		// Skip embedding and LM head weights — they are used for gather
		// (not matmul) and FP8 quantization causes degenerate decode output.
		if strings.Contains(name, "embed_tokens") || strings.Contains(name, "lm_head") {
			continue
		}
		// Skip norm weights and biases — they are small and benefit more
		// from full precision than from FP8 compression.
		if strings.Contains(name, "norm") || strings.HasSuffix(name, ".bias") {
			continue
		}
		f32 := t.Data()

		// Log min/max of original F32 data for diagnostics.
		var f32Min, f32Max float32
		if len(f32) > 0 {
			f32Min, f32Max = f32[0], f32[0]
			for _, v := range f32[1:] {
				if v < f32Min {
					f32Min = v
				}
				if v > f32Max {
					f32Max = v
				}
			}
		}

		fp8 := tensor.NewFP8E4M3Storage(f32)
		scale := fp8.Scale()
		log.Printf("[FP8 diag] QuantizeToFP8E4M3: tensor=%q shape=%v elems=%d scale=%.6g f32Min=%.6g f32Max=%.6g",
			name, t.Shape(), len(f32), scale, f32Min, f32Max)
		if scale == 0 || math.IsInf(float64(scale), 0) || math.IsNaN(float64(scale)) {
			log.Printf("[FP8 diag] WARNING: tensor=%q has abnormal scale=%.6g", name, scale)
		}

		quantized, err := tensor.NewWithStorage[float32](t.Shape(), fp8)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: FP8 quantize: %w", name, err)
		}
		tensors[name] = quantized
	}
	return tensors, nil
}
