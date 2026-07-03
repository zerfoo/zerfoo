package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"strings"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/tensor"
)

// embeddingVocabThreshold is the vocab-dimension floor used to classify a 2D
// tensor as an embedding table. Mirrors the existing Q8_0 embedding guard in
// decodeQ8Tensor; Gemma 4 Edge uses 262144 (per_layer_token_embd) and 256000
// (token_embd), both well above 50000, while ordinary weight matrices stay
// under 50000 on production transformer shapes.
const embeddingVocabThreshold = 50000

// isEmbeddingShape reports whether a tensor's shape looks like a 2D embedding
// lookup table (shape[0] is the vocab dimension).
func isEmbeddingShape(shape []int) bool {
	return len(shape) == 2 && shape[0] > embeddingVocabThreshold
}

// maxTensorElements is the element-count cap enforced during GGUF tensor
// loading. It bounds the allocation size computed from file-controlled
// dimensions so a corrupt or malicious file cannot force an oversized (or,
// via overflow, negative) allocation.
const maxTensorElements = 1 << 34

// computeNumElements multiplies a tensor's GGUF dimensions into a total
// element count, validating each dimension and checking for int64 overflow
// BEFORE multiplying rather than after.
//
// This ordering matters: the previous implementation (duplicated across four
// call sites) multiplied first and then checked "numElements > maxTensorElements",
// which has two bugs. First, the running product can land on exactly
// maxTensorElements (not rejected by a strict ">"), after which a further
// dimension multiply overflows int64 to a negative value that also fails the
// "> maxTensorElements" test -- so the negative element count silently passes
// validation and later panics deep in allocation code. Second, checking after
// the multiply means the overflow has already happened by the time it is
// detected. Checking "numElements > maxTensorElements/d" before multiplying
// makes the overflow structurally impossible to reach (deep-review 002,
// finding F1).
func computeNumElements(name string, dimensions []uint64) (int64, error) {
	var numElements int64 = 1
	for _, d := range dimensions {
		if d == 0 {
			return 0, fmt.Errorf("tensor %q: zero dimension is not allowed", name)
		}
		if d > math.MaxInt32 {
			return 0, fmt.Errorf("tensor %q: dimension %d exceeds maximum (%d)", name, d, int64(math.MaxInt32))
		}
		if numElements > maxTensorElements/int64(d) {
			return 0, fmt.Errorf("tensor %q: total elements exceeds maximum (%d)", name, int64(maxTensorElements))
		}
		numElements *= int64(d)
	}
	return numElements, nil
}

// pleNativeKQuantEnabled reports whether ZERFOO_GEMMA4_PLE_NATIVE_Q4K=1 is set.
// When enabled, Q4_K / Q5_K / Q6_K embedding-shaped tensors keep their native
// K-quant storage instead of being re-quantized to Q4_0 at load time.
//
// Motivation (T99.2.2.9 / H21). The default decode path for K-quant tensors
// takes a lossy round-trip: raw bytes -> K-quant storage -> f32 -> Q4_0
// storage. For weight matrices this is fine — the GEMV kernels need the Q4_0
// layout. For gather targets like `model.ple_embed_tokens.weight` (Gemma 4
// Edge Q4_K_M), the doubly-lossy conversion compounds per-row noise across
// the 35 PLE layers and was identified as the top structural candidate for
// the T99.2.2 PLE-decode degeneracy bug. This flag gates the fix so it can
// be A/B'd on DGX before being promoted.
func pleNativeKQuantEnabled() bool {
	return os.Getenv("ZERFOO_GEMMA4_PLE_NATIVE_Q4K") == "1"
}

// LoadTensors reads tensor data from a parsed GGUF file and returns them as
// float32 tensors keyed by name. Quantized tensors (Q4_0, Q8_0) are stored
// using their native quantized storage types for memory efficiency.
func LoadTensors(f *File, r io.ReadSeeker) (map[string]*tensor.TensorNumeric[float32], error) {
	result := make(map[string]*tensor.TensorNumeric[float32], len(f.Tensors))

	for i := range f.Tensors {
		ti := &f.Tensors[i]

		// Compute number of elements with overflow checks.
		numElements, err := computeNumElements(ti.Name, ti.Dimensions)
		if err != nil {
			return nil, err
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
		dataSize, err := TensorByteSize(ti.Type, int(numElements))
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

		// Decode based on type. Embedding tensors keep native quantization
		// to preserve precision (Q6_K has 64 levels vs Q4_0's 16; re-quantizing
		// produces zero values for small embeddings like Llama's BOS token).
		// Decode based on type.
		t, err := decodeTensor(ti.Type, shape, int(numElements), raw)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}
		result[ti.Name] = t
	}

	return result, nil
}

// TensorByteSize returns the number of bytes needed for a tensor of the given type and element count.
func TensorByteSize(typ GGMLType, numElements int) (int, error) {
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
	case GGMLTypeQ2_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 84, nil // 2 bytes d + 2 bytes dmin + 16 bytes scales + 64 bytes qs
	case GGMLTypeQ3_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 110, nil // 32 bytes hmask + 64 bytes qs + 12 bytes scales + 2 bytes d
	case GGMLTypeBF16:
		return numElements * 2, nil
	case GGMLTypeIQ4_NL:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 18, nil // 2 bytes fp16 scale + 16 bytes packed nibbles
	case GGMLTypeIQ3_S:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 110, nil // 2 bytes fp16 d + 64 qs + 8 qh + 32 signs + 4 scales
	case GGMLTypeIQ2_XXS:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 68, nil // 4 bytes fp32 scale + 64 bytes packed 2-bit data
	case GGMLTypeTQ2_0:
		// 2 bits per value, 4 values per byte.
		return (numElements + 3) / 4, nil
	default:
		return 0, fmt.Errorf("unsupported GGML type %d", typ)
	}
}

// decodeTensor converts raw bytes into a tensor based on GGML type.
// decodeTensorNative decodes a tensor keeping native quantized storage.
// Used for embedding tensors where precision matters more than GEMV speed.
func decodeTensorNative(typ GGMLType, shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	switch typ {
	case GGMLTypeQ4_K:
		q4k, err := tensor.NewQ4KStorageFromRaw(raw, numElements)
		if err != nil {
			return nil, fmt.Errorf("Q4_K native decode: %w", err)
		}
		return tensor.NewWithStorage[float32](shape, q4k)
	case GGMLTypeQ5_K:
		q5k, err := tensor.NewQ5KStorageFromRaw(raw, numElements)
		if err != nil {
			return nil, fmt.Errorf("Q5_K native decode: %w", err)
		}
		return tensor.NewWithStorage[float32](shape, q5k)
	case GGMLTypeQ6_K:
		q6k, err := tensor.NewQ6KStorageFromRaw(raw, numElements)
		if err != nil {
			return nil, fmt.Errorf("Q6_K native decode: %w", err)
		}
		return tensor.NewWithStorage[float32](shape, q6k)
	case GGMLTypeQ5_0:
		q5, err := tensor.NewQ5_0StorageFromRaw(raw, numElements)
		if err != nil {
			return nil, fmt.Errorf("Q5_0 native decode: %w", err)
		}
		return tensor.NewWithStorage[float32](shape, q5)
	default:
		// All other types (F32, F16, Q4_0, Q8_0, etc.) go through normal decode.
		return decodeTensor(typ, shape, numElements, raw)
	}
}

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
	case GGMLTypeQ2_K:
		return decodeQ2KTensor(shape, numElements, raw)
	case GGMLTypeQ3_K:
		return decodeQ3KTensor(shape, numElements, raw)
	case GGMLTypeBF16:
		return decodeBF16Tensor(shape, numElements, raw)
	case GGMLTypeIQ4_NL:
		return decodeIQ4NLTensor(shape, numElements, raw)
	case GGMLTypeIQ3_S:
		return decodeIQ3STensor(shape, numElements, raw)
	case GGMLTypeIQ2_XXS:
		return decodeIQ2XXSTensor(shape, numElements, raw)
	case GGMLTypeTQ2_0:
		return decodeTernaryTensor(shape, numElements, raw)
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
	// T99.2.2.9 H21 guard: for embedding-shaped tensors, keep native Q4_K
	// storage when ZERFOO_GEMMA4_PLE_NATIVE_Q4K=1. Avoids the lossy
	// Q4_K -> f32 -> Q4_0 round-trip for gather targets such as
	// model.ple_embed_tokens.weight.
	if pleNativeKQuantEnabled() && isEmbeddingShape(shape) {
		return tensor.NewWithStorage[float32](shape, q4k)
	}
	// Re-quantize Q4_K → Q4_0 for uniform fast GEMV decode path.
	// Q4_0 block_size=32 works with any hidden_size divisible by 32 (all models).
	// Q4_K block_size=256 fails when hidden_size%256!=0 (e.g., Gemma3-1B=1152).
	// The Q4_0 GEMV with separated GPU layout is the fastest decode path.
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
	// T99.2.2.9 H21 guard: symmetric with decodeQ4KTensor. Kept behind the
	// same flag so Q5_K_M / Q6_K_M GGUF variants of the same family inherit
	// the embedding-precision fix.
	if pleNativeKQuantEnabled() && isEmbeddingShape(shape) {
		return tensor.NewWithStorage[float32](shape, q5k)
	}
	f32 := make([]float32, numElements)
	q5k.Dequantize(f32)
	q4 := tensor.QuantizeQ4(f32)
	return tensor.NewWithStorage[float32](shape, q4)
}

func decodeQ6KTensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	q6k, err := tensor.NewQ6KStorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q6_K decode: %w", err)
	}
	// T99.2.2.9 H21 guard: see decodeQ4KTensor.
	if pleNativeKQuantEnabled() && isEmbeddingShape(shape) {
		return tensor.NewWithStorage[float32](shape, q6k)
	}
	f32 := make([]float32, numElements)
	q6k.Dequantize(f32)
	q4 := tensor.QuantizeQ4(f32)
	return tensor.NewWithStorage[float32](shape, q4)
}

func decodeQ5_0Tensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	q5, err := tensor.NewQ5_0StorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("Q5_0 decode: %w", err)
	}
	f32 := make([]float32, numElements)
	q5.Dequantize(f32)
	q4 := tensor.QuantizeQ4(f32)
	return tensor.NewWithStorage[float32](shape, q4)
}

// decodeQ2KTensor decodes Q2_K blocks and re-quantizes to Q4_0 for fast GEMV.
// Q2_K format: 256 elements per super-block, 84 bytes per block.
// Layout: 2 bytes fp16 d + 2 bytes fp16 dmin + 16 bytes scales + 64 bytes qs.
// Each byte in scales encodes two 4-bit values: low nibble = scale, high nibble = min.
// Each byte in qs encodes four 2-bit quantized values.
func decodeQ2KTensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	const blockSize = 256
	const blockBytes = 84
	nBlocks := (numElements + blockSize - 1) / blockSize

	if len(raw) < nBlocks*blockBytes {
		return nil, fmt.Errorf("Q2_K decode: need %d bytes, got %d", nBlocks*blockBytes, len(raw))
	}

	data := make([]float32, numElements)
	for bi := range nBlocks {
		off := bi * blockBytes
		d := float16.FromBits(binary.LittleEndian.Uint16(raw[off : off+2])).ToFloat32()
		dmin := float16.FromBits(binary.LittleEndian.Uint16(raw[off+2 : off+4])).ToFloat32()
		scales := raw[off+4 : off+4+16]
		qs := raw[off+20 : off+20+64]

		for j := range blockSize {
			// Determine which sub-block (of 16) this element belongs to.
			subBlock := j / 16
			sc := scales[subBlock]
			scaleVal := float32(sc & 0x0F)
			minVal := float32(sc >> 4)

			// Extract 2-bit quant from qs.
			byteIdx := j / 4
			bitShift := uint(j%4) * 2
			q := (qs[byteIdx] >> bitShift) & 0x03

			idx := bi*blockSize + j
			if idx < numElements {
				data[idx] = d*scaleVal*float32(q) - dmin*minVal
			}
		}
	}

	q4 := tensor.QuantizeQ4(data)
	return tensor.NewWithStorage[float32](shape, q4)
}

// decodeQ3KTensor decodes Q3_K blocks and re-quantizes to Q4_0 for fast GEMV.
// Q3_K format: 256 elements per super-block, 110 bytes per block.
// Layout: 32 bytes hmask + 64 bytes qs (low 2 bits) + 12 bytes scales + 2 bytes fp16 d.
// The 3-bit value is reconstructed as: low 2 bits from qs + high bit from hmask.
func decodeQ3KTensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	const blockSize = 256
	const blockBytes = 110
	nBlocks := (numElements + blockSize - 1) / blockSize

	if len(raw) < nBlocks*blockBytes {
		return nil, fmt.Errorf("Q3_K decode: need %d bytes, got %d", nBlocks*blockBytes, len(raw))
	}

	data := make([]float32, numElements)
	for bi := range nBlocks {
		off := bi * blockBytes
		hmask := raw[off : off+32]
		qs := raw[off+32 : off+32+64]
		scalesRaw := raw[off+96 : off+96+12]
		d := float16.FromBits(binary.LittleEndian.Uint16(raw[off+108 : off+110])).ToFloat32()

		// Decode 6-bit scales from 12-byte packed representation.
		// 16 sub-blocks of 16 elements each. The 12 bytes encode 16 6-bit scales.
		// Bytes 0-7: low 4 bits for scales 0-15 (two per byte, low/high nibble).
		// Bytes 8-11: high 2 bits for scales 0-15 (four per byte, 2 bits each).
		var scales [16]int8
		for i := range 8 {
			lo := scalesRaw[i] & 0x0F
			hi := (scalesRaw[8+i/4] >> (2 * uint(i%4))) & 0x03
			scales[i] = int8(lo|(hi<<4)) - 32
		}
		for i := range 8 {
			lo := scalesRaw[i] >> 4
			hi := (scalesRaw[10+i/4] >> (2 * uint(i%4))) & 0x03
			scales[8+i] = int8(lo|(hi<<4)) - 32
		}

		for j := range blockSize {
			subBlock := j / 16

			// Low 2 bits from qs.
			byteIdx := j / 4
			bitShift := uint(j%4) * 2
			q := int((qs[byteIdx] >> bitShift) & 0x03)

			// High bit from hmask.
			hmaskByte := j / 8
			hmaskBit := uint(j % 8)
			if (hmask[hmaskByte]>>hmaskBit)&1 == 0 {
				q |= 4
			}

			// The 3-bit value is in [0, 7], center at 4 to get [-4, 3].
			idx := bi*blockSize + j
			if idx < numElements {
				data[idx] = d * float32(scales[subBlock]) * float32(q-4)
			}
		}
	}

	q4 := tensor.QuantizeQ4(data)
	return tensor.NewWithStorage[float32](shape, q4)
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

	// Re-quantize Q8_0 weight matrices (2D, non-embedding) to Q4_0 for uniform
	// fast GEMV decode path. Embeddings (large vocab dim) keep Q8 for precision.
	// Without this, Q8_0 attn_v weights block the merged QKV optimization.
	if len(shape) == 2 && numElements > 0 {
		if !isEmbeddingShape(shape) {
			f32 := make([]float32, numElements)
			for bi := range nBlocks {
				s := scales[bi]
				for j := range 32 {
					idx := bi*32 + j
					if idx < numElements {
						f32[idx] = s * float32(quants[idx])
					}
				}
			}
			q4 := tensor.QuantizeQ4(f32)
			return tensor.NewWithStorage[float32](shape, q4)
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

func decodeTernaryTensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	expectedBytes := (numElements + 3) / 4
	if len(raw) < expectedBytes {
		return nil, fmt.Errorf("TQ2_0 decode: need %d bytes, got %d", expectedBytes, len(raw))
	}
	// Decode ternary (TQ2_0) values: 2 bits per element packed 4 per byte.
	// Encoding matches TernaryStorage: 00=-1, 01=0, 10=+1 (bits - 1).
	ts := tensor.NewTernaryStorage(numElements)
	copy(ts.RawBytes(), raw[:expectedBytes])
	return tensor.NewWithStorage[float32](shape, ts)
}

func decodeIQ4NLTensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	s, err := tensor.NewIQ4NLStorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("IQ4_NL decode: %w", err)
	}
	// Re-quantize IQ4_NL to Q4_0 for fast GEMV decode path.
	f32 := make([]float32, numElements)
	s.Dequantize(f32)
	q4 := tensor.QuantizeQ4(f32)
	return tensor.NewWithStorage[float32](shape, q4)
}

func decodeIQ3STensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	s, err := tensor.NewIQ3SStorageFromRaw(raw, numElements)
	if err != nil {
		return nil, fmt.Errorf("IQ3_S decode: %w", err)
	}
	// Re-quantize IQ3_S to Q4_0 for fast GEMV decode path.
	f32 := make([]float32, numElements)
	s.Dequantize(f32)
	q4 := tensor.QuantizeQ4(f32)
	return tensor.NewWithStorage[float32](shape, q4)
}

func decodeIQ2XXSTensor(shape []int, numElements int, raw []byte) (*tensor.TensorNumeric[float32], error) {
	nBlocks := (numElements + 255) / 256
	const bytesPerBlock = 64 // 256 elements / 4 per byte
	const blockTotalBytes = 68 // 4 bytes scale + 64 bytes data
	if len(raw) < nBlocks*blockTotalBytes {
		return nil, fmt.Errorf("IQ2_XXS decode: need %d bytes, got %d", nBlocks*blockTotalBytes, len(raw))
	}

	s := tensor.NewIQ2XXSStorage(numElements)
	for bi := range nBlocks {
		off := bi * blockTotalBytes
		scale := math.Float32frombits(binary.LittleEndian.Uint32(raw[off : off+4]))
		s.SetBlock(bi, scale, raw[off+4:off+4+bytesPerBlock])
	}

	// Re-quantize IQ2_XXS to Q4_0 for fast GEMV decode path.
	f32 := s.Dequantize()
	q4 := tensor.QuantizeQ4(f32)
	return tensor.NewWithStorage[float32](shape, q4)
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
		slog.Debug("QuantizeToFP8E4M3", "tensor", name, "shape", t.Shape(), "elems", len(f32), "scale", scale, "f32Min", f32Min, "f32Max", f32Max)
		if scale == 0 || math.IsInf(float64(scale), 0) || math.IsNaN(float64(scale)) {
			slog.Warn("tensor has abnormal scale", "tensor", name, "scale", scale)
		}

		quantized, err := tensor.NewWithStorage[float32](t.Shape(), fp8)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: FP8 quantize: %w", name, err)
		}
		tensors[name] = quantized
	}
	return tensors, nil
}
