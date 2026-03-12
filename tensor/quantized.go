package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/device"
)

const q4BlockSize = 32

// q4Block represents a single Q4_0 quantization block.
// Format: 2 bytes float16 scale + 16 bytes packed 4-bit data = 18 bytes per 32 values.
type q4Block struct {
	scale float16.Float16
	data  [16]byte // 32 x 4-bit values packed into 16 bytes
}

// Q4Storage holds Q4_0 quantized tensor data on CPU.
type Q4Storage struct {
	blocks      []q4Block
	len         int        // number of logical float32 elements (before padding)
	cachedSlice []float32  // lazily populated on first Slice() call

	// GPU-resident copy of the raw bytes (optional).
	// Set by GPUEngine.UploadWeights to avoid per-op H2D copies.
	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int
}

// QuantizeQ4 quantizes a float32 slice into Q4_0 format.
// The input is padded to a multiple of 32 if necessary.
func QuantizeQ4(src []float32) *Q4Storage {
	n := len(src)
	nBlocks := (n + q4BlockSize - 1) / q4BlockSize
	blocks := make([]q4Block, nBlocks)

	for bi := range nBlocks {
		offset := bi * q4BlockSize

		// Find absmax for this block.
		var absMax float32
		for j := range q4BlockSize {
			idx := offset + j
			var v float32
			if idx < n {
				v = src[idx]
			}
			if av := float32(math.Abs(float64(v))); av > absMax {
				absMax = av
			}
		}

		// Compute scale: maps [-absMax, absMax] to [-8, 7] (signed 4-bit range).
		var scale float32
		if absMax > 0 {
			scale = absMax / 7.0
		}
		blocks[bi].scale = float16.FromFloat32(scale)

		// Quantize values to 4-bit signed integers and pack.
		// Low nibble stores the first half (positions 0-15), high nibble stores
		// the second half (positions 16-31), matching the GGML Q4_0 format.
		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}
		const halfBlock = q4BlockSize / 2
		for j := range halfBlock {
			var v0, v1 float32
			if offset+j < n {
				v0 = src[offset+j]
			}
			if offset+j+halfBlock < n {
				v1 = src[offset+j+halfBlock]
			}

			q0 := clampInt(int(math.Round(float64(v0*invScale))), -8, 7)
			q1 := clampInt(int(math.Round(float64(v1*invScale))), -8, 7)

			// Pack: low nibble = first half element, high nibble = second half element.
			blocks[bi].data[j] = byte(q0+8) | (byte(q1+8) << 4)
		}
	}

	return &Q4Storage{blocks: blocks, len: n}
}

// Dequantize unpacks Q4_0 blocks into dst. len(dst) must be >= q.Len().
// Low nibbles map to the first half (positions 0-15) and high nibbles map to
// the second half (positions 16-31), matching llama.cpp's dequantize_row_q4_0.
func (q *Q4Storage) Dequantize(dst []float32) {
	const halfBlock = q4BlockSize / 2
	for bi, blk := range q.blocks {
		scale := blk.scale.ToFloat32()
		offset := bi * q4BlockSize
		for j := range halfBlock {
			packed := blk.data[j]
			q0 := int(packed&0x0F) - 8
			q1 := int(packed>>4) - 8

			if idx := offset + j; idx < q.len {
				dst[idx] = float32(q0) * scale
			}
			if idx := offset + j + halfBlock; idx < q.len {
				dst[idx] = float32(q1) * scale
			}
		}
	}
}

// Len returns the number of logical float32 elements.
func (q *Q4Storage) Len() int { return q.len }

// NumBlocks returns the number of Q4_0 blocks.
func (q *Q4Storage) NumBlocks() int { return len(q.blocks) }

// ByteSize returns the raw byte size of the quantized data.
// Each block is 18 bytes (2 byte scale + 16 bytes packed data).
func (q *Q4Storage) ByteSize() int { return len(q.blocks) * 18 }

// Slice returns a dequantized float32 view of the data.
// The result is cached: the first call dequantizes, subsequent calls
// return the same slice. This avoids O(N) re-allocation and GC pressure
// when operations like MatMul call Data() on Q4-backed weight tensors.
func (q *Q4Storage) Slice() []float32 {
	if q.cachedSlice != nil {
		return q.cachedSlice
	}
	dst := make([]float32, q.len)
	q.Dequantize(dst)
	q.cachedSlice = dst
	return dst
}

// Set is not supported on quantized storage (weights are immutable).
func (q *Q4Storage) Set(_ []float32) { panic("Q4Storage is immutable") }

// DeviceType returns device.CPU.
func (q *Q4Storage) DeviceType() device.Type { return device.CPU }

// RawBytes serializes Q4_0 blocks as contiguous bytes for GPU upload.
// Each block is 18 bytes: 2 bytes little-endian float16 scale + 16 bytes packed data.
func (q *Q4Storage) RawBytes() []byte {
	out := make([]byte, len(q.blocks)*18)
	for i, blk := range q.blocks {
		off := i * 18
		binary.LittleEndian.PutUint16(out[off:off+2], blk.scale.Bits())
		copy(out[off+2:off+18], blk.data[:])
	}
	return out
}

// RawBytesGPU serializes Q4_0 blocks in a GPU-optimized separated layout.
// The layout is global (not per-row), so it works regardless of how the
// weight matrix is logically viewed (before or after virtual transpose):
//
//	[all_scales: N * 2 bytes] [padding to 16-byte align] [all_data: N * 16 bytes]
//
// The kernel indexes by block_idx = row * blocks_per_row + bi, which is
// the same linear block index regardless of the row definition.
//
// blocksPerRow is unused but kept for API compatibility.
func (q *Q4Storage) RawBytesGPU(blocksPerRow int) []byte {
	totalBlocks := len(q.blocks)
	scaleBytes := totalBlocks * 2
	// Pad scales to 16-byte boundary for aligned uint4 loads on data.
	paddedScaleBytes := (scaleBytes + 15) &^ 15
	dataBytes := totalBlocks * 16
	totalSize := paddedScaleBytes + dataBytes

	out := make([]byte, totalSize)

	// Write all scales contiguously.
	for i, blk := range q.blocks {
		binary.LittleEndian.PutUint16(out[i*2:i*2+2], blk.scale.Bits())
	}
	// Write all packed data contiguously after the padded scale region.
	for i, blk := range q.blocks {
		copy(out[paddedScaleBytes+i*16:paddedScaleBytes+i*16+16], blk.data[:])
	}
	return out
}

// Q4GPUScaleOffset returns the byte offset from the start of RawBytesGPU
// output where the scale region begins (always 0).
func Q4GPUScaleOffset() int { return 0 }

// Q4GPUDataOffset returns the byte offset from the start of RawBytesGPU
// output where the packed data region begins, given the total number of blocks.
func Q4GPUDataOffset(totalBlocks int) int {
	scaleBytes := totalBlocks * 2
	return (scaleBytes + 15) &^ 15
}

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw bytes.
// byteSize must match len(RawBytes()). The caller retains ownership of the pointer.
func (q *Q4Storage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	q.gpuPtr = ptr
	q.gpuByteSize = byteSize
	q.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
// Returns nil if no GPU copy exists.
func (q *Q4Storage) GPUPtr() (unsafe.Pointer, int, int) {
	return q.gpuPtr, q.gpuByteSize, q.gpuDeviceID
}

// BlockScaleF32 returns the dequantization scale for block i as float32.
func (q *Q4Storage) BlockScaleF32(i int) float32 {
	return q.blocks[i].scale.ToFloat32()
}

// BlockData returns a pointer to the 16 packed bytes for block i.
func (q *Q4Storage) BlockData(i int) *byte {
	return &q.blocks[i].data[0]
}

// NewQ4StorageFromRaw creates Q4Storage from raw block data in the standard
// Q4_0 format: 18 bytes per block (2 bytes float16 scale LE + 16 bytes packed nibbles).
// numElements is the number of logical float32 elements the data represents.
func NewQ4StorageFromRaw(raw []byte, numElements int) (*Q4Storage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + q4BlockSize - 1) / q4BlockSize
	const blockBytes = 18
	if len(raw) < nBlocks*blockBytes {
		return nil, fmt.Errorf("Q4_0 raw data too short: need %d bytes for %d blocks, got %d", nBlocks*blockBytes, nBlocks, len(raw))
	}

	blocks := make([]q4Block, nBlocks)
	for i := range nBlocks {
		off := i * blockBytes
		blocks[i].scale = float16.FromBits(binary.LittleEndian.Uint16(raw[off : off+2]))
		copy(blocks[i].data[:], raw[off+2:off+blockBytes])
	}
	return &Q4Storage{blocks: blocks, len: numElements}, nil
}

// BlockPtr returns an unsafe pointer to block i's q4Block struct (18 bytes).
// The layout is: 2 bytes float16 scale (LE) + 16 bytes packed nibble data.
// Blocks are contiguous in memory with 18-byte stride.
func (q *Q4Storage) BlockPtr(i int) *byte {
	return (*byte)(unsafe.Pointer(&q.blocks[i]))
}

// Ensure Q4Storage implements Storage[float32].
var _ Storage[float32] = (*Q4Storage)(nil)

// ---------------------------------------------------------------------------
// Q8_0 format: 32 values per block.
// Each block = 4 bytes float32 scale + 32 bytes int8 data = 36 bytes per 32 values.
// ---------------------------------------------------------------------------

const q8BlockSize = 32

type q8Block struct {
	scale float32
	data  [32]int8
}

// Q8Storage holds Q8_0 quantized tensor data on CPU.
type Q8Storage struct {
	blocks []q8Block
	len    int

	// GPU-resident copy of the raw bytes (optional).
	// Set by GPUEngine.UploadWeights to avoid per-op dequant+H2D copies.
	gpuPtr      unsafe.Pointer
	gpuByteSize int
	gpuDeviceID int
}

// QuantizeQ8 quantizes a float32 slice into Q8_0 format.
func QuantizeQ8(src []float32) *Q8Storage {
	n := len(src)
	nBlocks := (n + q8BlockSize - 1) / q8BlockSize
	blocks := make([]q8Block, nBlocks)

	for bi := range nBlocks {
		offset := bi * q8BlockSize

		var absMax float32
		for j := range q8BlockSize {
			idx := offset + j
			var v float32
			if idx < n {
				v = src[idx]
			}
			if av := float32(math.Abs(float64(v))); av > absMax {
				absMax = av
			}
		}

		var scale float32
		if absMax > 0 {
			scale = absMax / 127.0
		}
		blocks[bi].scale = scale

		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}
		for j := range q8BlockSize {
			var v float32
			if offset+j < n {
				v = src[offset+j]
			}
			blocks[bi].data[j] = int8(clampInt(int(math.Round(float64(v*invScale))), -128, 127))
		}
	}

	return &Q8Storage{blocks: blocks, len: n}
}

// Dequantize unpacks Q8_0 blocks into dst.
func (q *Q8Storage) Dequantize(dst []float32) {
	for bi, blk := range q.blocks {
		offset := bi * q8BlockSize
		for j := range q8BlockSize {
			idx := offset + j
			if idx >= q.len {
				break
			}
			dst[idx] = float32(blk.data[j]) * blk.scale
		}
	}
}

// DequantizeBlock unpacks a single Q8_0 block into a 32-element buffer.
func (q *Q8Storage) DequantizeBlock(blockIdx int, dst *[32]float32) {
	blk := q.blocks[blockIdx]
	for j := range 32 {
		dst[j] = float32(blk.data[j]) * blk.scale
	}
}

// DequantizeRange unpacks Q8_0 blocks covering the range [start, start+count)
// into dst, which must have length >= count.
func (q *Q8Storage) DequantizeRange(dst []float32, start, count int) {
	end := start + count
	if end > q.len {
		end = q.len
	}
	startBlock := start / q8BlockSize
	endBlock := (end + q8BlockSize - 1) / q8BlockSize
	if endBlock > len(q.blocks) {
		endBlock = len(q.blocks)
	}
	for bi := startBlock; bi < endBlock; bi++ {
		blk := q.blocks[bi]
		offset := bi * q8BlockSize
		for j := range q8BlockSize {
			idx := offset + j
			if idx < start {
				continue
			}
			if idx >= end {
				break
			}
			dst[idx-start] = float32(blk.data[j]) * blk.scale
		}
	}
}

// Len returns the number of logical float32 elements.
func (q *Q8Storage) Len() int { return q.len }

// NumBlocks returns the number of Q8_0 blocks.
func (q *Q8Storage) NumBlocks() int { return len(q.blocks) }

// ByteSize returns the raw byte size of the quantized data.
func (q *Q8Storage) ByteSize() int { return len(q.blocks) * 36 }

// Slice returns a dequantized float32 copy of the data.
func (q *Q8Storage) Slice() []float32 {
	dst := make([]float32, q.len)
	q.Dequantize(dst)
	return dst
}

// Set is not supported on quantized storage (weights are immutable).
func (q *Q8Storage) Set(_ []float32) { panic("Q8Storage is immutable") }

// DeviceType returns device.CPU.
func (q *Q8Storage) DeviceType() device.Type { return device.CPU }

// BlockScale returns the float32 scale for block i.
func (q *Q8Storage) BlockScale(i int) float32 {
	return q.blocks[i].scale
}

// BlockQuants returns the int8 quantized values for block i.
func (q *Q8Storage) BlockQuants(i int) []int8 {
	return q.blocks[i].data[:]
}

// RawBytes serializes Q8_0 blocks as contiguous bytes for GPU upload.
// Each block is 36 bytes: 4 bytes little-endian float32 scale + 32 bytes int8 data.
func (q *Q8Storage) RawBytes() []byte {
	const blockBytes = 36
	out := make([]byte, len(q.blocks)*blockBytes)
	for i, blk := range q.blocks {
		off := i * blockBytes
		binary.LittleEndian.PutUint32(out[off:off+4], math.Float32bits(blk.scale))
		for j, v := range blk.data {
			out[off+4+j] = byte(v)
		}
	}
	return out
}

// SetGPUPtr stores a pre-uploaded GPU device pointer for the raw bytes.
// byteSize must match len(RawBytes()). The caller retains ownership of the pointer.
func (q *Q8Storage) SetGPUPtr(ptr unsafe.Pointer, byteSize, deviceID int) {
	q.gpuPtr = ptr
	q.gpuByteSize = byteSize
	q.gpuDeviceID = deviceID
}

// GPUPtr returns the cached GPU device pointer, byte size, and device ID.
// Returns nil if no GPU copy exists.
func (q *Q8Storage) GPUPtr() (unsafe.Pointer, int, int) {
	return q.gpuPtr, q.gpuByteSize, q.gpuDeviceID
}

// NewQ8StorageFromBlocks creates Q8Storage from pre-decoded block data.
// scales has one entry per block. quants has 32 int8 values per block (flattened).
// numElements is the number of logical float32 elements.
func NewQ8StorageFromBlocks(scales []float32, quants []int8, numElements int) (*Q8Storage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + q8BlockSize - 1) / q8BlockSize
	if len(scales) != nBlocks {
		return nil, fmt.Errorf("expected %d scales for %d elements, got %d", nBlocks, numElements, len(scales))
	}
	if len(quants) != nBlocks*q8BlockSize {
		return nil, fmt.Errorf("expected %d quants for %d blocks, got %d", nBlocks*q8BlockSize, nBlocks, len(quants))
	}

	blocks := make([]q8Block, nBlocks)
	for i := range nBlocks {
		blocks[i].scale = scales[i]
		copy(blocks[i].data[:], quants[i*q8BlockSize:(i+1)*q8BlockSize])
	}
	return &Q8Storage{blocks: blocks, len: numElements}, nil
}

// Ensure Q8Storage implements Storage[float32].
var _ Storage[float32] = (*Q8Storage)(nil)

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
