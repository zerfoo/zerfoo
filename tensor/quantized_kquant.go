package tensor

import (
	"encoding/binary"
	"fmt"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/device"
)

// Q4_K format: super-blocks of 256 values.
// Each super-block is 144 bytes:
//   - 2 bytes: fp16 d (super-block scale)
//   - 2 bytes: fp16 dmin (super-block min)
//   - 12 bytes: packed 6-bit scales and mins for 8 sub-blocks
//   - 128 bytes: 256 x 4-bit quantized values packed
//
// Reference: llama.cpp ggml-quants.c dequantize_row_q4_K
const (
	q4KSuperBlockSize  = 256
	q4KBlockBytes      = 144
	q4KNumSubBlocks    = 8
	q4KSubBlockSize    = 32
)

// decodeQ4KScalesMins extracts the 6-bit scales and mins from the 12-byte
// packed region at raw[4:16] of a Q4_K super-block.
// Layout (matches llama.cpp get_scale_min_k4):
//
//	sc[0:4] = 6-bit scales for sub-blocks 0-3 (bits 6-7 hold high bits of scales 4-7)
//	sc[4:8] = 6-bit mins for sub-blocks 0-3 (bits 6-7 hold high bits of mins 4-7)
//	sc[8:12] = low 4 bits scale + high 4 bits min for sub-blocks 4-7
func decodeQ4KScalesMins(raw []byte) (scales, mins [q4KNumSubBlocks]uint8) {
	sc := raw[4:16] // 12-byte scales region

	// Sub-blocks 0-3: 6 low bits from separate byte ranges.
	for i := range 4 {
		scales[i] = sc[i] & 63
		mins[i] = sc[4+i] & 63
	}
	// Sub-blocks 4-7: 4 bits from bytes 8-11 + 2 high bits from bytes 0-3 / 4-7.
	for i := range 4 {
		scales[4+i] = (sc[8+i] & 0xF) | ((sc[i] >> 6) << 4)
		mins[4+i] = (sc[8+i] >> 4) | ((sc[4+i] >> 6) << 4)
	}
	return
}

// DequantizeQ4K dequantizes one Q4_K super-block (144 bytes) into 256 float32 values.
// Each 32 bytes of quantized data produces 64 output values: low nibbles map to
// the first 32 positions and high nibbles map to the next 32 positions.
// This matches llama.cpp's dequantize_row_q4_K.
func DequantizeQ4K(raw []byte, dst []float32) {
	d := float16.FromBits(binary.LittleEndian.Uint16(raw[0:2])).ToFloat32()
	dmin := float16.FromBits(binary.LittleEndian.Uint16(raw[2:4])).ToFloat32()

	scales, mins := decodeQ4KScalesMins(raw)

	qdata := raw[16:] // 128 bytes of packed 4-bit values
	// Process 4 groups of 64 elements (2 sub-blocks each).
	for group := range 4 {
		sb0 := group * 2
		sb1 := group*2 + 1
		sc0 := d * float32(scales[sb0])
		mn0 := dmin * float32(mins[sb0])
		sc1 := d * float32(scales[sb1])
		mn1 := dmin * float32(mins[sb1])

		baseOut := group * 64
		baseQ := group * 32
		for l := range 32 {
			q := qdata[baseQ+l]
			dst[baseOut+l] = sc0*float32(q&0xF) - mn0
			dst[baseOut+l+32] = sc1*float32(q>>4) - mn1
		}
	}
}

// Q4KStorage holds Q4_K quantized tensor data on CPU.
type Q4KStorage struct {
	raw []byte // raw super-block data
	len int    // number of logical float32 elements
}

// NewQ4KStorageFromRaw creates Q4KStorage from raw super-block data.
func NewQ4KStorageFromRaw(raw []byte, numElements int) (*Q4KStorage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + q4KSuperBlockSize - 1) / q4KSuperBlockSize
	need := nBlocks * q4KBlockBytes
	if len(raw) < need {
		return nil, fmt.Errorf("Q4_K raw data too short: need %d bytes for %d blocks, got %d", need, nBlocks, len(raw))
	}
	data := make([]byte, need)
	copy(data, raw[:need])
	return &Q4KStorage{raw: data, len: numElements}, nil
}

// Dequantize unpacks all Q4_K super-blocks into dst.
func (q *Q4KStorage) Dequantize(dst []float32) {
	nBlocks := (q.len + q4KSuperBlockSize - 1) / q4KSuperBlockSize
	for bi := range nBlocks {
		blockRaw := q.raw[bi*q4KBlockBytes : (bi+1)*q4KBlockBytes]
		off := bi * q4KSuperBlockSize
		remaining := q.len - off
		if remaining >= q4KSuperBlockSize {
			DequantizeQ4K(blockRaw, dst[off:off+q4KSuperBlockSize])
		} else {
			var tmp [q4KSuperBlockSize]float32
			DequantizeQ4K(blockRaw, tmp[:])
			copy(dst[off:], tmp[:remaining])
		}
	}
}

func (q *Q4KStorage) Len() int                    { return q.len }
func (q *Q4KStorage) Slice() []float32             { dst := make([]float32, q.len); q.Dequantize(dst); return dst }
func (q *Q4KStorage) Set(_ []float32)              { panic("Q4KStorage is immutable") }
func (q *Q4KStorage) DeviceType() device.Type      { return device.CPU }

var _ Storage[float32] = (*Q4KStorage)(nil)

// Q6_K format: super-blocks of 256 values.
// Each super-block is 210 bytes:
//   - 128 bytes: ql (low 4 bits of each 6-bit value)
//   - 64 bytes: qh (high 2 bits of each 6-bit value)
//   - 16 bytes: int8 scales for 16 sub-blocks of 16 values
//   - 2 bytes: fp16 d (super-block scale)
//
// Reference: llama.cpp ggml-quants.c dequantize_row_q6_K
const (
	q6KSuperBlockSize = 256
	q6KBlockBytes     = 210
)

// DequantizeQ6K dequantizes one Q6_K super-block (210 bytes) into 256 float32 values.
// Each 128-element half uses 64 ql bytes + 32 qh bytes to produce 4 groups of 32:
//   low nibbles of ql[0:32]  + qh bits 0-1 -> positions 0-31
//   low nibbles of ql[32:64] + qh bits 2-3 -> positions 32-63
//   high nibbles of ql[0:32] + qh bits 4-5 -> positions 64-95
//   high nibbles of ql[32:64]+ qh bits 6-7 -> positions 96-127
// This matches llama.cpp's dequantize_row_q6_K.
func DequantizeQ6K(raw []byte, dst []float32) {
	ql := raw[0:128]   // low 4 bits
	qh := raw[128:192] // high 2 bits
	sc := raw[192:208] // int8 scales for 16 sub-blocks
	d := float16.FromBits(binary.LittleEndian.Uint16(raw[208:210])).ToFloat32()

	// Process two 128-element halves.
	for half := range 2 {
		qlOff := half * 64
		qhOff := half * 32
		scOff := half * 8
		outOff := half * 128

		for l := range 32 {
			is := l / 16 // sub-block offset within each group of 32

			q1 := int8((ql[qlOff+l]&0xF)|((qh[qhOff+l]&3)<<4)) - 32
			q2 := int8((ql[qlOff+32+l]&0xF)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
			q3 := int8((ql[qlOff+l]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
			q4 := int8((ql[qlOff+32+l]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32

			dst[outOff+l] = d * float32(int8(sc[scOff+is+0])) * float32(q1)
			dst[outOff+32+l] = d * float32(int8(sc[scOff+is+2])) * float32(q2)
			dst[outOff+64+l] = d * float32(int8(sc[scOff+is+4])) * float32(q3)
			dst[outOff+96+l] = d * float32(int8(sc[scOff+is+6])) * float32(q4)
		}
	}
}

// Q6KStorage holds Q6_K quantized tensor data on CPU.
type Q6KStorage struct {
	raw []byte
	len int
}

// NewQ6KStorageFromRaw creates Q6KStorage from raw super-block data.
func NewQ6KStorageFromRaw(raw []byte, numElements int) (*Q6KStorage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + q6KSuperBlockSize - 1) / q6KSuperBlockSize
	need := nBlocks * q6KBlockBytes
	if len(raw) < need {
		return nil, fmt.Errorf("Q6_K raw data too short: need %d bytes for %d blocks, got %d", need, nBlocks, len(raw))
	}
	data := make([]byte, need)
	copy(data, raw[:need])
	return &Q6KStorage{raw: data, len: numElements}, nil
}

func (q *Q6KStorage) Dequantize(dst []float32) {
	nBlocks := (q.len + q6KSuperBlockSize - 1) / q6KSuperBlockSize
	for bi := range nBlocks {
		blockRaw := q.raw[bi*q6KBlockBytes : (bi+1)*q6KBlockBytes]
		off := bi * q6KSuperBlockSize
		remaining := q.len - off
		if remaining >= q6KSuperBlockSize {
			DequantizeQ6K(blockRaw, dst[off:off+q6KSuperBlockSize])
		} else {
			var tmp [q6KSuperBlockSize]float32
			DequantizeQ6K(blockRaw, tmp[:])
			copy(dst[off:], tmp[:remaining])
		}
	}
}

func (q *Q6KStorage) Len() int               { return q.len }
func (q *Q6KStorage) Slice() []float32        { dst := make([]float32, q.len); q.Dequantize(dst); return dst }
func (q *Q6KStorage) Set(_ []float32)         { panic("Q6KStorage is immutable") }
func (q *Q6KStorage) DeviceType() device.Type { return device.CPU }

var _ Storage[float32] = (*Q6KStorage)(nil)

// Q5_K format: super-blocks of 256 values.
// Each super-block is 176 bytes:
//   - 2 bytes: fp16 d (super-block scale)
//   - 2 bytes: fp16 dmin (super-block min)
//   - 12 bytes: packed 6-bit scales and mins for 8 sub-blocks (same as Q4_K)
//   - 128 bytes: ql (low 4 bits of each 5-bit value)
//   - 32 bytes: qh (high 1 bit of each value, packed)
//
// Reference: llama.cpp ggml-quants.c dequantize_row_q5_K
const (
	q5KSuperBlockSize = 256
	q5KBlockBytes     = 176
)

// DequantizeQ5K dequantizes one Q5_K super-block (176 bytes) into 256 float32 values.
// Same split ordering as Q4_K, but each element has an extra high bit from qh.
// For each group of 64 elements (32 bytes of ql):
//   low nibbles + qh bit (2*group)   -> positions j..j+31
//   high nibbles + qh bit (2*group+1) -> positions j+32..j+63
// This matches llama.cpp's dequantize_row_q5_K.
func DequantizeQ5K(raw []byte, dst []float32) {
	d := float16.FromBits(binary.LittleEndian.Uint16(raw[0:2])).ToFloat32()
	dmin := float16.FromBits(binary.LittleEndian.Uint16(raw[2:4])).ToFloat32()

	scales, mins := decodeQ4KScalesMins(raw)

	ql := raw[16:144]  // 128 bytes: low 4 bits
	qh := raw[144:176] // 32 bytes: high 1 bit (256 bits)

	// u1 and u2 are bit masks that rotate through qh bits.
	u1 := uint8(1)
	u2 := uint8(2)
	for group := range 4 {
		sb0 := group * 2
		sb1 := group*2 + 1
		sc0 := d * float32(scales[sb0])
		mn0 := dmin * float32(mins[sb0])
		sc1 := d * float32(scales[sb1])
		mn1 := dmin * float32(mins[sb1])

		baseOut := group * 64
		baseQ := group * 32
		for l := range 32 {
			q := ql[baseQ+l]
			hb := qh[l]

			var h0, h1 uint8
			if hb&u1 != 0 {
				h0 = 16
			}
			if hb&u2 != 0 {
				h1 = 16
			}

			dst[baseOut+l] = sc0*float32((q&0xF)|h0) - mn0
			dst[baseOut+l+32] = sc1*float32((q>>4)|h1) - mn1
		}
		u1 <<= 2
		u2 <<= 2
	}
}

// Q5KStorage holds Q5_K quantized tensor data on CPU.
type Q5KStorage struct {
	raw []byte
	len int
}

// NewQ5KStorageFromRaw creates Q5KStorage from raw super-block data.
func NewQ5KStorageFromRaw(raw []byte, numElements int) (*Q5KStorage, error) {
	if numElements <= 0 {
		return nil, fmt.Errorf("numElements must be positive, got %d", numElements)
	}
	nBlocks := (numElements + q5KSuperBlockSize - 1) / q5KSuperBlockSize
	need := nBlocks * q5KBlockBytes
	if len(raw) < need {
		return nil, fmt.Errorf("Q5_K raw data too short: need %d bytes for %d blocks, got %d", need, nBlocks, len(raw))
	}
	data := make([]byte, need)
	copy(data, raw[:need])
	return &Q5KStorage{raw: data, len: numElements}, nil
}

func (q *Q5KStorage) Dequantize(dst []float32) {
	nBlocks := (q.len + q5KSuperBlockSize - 1) / q5KSuperBlockSize
	for bi := range nBlocks {
		blockRaw := q.raw[bi*q5KBlockBytes : (bi+1)*q5KBlockBytes]
		off := bi * q5KSuperBlockSize
		remaining := q.len - off
		if remaining >= q5KSuperBlockSize {
			DequantizeQ5K(blockRaw, dst[off:off+q5KSuperBlockSize])
		} else {
			var tmp [q5KSuperBlockSize]float32
			DequantizeQ5K(blockRaw, tmp[:])
			copy(dst[off:], tmp[:remaining])
		}
	}
}

func (q *Q5KStorage) Len() int               { return q.len }
func (q *Q5KStorage) Slice() []float32        { dst := make([]float32, q.len); q.Dequantize(dst); return dst }
func (q *Q5KStorage) Set(_ []float32)         { panic("Q5KStorage is immutable") }
func (q *Q5KStorage) DeviceType() device.Type { return device.CPU }

var _ Storage[float32] = (*Q5KStorage)(nil)
