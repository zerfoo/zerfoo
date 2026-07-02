//go:build arm64

package xblas

import "unsafe"

// q4DotBlockSIMD computes the dot product of one Q4 block (32 quantized values)
// with 32 float32 activation values using NEON. The Q4 block consists of 16
// packed bytes (two 4-bit values per byte) and a float32 scale factor.
//
// This fuses dequantization and dot product in NEON registers: nibbles are
// extracted, converted to float32, and immediately multiplied with activations
// via FMLA — no intermediate buffer is written to memory.
//
// Implemented in q4dot_arm64.s.
//
//go:noescape
func q4DotBlockSIMD(packed *byte, scale float32, x *float32) float32

// q4DotRowSIMD processes numBlocks Q4 blocks starting at blockPtr,
// computing the dot product against activation vector x.
// Each block is 18 bytes (2B float16 scale LE + 16B packed nibbles).
// Activations advance by 32 float32 per block.
//
// Implemented in q4dot_arm64.s.
//
//go:noescape
func q4DotRowSIMD(blockPtr unsafe.Pointer, x *float32, numBlocks int) float32

// q4DotBlock dispatches to the NEON implementation on arm64.
func q4DotBlock(packed *byte, scale float32, x *float32, n int) float32 {
	if n >= 32 {
		return q4DotBlockSIMD(packed, scale, x)
	}
	return q4DotBlockScalar(packed, scale, x, n)
}

// q4DotRow computes dot product of one Q4 row (numBlocks consecutive 18-byte blocks)
// against float32 activations. On arm64, this runs entirely in NEON assembly.
func q4DotRow(blockPtr unsafe.Pointer, x *float32, numBlocks int) float32 {
	return q4DotRowSIMD(blockPtr, x, numBlocks)
}
