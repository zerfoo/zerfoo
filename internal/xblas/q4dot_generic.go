//go:build !arm64

package xblas

import "unsafe"

// q4DotBlock on non-arm64 platforms uses the scalar fallback.
func q4DotBlock(packed *byte, scale float32, x *float32, n int) float32 {
	return q4DotBlockScalar(packed, scale, x, n)
}

// q4DotRow on non-arm64 platforms uses per-block scalar fallback.
func q4DotRow(blockPtr unsafe.Pointer, x *float32, numBlocks int) float32 {
	return q4DotRowScalar(blockPtr, x, numBlocks)
}
