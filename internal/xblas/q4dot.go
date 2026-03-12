package xblas

import "unsafe"

// q4DotBlockScalar is the pure-Go scalar implementation.
// It computes the dot product of one Q4 block (32 quantized values) with
// n float32 activation values. The Q4 block consists of 16 packed bytes
// (two 4-bit values per byte) and a float32 scale factor.
// GGML Q4_0 split format: low nibbles → positions 0-15, high nibbles → 16-31.
func q4DotBlockScalar(packed *byte, scale float32, x *float32, n int) float32 {
	if n > 32 {
		n = 32
	}
	data := unsafe.Slice(packed, 16)
	xSlice := unsafe.Slice(x, n)

	var sum float32
	for p := range 16 {
		if p >= n {
			break
		}
		byteVal := data[p]
		lo := float32(int(byteVal&0x0F) - 8)
		hi := float32(int(byteVal>>4) - 8)
		sum += lo * xSlice[p]
		if p+16 < n {
			sum += hi * xSlice[p+16]
		}
	}
	return sum * scale
}

// q4DotRowScalar computes the dot product of numBlocks consecutive Q4 blocks
// against float32 activations using scalar code. Each block is 18 bytes:
// 2 bytes float16 scale (LE) + 16 bytes packed nibbles.
func q4DotRowScalar(blockPtr unsafe.Pointer, x *float32, numBlocks int) float32 {
	base := (*byte)(blockPtr)
	xSlice := unsafe.Slice(x, numBlocks*32)
	var total float32
	for bi := range numBlocks {
		// Read float16 scale (2 bytes LE) and convert to float32.
		scalePtr := unsafe.Add(unsafe.Pointer(base), bi*18)
		scaleBits := uint16(*(*byte)(scalePtr)) | uint16(*(*byte)(unsafe.Add(scalePtr, 1)))<<8
		scale := float16BitsToFloat32(scaleBits)

		dataPtr := (*byte)(unsafe.Add(unsafe.Pointer(base), bi*18+2))
		total += q4DotBlockScalar(dataPtr, scale, &xSlice[bi*32], 32)
	}
	return total
}

// float16BitsToFloat32 converts IEEE 754 half-precision bits to float32.
func float16BitsToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) & 1
	exp := uint32(bits>>10) & 0x1F
	frac := uint32(bits) & 0x3FF

	switch exp {
	case 0:
		if frac == 0 {
			return 0
		}
		// Subnormal: 2^(-14) * (frac/1024)
		f := float32(frac) / 1024.0
		f *= 1.0 / 16384.0 // 2^(-14)
		if sign == 1 {
			f = -f
		}
		return f
	case 31:
		return 0 // Inf/NaN → treat as 0 for quantization
	default:
		// Normal: (-1)^sign * 2^(exp-15) * (1 + frac/1024)
		e := int(exp) - 15 + 127
		bits32 := (sign << 31) | (uint32(e) << 23) | (frac << 13)
		return *(*float32)(unsafe.Pointer(&bits32))
	}
}
