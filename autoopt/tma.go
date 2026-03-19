package autoopt

import "fmt"

// TMADim specifies the dimensionality of a TMA descriptor.
type TMADim int

const (
	TMA2D TMADim = 2
	TMA3D TMADim = 3
)

// SwizzlePattern defines the memory swizzle mode for TMA loads.
type SwizzlePattern int

const (
	// SwizzleNone disables swizzling (linear access).
	SwizzleNone SwizzlePattern = iota

	// Swizzle32B applies 32-byte interleaving to reduce bank conflicts.
	Swizzle32B

	// Swizzle64B applies 64-byte interleaving.
	Swizzle64B

	// Swizzle128B applies 128-byte interleaving (best for large tiles).
	Swizzle128B
)

// String returns the swizzle pattern name.
func (s SwizzlePattern) String() string {
	switch s {
	case SwizzleNone:
		return "none"
	case Swizzle32B:
		return "32B"
	case Swizzle64B:
		return "64B"
	case Swizzle128B:
		return "128B"
	default:
		return "unknown"
	}
}

// TMAConfig describes a Tensor Memory Accelerator descriptor configuration
// for asynchronous bulk data movement between global and shared memory.
type TMAConfig struct {
	// Dim is the dimensionality of the TMA operation (2D or 3D).
	Dim TMADim

	// BoxDimX is the width of the TMA box in elements (innermost dimension).
	BoxDimX int

	// BoxDimY is the height of the TMA box in elements.
	BoxDimY int

	// BoxDimZ is the depth of the TMA box in elements (only used for TMA3D).
	BoxDimZ int

	// ElementSizeBytes is the size of each element in bytes (e.g. 2 for FP16, 4 for FP32).
	ElementSizeBytes int

	// Swizzle is the shared memory swizzle pattern.
	Swizzle SwizzlePattern

	// GlobalStride is the stride in bytes between rows in global memory.
	// Zero means the rows are contiguous (stride = BoxDimX * ElementSizeBytes).
	GlobalStride int64
}

// Validate checks that the TMA configuration is valid for SM90 hardware.
func (c *TMAConfig) Validate() error {
	if c.Dim != TMA2D && c.Dim != TMA3D {
		return fmt.Errorf("tma: unsupported dimension %d (must be 2 or 3)", c.Dim)
	}

	if c.ElementSizeBytes <= 0 || c.ElementSizeBytes > 16 {
		return fmt.Errorf("tma: element size %d bytes out of range [1, 16]", c.ElementSizeBytes)
	}

	if c.BoxDimX <= 0 || c.BoxDimX > 256 {
		return fmt.Errorf("tma: BoxDimX %d out of range [1, 256]", c.BoxDimX)
	}

	if c.BoxDimY <= 0 || c.BoxDimY > 256 {
		return fmt.Errorf("tma: BoxDimY %d out of range [1, 256]", c.BoxDimY)
	}

	if c.Dim == TMA3D && (c.BoxDimZ <= 0 || c.BoxDimZ > 256) {
		return fmt.Errorf("tma: BoxDimZ %d out of range [1, 256] for 3D TMA", c.BoxDimZ)
	}

	// TMA box size (in bytes) must not exceed shared memory transaction limit.
	boxBytes := int64(c.BoxDimX) * int64(c.BoxDimY) * int64(c.ElementSizeBytes)
	if c.Dim == TMA3D {
		boxBytes *= int64(c.BoxDimZ)
	}
	// SM90 supports up to 128 KiB per TMA transaction.
	const maxTMABytes = 128 * 1024
	if boxBytes > maxTMABytes {
		return fmt.Errorf("tma: box size %d bytes exceeds maximum %d bytes", boxBytes, maxTMABytes)
	}

	// Box dimensions must be aligned to element size.
	rowBytes := int64(c.BoxDimX) * int64(c.ElementSizeBytes)
	if rowBytes%16 != 0 {
		return fmt.Errorf("tma: row size %d bytes must be a multiple of 16", rowBytes)
	}

	return nil
}

// BoxBytes returns the total size of one TMA box in bytes.
func (c *TMAConfig) BoxBytes() int64 {
	b := int64(c.BoxDimX) * int64(c.BoxDimY) * int64(c.ElementSizeBytes)
	if c.Dim == TMA3D {
		b *= int64(c.BoxDimZ)
	}
	return b
}

// IsTMACompatible checks whether a tensor with the given dimensions and element
// size can be efficiently loaded via TMA. Returns true if the layout satisfies
// alignment and size constraints.
func IsTMACompatible(rows, cols, elementSizeBytes int) bool {
	if rows <= 0 || cols <= 0 || elementSizeBytes <= 0 {
		return false
	}
	// Row width must be a multiple of 16 bytes for TMA alignment.
	rowBytes := cols * elementSizeBytes
	if rowBytes%16 != 0 {
		return false
	}
	// Total must fit in a reasonable TMA transaction.
	totalBytes := int64(rows) * int64(cols) * int64(elementSizeBytes)
	const maxTMABytes = 128 * 1024
	return totalBytes <= maxTMABytes
}

// RecommendSwizzle selects the optimal swizzle pattern based on element size
// and tile width. Larger swizzle patterns reduce bank conflicts for wider tiles.
func RecommendSwizzle(elementSizeBytes, tileWidth int) SwizzlePattern {
	rowBytes := elementSizeBytes * tileWidth
	switch {
	case rowBytes >= 128:
		return Swizzle128B
	case rowBytes >= 64:
		return Swizzle64B
	case rowBytes >= 32:
		return Swizzle32B
	default:
		return SwizzleNone
	}
}
