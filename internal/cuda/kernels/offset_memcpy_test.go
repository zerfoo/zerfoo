package kernels

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func TestOffsetMemcpy(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	const (
		dim       = 128
		maxSeqLen = 32
		pos       = 5
	)

	// Allocate dst buffer: maxSeqLen * dim floats, zeroed via H2D copy.
	dstZeros := make([]float32, maxSeqLen*dim)
	devDst := toDevice(t, dstZeros)
	defer func() { _ = cuda.Free(devDst) }()

	// Fill src with known values: src[i] = float32(i) + 1.0.
	src := make([]float32, dim)
	for i := range src {
		src[i] = float32(i) + 1.0
	}
	devSrc := toDevice(t, src)
	defer func() { _ = cuda.Free(devSrc) }()

	// Set counter to pos (5) via H2D copy.
	counter := int32(pos)
	devCounter, err := cuda.Malloc(4)
	if err != nil {
		t.Fatalf("Malloc counter: %v", err)
	}
	defer func() { _ = cuda.Free(devCounter) }()
	if err := cuda.Memcpy(devCounter, unsafe.Pointer(&counter), 4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy counter H2D: %v", err)
	}

	// Launch kernel.
	if err := OffsetMemcpy(devDst, devSrc, devCounter, dim, maxSeqLen, nil); err != nil {
		t.Fatalf("OffsetMemcpy: %v", err)
	}

	// Read back entire dst buffer.
	result := fromDevice(t, devDst, maxSeqLen*dim)

	// Verify data at offset pos*dim.
	for i := 0; i < dim; i++ {
		got := result[pos*dim+i]
		want := float32(i) + 1.0
		if got != want {
			t.Errorf("dst[%d*%d+%d] = %f, want %f", pos, dim, i, got, want)
		}
	}

	// Verify other positions are still zero.
	for row := 0; row < maxSeqLen; row++ {
		if row == pos {
			continue
		}
		for col := 0; col < dim; col++ {
			if v := result[row*dim+col]; v != 0 {
				t.Errorf("dst[%d*%d+%d] = %f, want 0 (untouched)", row, dim, col, v)
			}
		}
	}
}

func TestOffsetMemcpyBoundsCheck(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	const (
		dim       = 64
		maxSeqLen = 8
	)

	// Allocate dst buffer, zeroed via H2D copy.
	dstZeros := make([]float32, maxSeqLen*dim)
	devDst := toDevice(t, dstZeros)
	defer func() { _ = cuda.Free(devDst) }()

	// Fill src with ones.
	src := make([]float32, dim)
	for i := range src {
		src[i] = 1.0
	}
	devSrc := toDevice(t, src)
	defer func() { _ = cuda.Free(devSrc) }()

	// Set counter to maxSeqLen (out of bounds).
	counter := int32(maxSeqLen)
	devCounter, err := cuda.Malloc(4)
	if err != nil {
		t.Fatalf("Malloc counter: %v", err)
	}
	defer func() { _ = cuda.Free(devCounter) }()
	if err := cuda.Memcpy(devCounter, unsafe.Pointer(&counter), 4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy counter H2D: %v", err)
	}

	// Launch kernel -- should be a no-op due to bounds check.
	if err := OffsetMemcpy(devDst, devSrc, devCounter, dim, maxSeqLen, nil); err != nil {
		t.Fatalf("OffsetMemcpy: %v", err)
	}

	// Verify dst is all zeros (kernel should not have written anything).
	result := fromDevice(t, devDst, maxSeqLen*dim)
	for i, v := range result {
		if v != 0 {
			t.Errorf("dst[%d] = %f, want 0 (out-of-bounds write)", i, v)
		}
	}
}

func TestOffsetMemcpyGracefulWithoutCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("CUDA available, skipping graceful-failure test")
	}
	err := OffsetMemcpy(nil, nil, nil, 1, 1, nil)
	if err == nil {
		t.Error("OffsetMemcpy should return error without CUDA")
	}
}
