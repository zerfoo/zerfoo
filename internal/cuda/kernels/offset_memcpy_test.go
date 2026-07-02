package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/float16"
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

func TestOffsetMemcpyFP16(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	const (
		dim       = 128
		maxSeqLen = 32
		pos       = 3
	)

	// Allocate FP16 dst buffer: maxSeqLen * dim * 2 bytes, zeroed.
	dstBytes := maxSeqLen * dim * 2
	devDst, err := cuda.Malloc(dstBytes)
	if err != nil {
		t.Fatalf("Malloc dst: %v", err)
	}
	defer func() { _ = cuda.Free(devDst) }()
	zeros := make([]byte, dstBytes)
	if err := cuda.Memcpy(devDst, unsafe.Pointer(&zeros[0]), dstBytes, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy zeros H2D: %v", err)
	}

	// Fill F32 src with known values: src[i] = float32(i) + 1.0.
	src := make([]float32, dim)
	for i := range src {
		src[i] = float32(i) + 1.0
	}
	devSrc := toDevice(t, src)
	defer func() { _ = cuda.Free(devSrc) }()

	// Set counter to pos (3).
	counter := int32(pos)
	devCounter, err := cuda.Malloc(4)
	if err != nil {
		t.Fatalf("Malloc counter: %v", err)
	}
	defer func() { _ = cuda.Free(devCounter) }()
	if err := cuda.Memcpy(devCounter, unsafe.Pointer(&counter), 4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy counter H2D: %v", err)
	}

	// Launch FP16 kernel.
	if err := OffsetMemcpyFP16(devDst, devSrc, devCounter, dim, maxSeqLen, nil); err != nil {
		t.Fatalf("OffsetMemcpyFP16: %v", err)
	}

	// Read back entire FP16 dst buffer as uint16 values.
	result := make([]uint16, maxSeqLen*dim)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devDst, dstBytes, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	// Verify data at offset pos*dim: each FP16 value should match F32 src.
	for i := 0; i < dim; i++ {
		got := float16.Float16(result[pos*dim+i]).ToFloat32()
		want := float32(i) + 1.0
		if math.Abs(float64(got-want)) > 0.5 {
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
				t.Errorf("dst[%d*%d+%d] = 0x%04x, want 0 (untouched)", row, dim, col, v)
			}
		}
	}
}

func TestOffsetMemcpyFP16GracefulWithoutCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("CUDA available, skipping graceful-failure test")
	}
	err := OffsetMemcpyFP16(nil, nil, nil, 1, 1, nil)
	if err == nil {
		t.Error("OffsetMemcpyFP16 should return error without CUDA")
	}
}
