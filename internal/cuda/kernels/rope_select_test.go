package kernels

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func TestRoPESelectParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	const (
		positions   = 32
		halfRotary  = 64
		targetPos   = 7
	)

	// Build cos/sin tables with deterministic values.
	cosTable := make([]float32, positions*halfRotary)
	sinTable := make([]float32, positions*halfRotary)
	for i := range cosTable {
		cosTable[i] = float32(i)*0.01 + 0.5
		sinTable[i] = float32(i)*0.01 + 1.5
	}

	// Expected output: table[targetPos*halfRotary : (targetPos+1)*halfRotary].
	expectedCos := cosTable[targetPos*halfRotary : (targetPos+1)*halfRotary]
	expectedSin := sinTable[targetPos*halfRotary : (targetPos+1)*halfRotary]

	// Upload tables to GPU.
	devCosTable := toDevice(t, cosTable)
	defer func() { _ = cuda.Free(devCosTable) }()
	devSinTable := toDevice(t, sinTable)
	defer func() { _ = cuda.Free(devSinTable) }()

	// Allocate output buffers.
	devCosOut, err := cuda.Malloc(halfRotary * 4)
	if err != nil {
		t.Fatalf("Malloc cosOut: %v", err)
	}
	defer func() { _ = cuda.Free(devCosOut) }()

	devSinOut, err := cuda.Malloc(halfRotary * 4)
	if err != nil {
		t.Fatalf("Malloc sinOut: %v", err)
	}
	defer func() { _ = cuda.Free(devSinOut) }()

	// Upload counter value (int32 = 4 bytes).
	counter := int32(targetPos)
	devCounter, err := cuda.Malloc(4)
	if err != nil {
		t.Fatalf("Malloc counter: %v", err)
	}
	defer func() { _ = cuda.Free(devCounter) }()

	if err := cuda.Memcpy(devCounter, unsafe.Pointer(&counter), 4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy counter H2D: %v", err)
	}

	// Launch kernel.
	if err := RoPESelect(devCosTable, devSinTable, devCosOut, devSinOut, devCounter, halfRotary, nil); err != nil {
		t.Fatalf("RoPESelect: %v", err)
	}

	// Read back and verify.
	gotCos := fromDevice(t, devCosOut, halfRotary)
	gotSin := fromDevice(t, devSinOut, halfRotary)

	for i := 0; i < halfRotary; i++ {
		if gotCos[i] != expectedCos[i] {
			t.Errorf("cos[%d] = %f, want %f", i, gotCos[i], expectedCos[i])
		}
		if gotSin[i] != expectedSin[i] {
			t.Errorf("sin[%d] = %f, want %f", i, gotSin[i], expectedSin[i])
		}
	}
}

func TestRoPESelectPosition0(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	const (
		positions  = 8
		halfRotary = 16
	)

	cosTable := make([]float32, positions*halfRotary)
	sinTable := make([]float32, positions*halfRotary)
	for i := range cosTable {
		cosTable[i] = float32(i) + 1.0
		sinTable[i] = float32(i) + 100.0
	}

	devCosTable := toDevice(t, cosTable)
	defer func() { _ = cuda.Free(devCosTable) }()
	devSinTable := toDevice(t, sinTable)
	defer func() { _ = cuda.Free(devSinTable) }()

	devCosOut, err := cuda.Malloc(halfRotary * 4)
	if err != nil {
		t.Fatalf("Malloc cosOut: %v", err)
	}
	defer func() { _ = cuda.Free(devCosOut) }()

	devSinOut, err := cuda.Malloc(halfRotary * 4)
	if err != nil {
		t.Fatalf("Malloc sinOut: %v", err)
	}
	defer func() { _ = cuda.Free(devSinOut) }()

	counter := int32(0)
	devCounter, err := cuda.Malloc(4)
	if err != nil {
		t.Fatalf("Malloc counter: %v", err)
	}
	defer func() { _ = cuda.Free(devCounter) }()

	if err := cuda.Memcpy(devCounter, unsafe.Pointer(&counter), 4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy counter H2D: %v", err)
	}

	if err := RoPESelect(devCosTable, devSinTable, devCosOut, devSinOut, devCounter, halfRotary, nil); err != nil {
		t.Fatalf("RoPESelect: %v", err)
	}

	gotCos := fromDevice(t, devCosOut, halfRotary)
	gotSin := fromDevice(t, devSinOut, halfRotary)

	for i := 0; i < halfRotary; i++ {
		if gotCos[i] != cosTable[i] {
			t.Errorf("cos[%d] = %f, want %f", i, gotCos[i], cosTable[i])
		}
		if gotSin[i] != sinTable[i] {
			t.Errorf("sin[%d] = %f, want %f", i, gotSin[i], sinTable[i])
		}
	}
}

func TestRoPESelectGracefulWithoutCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("CUDA available, skipping graceful-failure test")
	}

	err := RoPESelect(nil, nil, nil, nil, nil, 64, nil)
	if err == nil {
		t.Error("RoPESelect should return error without CUDA")
	}
}
