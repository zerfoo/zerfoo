package kernels

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// cpuGather performs a CPU gather: output[i, :] = table[indices[i], :].
func cpuGather(table []float32, D int, indices []int) []float32 {
	V := len(table) / D
	N := len(indices)
	out := make([]float32, N*D)
	for i, idx := range indices {
		if idx < 0 {
			idx = 0
		}
		if idx >= V {
			idx = V - 1
		}
		copy(out[i*D:(i+1)*D], table[idx*D:(idx+1)*D])
	}
	return out
}

func TestGatherInt64Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	tests := []struct {
		name    string
		V, D    int
		indices []int
	}{
		{
			name:    "single_index",
			V:       4, D: 3,
			indices: []int{2},
		},
		{
			name:    "multiple_indices",
			V:       5, D: 4,
			indices: []int{0, 3, 1, 4},
		},
		{
			name:    "duplicate_indices",
			V:       3, D: 2,
			indices: []int{1, 1, 0, 2, 0},
		},
		{
			name:    "large_embedding_dim",
			V:       2, D: 512,
			indices: []int{0, 1, 0},
		},
		{
			name:    "single_vocab",
			V:       1, D: 8,
			indices: []int{0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build embedding table with deterministic values.
			table := make([]float32, tt.V*tt.D)
			for i := range table {
				table[i] = float32(i) + 0.5
			}

			// CPU reference.
			expected := cpuGather(table, tt.D, tt.indices)

			// Upload table to GPU.
			devTable := toDevice(t, table)
			defer func() { _ = cuda.Free(devTable) }()

			// Upload indices as int64 (Go int = 8 bytes on 64-bit).
			N := len(tt.indices)
			intSize := int(unsafe.Sizeof(int(0)))
			idxBytes := N * intSize
			devIdx, err := cuda.Malloc(idxBytes)
			if err != nil {
				t.Fatalf("Malloc indices: %v", err)
			}
			defer func() { _ = cuda.Free(devIdx) }()

			if err := cuda.Memcpy(devIdx, unsafe.Pointer(&tt.indices[0]), idxBytes, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy indices H2D: %v", err)
			}

			// Allocate output.
			outN := N * tt.D
			devOut, err := cuda.Malloc(outN * 4)
			if err != nil {
				t.Fatalf("Malloc output: %v", err)
			}
			defer func() { _ = cuda.Free(devOut) }()

			// Launch kernel.
			if err := Gather(devTable, devIdx, devOut, N, tt.D, tt.V, nil); err != nil {
				t.Fatalf("Gather: %v", err)
			}

			// Read back and compare.
			result := fromDevice(t, devOut, outN)
			for i, want := range expected {
				if result[i] != want {
					t.Errorf("[%d] = %f, want %f", i, result[i], want)
				}
			}
		})
	}
}

func TestGatherI32Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	tests := []struct {
		name    string
		V, D    int
		indices []int32
	}{
		{
			name:    "single_index",
			V:       4, D: 3,
			indices: []int32{2},
		},
		{
			name:    "multiple_indices",
			V:       5, D: 4,
			indices: []int32{0, 3, 1, 4},
		},
		{
			name:    "duplicate_indices",
			V:       3, D: 2,
			indices: []int32{1, 1, 0, 2, 0},
		},
		{
			name:    "large_embedding_dim",
			V:       2, D: 512,
			indices: []int32{0, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build embedding table.
			table := make([]float32, tt.V*tt.D)
			for i := range table {
				table[i] = float32(i) + 0.5
			}

			// CPU reference (convert int32 indices to int for cpuGather).
			intIndices := make([]int, len(tt.indices))
			for i, v := range tt.indices {
				intIndices[i] = int(v)
			}
			expected := cpuGather(table, tt.D, intIndices)

			// Upload table.
			devTable := toDevice(t, table)
			defer func() { _ = cuda.Free(devTable) }()

			// Upload int32 indices.
			N := len(tt.indices)
			idxBytes := N * 4 // int32 = 4 bytes
			devIdx, err := cuda.Malloc(idxBytes)
			if err != nil {
				t.Fatalf("Malloc indices: %v", err)
			}
			defer func() { _ = cuda.Free(devIdx) }()

			if err := cuda.Memcpy(devIdx, unsafe.Pointer(&tt.indices[0]), idxBytes, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy indices H2D: %v", err)
			}

			// Allocate output.
			outN := N * tt.D
			devOut, err := cuda.Malloc(outN * 4)
			if err != nil {
				t.Fatalf("Malloc output: %v", err)
			}
			defer func() { _ = cuda.Free(devOut) }()

			// Launch int32 kernel.
			if err := GatherI32(devTable, devIdx, devOut, N, tt.D, tt.V, nil); err != nil {
				t.Fatalf("GatherI32: %v", err)
			}

			// Read back and compare (exact match for integer indexing).
			result := fromDevice(t, devOut, outN)
			for i, want := range expected {
				if result[i] != want {
					t.Errorf("[%d] = %f, want %f", i, result[i], want)
				}
			}
		})
	}
}

func TestGatherSignatureCompat(t *testing.T) {
	// Verify that Gather and GatherI32 have the expected signatures.
	type gatherFn = func(table, indices, output unsafe.Pointer, N, D, V int, s unsafe.Pointer) error
	_ = assignFunc[gatherFn](Gather)
	_ = assignFunc[gatherFn](GatherI32)
}

func TestGatherGracefulWithoutCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("CUDA available, skipping graceful-failure test")
	}

	tests := []struct {
		name string
		fn   func() error
	}{
		{"Gather", func() error { return Gather(nil, nil, nil, 1, 1, 1, nil) }},
		{"GatherI32", func() error { return GatherI32(nil, nil, nil, 1, 1, 1, nil) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.fn()
			if err == nil {
				t.Errorf("%s should return error without CUDA", tt.name)
			}
		})
	}
}
