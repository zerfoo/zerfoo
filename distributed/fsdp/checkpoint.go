package fsdp

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"unsafe"

	"github.com/zerfoo/zerfoo/distributed"
	"github.com/zerfoo/zerfoo/model/gguf"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// SaveCheckpoint gathers all parameter shards via AllGather and writes them
// as a GGUF v3 checkpoint file. Only rank 0 writes the file; other ranks
// participate in AllGather but do not perform I/O.
func SaveCheckpoint[T tensor.Numeric](path string, module *ShardedModule[T], rank int) error {
	// AllGather all parameter shards to reconstruct full tensors.
	fullParams, err := gatherAllParams(module)
	if err != nil {
		return fmt.Errorf("fsdp checkpoint: allgather failed: %w", err)
	}

	// Only rank 0 writes the checkpoint file.
	if rank != 0 {
		return nil
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("fsdp checkpoint: create file: %w", err)
	}
	defer f.Close()

	if err := writeGGUF(f, fullParams); err != nil {
		return fmt.Errorf("fsdp checkpoint: write gguf: %w", err)
	}

	return nil
}

// LoadCheckpoint reads a GGUF checkpoint on rank 0 and distributes tensor
// shards to each rank's ShardedModule. Each rank receives its 1/worldSize
// slice of every parameter.
func LoadCheckpoint[T tensor.Numeric](path string, module *ShardedModule[T], rank int) error {
	// All ranks read the file (in a real distributed setting, rank 0 would
	// broadcast, but for simplicity and testing each rank reads directly).
	fullParams, err := readGGUFCheckpoint[T](path)
	if err != nil {
		return fmt.Errorf("fsdp checkpoint: read gguf: %w", err)
	}

	// Distribute shards: each rank takes its slice.
	params := module.module.Parameters()
	for _, p := range params {
		fullData, ok := fullParams[p.Name]
		if !ok {
			return fmt.Errorf("fsdp checkpoint: parameter %q not found in checkpoint", p.Name)
		}

		fullSize := len(fullData)
		shardSize := fullSize / module.worldSize
		start := module.rank * shardSize
		end := start + shardSize

		shard := make([]T, shardSize)
		copy(shard, fullData[start:end])

		module.shards[p.Name] = shard
		module.originalSizes[p.Name] = fullSize

		p.Value.SetData(shard)
		p.Value.SetShape([]int{shardSize})
	}

	return nil
}

// gatherAllParams reconstructs full parameter tensors from shards.
// Uses the same AllGather logic as ShardedModule.gatherParameters but
// returns the gathered data as a map without modifying the module state.
func gatherAllParams[T tensor.Numeric](module *ShardedModule[T]) (map[string][]T, error) {
	result := make(map[string][]T, len(module.shards))

	for name, shard := range module.shards {
		fullSize := module.originalSizes[name]
		fullData := make([]T, fullSize)

		if module.comm != nil {
			err := allGatherGeneric(
				module.comm,
				unsafe.Pointer(&shard[0]),
				unsafe.Pointer(&fullData[0]),
				len(shard),
			)
			if err != nil {
				return nil, err
			}
		} else {
			// In-process simulation: tile the shard for testing without NCCL.
			for i := 0; i < module.worldSize; i++ {
				copy(fullData[i*len(shard):], shard)
			}
		}

		result[name] = fullData
	}

	return result, nil
}

// allGatherGeneric dispatches to the appropriate NCCLAllGather variant
// based on the element size.
func allGatherGeneric(comm *distributed.NCCLComm, sendbuf, recvbuf unsafe.Pointer, sendcount int) error {
	return distributed.NCCLAllGather(comm, sendbuf, recvbuf, sendcount, 0)
}

// writeGGUF writes parameter tensors in GGUF v3 format using the shared
// ztensor/gguf writer.
func writeGGUF[T tensor.Numeric](w io.Writer, params map[string][]T) error {
	gw := ztensorgguf.NewWriter()

	gw.AddMetadataString("general.architecture", "checkpoint")

	// Sort parameter names for deterministic output.
	names := make([]string, 0, len(params))
	for name := range params {
		names = append(names, name)
	}
	sort.Strings(names)

	// Determine element size and GGML type.
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	ggmlType := ggmlTypeForElemSize(elemSize)

	// Add tensors as raw bytes with 1D (flat) shape.
	for _, name := range names {
		data := params[name]
		raw := sliceToBytes(data, elemSize)
		gw.AddTensor(name, int(ggmlType), []int{len(data)}, raw)
	}

	return gw.Write(w)
}

// sliceToBytes converts a typed slice to raw little-endian bytes.
func sliceToBytes[T tensor.Numeric](data []T, elemSize int) []byte {
	b := make([]byte, len(data)*elemSize)
	switch elemSize {
	case 4:
		for i, v := range data {
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(float32(v)))
		}
	case 8:
		for i, v := range data {
			binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(float64(v)))
		}
	}
	return b
}

// readGGUFCheckpoint reads a GGUF checkpoint file and returns full parameter
// data for each tensor.
func readGGUFCheckpoint[T tensor.Numeric](path string) (map[string][]T, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gf, err := gguf.Parse(f)
	if err != nil {
		return nil, fmt.Errorf("parse gguf: %w", err)
	}

	var zero T
	elemSize := int(unsafe.Sizeof(zero))

	result := make(map[string][]T, len(gf.Tensors))
	for _, ti := range gf.Tensors {
		// Compute element count from dimensions.
		count := 1
		for _, d := range ti.Dimensions {
			count *= int(d)
		}

		data := make([]T, count)

		// Seek to tensor data.
		offset := gf.DataOffset + int64(ti.Offset)
		if _, err := f.Seek(offset, io.SeekStart); err != nil {
			return nil, fmt.Errorf("seek tensor %q: %w", ti.Name, err)
		}

		// Read tensor data.
		raw := make([]byte, count*elemSize)
		if _, err := io.ReadFull(f, raw); err != nil {
			return nil, fmt.Errorf("read tensor %q: %w", ti.Name, err)
		}

		// Convert bytes to typed slice.
		for i := range data {
			switch elemSize {
			case 4:
				bits := binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])
				data[i] = *(*T)(unsafe.Pointer(&bits))
			case 8:
				bits := binary.LittleEndian.Uint64(raw[i*8 : (i+1)*8])
				data[i] = *(*T)(unsafe.Pointer(&bits))
			default:
				return nil, fmt.Errorf("unsupported element size %d", elemSize)
			}
		}

		result[ti.Name] = data
	}

	return result, nil
}

// ggmlTypeForElemSize returns the GGML type constant for the given element size.
func ggmlTypeForElemSize(elemSize int) gguf.GGMLType {
	switch elemSize {
	case 4:
		return gguf.GGMLTypeF32
	case 8:
		// float64 — no standard GGML type, use F32 as placeholder.
		// In practice, checkpoints use float32.
		return gguf.GGMLTypeF32
	default:
		return gguf.GGMLTypeF32
	}
}
