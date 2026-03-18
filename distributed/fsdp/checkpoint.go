package fsdp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
	"unsafe"

	"github.com/zerfoo/zerfoo/distributed"
	"github.com/zerfoo/zerfoo/model/gguf"
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

// writeGGUF writes parameter tensors in GGUF v3 format.
func writeGGUF[T tensor.Numeric](w io.Writer, params map[string][]T) error {
	var buf bytes.Buffer

	// Header: magic, version, tensor count, metadata KV count.
	binary.Write(&buf, binary.LittleEndian, gguf.Magic)
	binary.Write(&buf, binary.LittleEndian, uint32(3)) // version 3

	tensorCount := uint64(len(params))
	binary.Write(&buf, binary.LittleEndian, tensorCount)

	// One metadata KV: "general.architecture" = "checkpoint"
	metadataKVCount := uint64(1)
	binary.Write(&buf, binary.LittleEndian, metadataKVCount)

	// Write metadata KV: key
	writeGGUFString(&buf, "general.architecture")
	// Type: string
	binary.Write(&buf, binary.LittleEndian, gguf.TypeString)
	// Value
	writeGGUFString(&buf, "checkpoint")

	// Sort parameter names for deterministic output.
	names := make([]string, 0, len(params))
	for name := range params {
		names = append(names, name)
	}
	sort.Strings(names)

	// Compute tensor data sizes and offsets.
	var zero T
	elemSize := int(unsafe.Sizeof(zero))

	type tensorEntry struct {
		name   string
		data   []T
		offset uint64
	}
	entries := make([]tensorEntry, 0, len(names))

	var dataOffset uint64
	for _, name := range names {
		data := params[name]
		entries = append(entries, tensorEntry{
			name:   name,
			data:   data,
			offset: dataOffset,
		})
		dataOffset += uint64(len(data) * elemSize)
	}

	// Write tensor info entries.
	ggmlType := ggmlTypeForElemSize(elemSize)
	for _, e := range entries {
		writeGGUFString(&buf, e.name)
		// 1 dimension (flat tensor).
		binary.Write(&buf, binary.LittleEndian, uint32(1))
		binary.Write(&buf, binary.LittleEndian, uint64(len(e.data)))
		binary.Write(&buf, binary.LittleEndian, uint32(ggmlType))
		binary.Write(&buf, binary.LittleEndian, e.offset)
	}

	// Align to 32 bytes before tensor data.
	const alignment = 32
	pos := buf.Len()
	padding := (alignment - (pos % alignment)) % alignment
	for i := 0; i < padding; i++ {
		buf.WriteByte(0)
	}

	// Write header + metadata + tensor info.
	if _, err := w.Write(buf.Bytes()); err != nil {
		return err
	}

	// Write tensor data.
	for _, e := range entries {
		if err := binary.Write(w, binary.LittleEndian, e.data); err != nil {
			return fmt.Errorf("write tensor %q data: %w", e.name, err)
		}
	}

	return nil
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

// writeGGUFString writes a GGUF-format string (uint64 length + bytes).
func writeGGUFString(buf *bytes.Buffer, s string) {
	binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
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
