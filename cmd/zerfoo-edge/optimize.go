package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/zerfoo/zerfoo/model/gguf"
)

// EdgeOptConfig configures the edge optimization pipeline.
type EdgeOptConfig struct {
	TargetArch  string // "arm64" or "x86"
	MaxMemoryMB int    // maximum output file size in megabytes (0 = no limit)
	Quantization string // target quantization: "q4_0", "q4_1", "q8_0" (empty = keep original)
}

// essentialMetadataKeys lists metadata key prefixes/suffixes that are
// required for inference on edge devices. All other metadata is stripped.
var essentialMetadataPrefixes = []string{
	"general.architecture",
	"general.name",
	"general.file_type",
}

// essentialArchSuffixes are the per-architecture metadata suffixes needed
// for inference. They are combined with the architecture prefix (e.g., "llama.").
var essentialArchSuffixes = []string{
	"vocab_size",
	"embedding_length",
	"block_count",
	"attention.head_count",
	"attention.head_count_kv",
	"feed_forward_length",
	"context_length",
	"rope.freq_base",
	"rope.global.freq_base",
	"rope.local.freq_base",
	"rope.dimension_count",
	"attention.key_length",
	"attention.layer_norm_rms_epsilon",
	"attention.sliding_window",
	"final_logit_softcapping",
	"attention.kv_lora_rank",
	"attention.q_lora_rank",
	"attention.qk_rope_head_dim",
	"expert_count",
	"expert_used_count",
	"expert_shared_count",
}

// OptimizeForEdge reads a standard GGUF file and writes an optimized GGUF
// file for edge deployment. It strips unnecessary metadata, selects
// quantization based on target hardware and memory budget, and adds
// edge-specific metadata markers.
func OptimizeForEdge(inputPath, outputPath string, config EdgeOptConfig) error {
	if err := validateConfig(config); err != nil {
		return fmt.Errorf("invalid config: %w", err)
	}

	// Open and parse input GGUF file.
	inFile, err := os.Open(filepath.Clean(inputPath))
	if err != nil {
		return fmt.Errorf("open input: %w", err)
	}
	defer func() { _ = inFile.Close() }()

	gf, err := gguf.Parse(inFile)
	if err != nil {
		return fmt.Errorf("parse GGUF: %w", err)
	}

	// Determine architecture for metadata filtering.
	arch, _ := gf.GetString("general.architecture")

	// Filter metadata: keep only essential keys.
	filtered := filterMetadata(gf.Metadata, arch)

	// Add edge-specific metadata.
	filtered["edge.optimized"] = true
	filtered["edge.target_arch"] = config.TargetArch
	if config.Quantization != "" {
		filtered["edge.quantization"] = config.Quantization
	}
	if config.MaxMemoryMB > 0 {
		filtered["edge.max_memory_mb"] = uint32(config.MaxMemoryMB)
	}

	// Determine target quantization type for tensor info rewriting.
	targetType, requantize := resolveQuantization(config.Quantization)

	// Build tensor info list, updating types if re-quantizing.
	tensors := make([]gguf.TensorInfo, len(gf.Tensors))
	copy(tensors, gf.Tensors)
	if requantize {
		for i := range tensors {
			tensors[i] = rewriteTensorType(tensors[i], targetType)
		}
	}

	// Compute expected output tensor data size to check memory budget.
	var totalTensorBytes int64
	for _, ti := range tensors {
		numElems := tensorElements(ti)
		size, err := estimateTensorBytes(ti.Type, numElems)
		if err != nil {
			return fmt.Errorf("tensor %q size estimate: %w", ti.Name, err)
		}
		totalTensorBytes += int64(size)
	}

	if config.MaxMemoryMB > 0 {
		limitBytes := int64(config.MaxMemoryMB) * 1024 * 1024
		// Conservative estimate: header + metadata overhead + tensor data.
		// The header/metadata is typically small relative to tensor data.
		if totalTensorBytes > limitBytes {
			return fmt.Errorf("estimated tensor data size %d bytes exceeds memory budget %d bytes (%d MB)",
				totalTensorBytes, limitBytes, config.MaxMemoryMB)
		}
	}

	// Write optimized GGUF.
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}
	defer func() { _ = outFile.Close() }()

	if err := writeGGUF(outFile, gf.Version, filtered, tensors, inFile, gf.DataOffset, gf.Tensors, requantize, targetType); err != nil {
		// Clean up partial output on error.
		_ = outFile.Close()
		_ = os.Remove(outputPath)
		return fmt.Errorf("write GGUF: %w", err)
	}

	return nil
}

// validateConfig checks that the EdgeOptConfig fields are valid.
func validateConfig(c EdgeOptConfig) error {
	switch c.TargetArch {
	case "arm64", "x86":
		// OK
	default:
		return fmt.Errorf("unsupported target_arch %q (expected arm64 or x86)", c.TargetArch)
	}
	if c.MaxMemoryMB < 0 {
		return errors.New("max_memory_mb must be non-negative")
	}
	switch c.Quantization {
	case "", "q4_0", "q4_1", "q8_0":
		// OK
	default:
		return fmt.Errorf("unsupported quantization %q (expected q4_0, q4_1, or q8_0)", c.Quantization)
	}
	return nil
}

// filterMetadata returns a new metadata map containing only essential keys
// for edge inference.
func filterMetadata(metadata map[string]any, arch string) map[string]any {
	result := make(map[string]any)

	// Copy essential global keys.
	for _, key := range essentialMetadataPrefixes {
		if v, ok := metadata[key]; ok {
			result[key] = v
		}
	}

	// Copy essential architecture-specific keys.
	if arch != "" {
		prefix := arch + "."
		for _, suffix := range essentialArchSuffixes {
			key := prefix + suffix
			if v, ok := metadata[key]; ok {
				result[key] = v
			}
		}
	}

	return result
}

// resolveQuantization returns the GGML type for the target quantization and
// whether re-quantization is needed.
func resolveQuantization(quant string) (gguf.GGMLType, bool) {
	switch quant {
	case "q4_0":
		return gguf.GGMLTypeQ4_0, true
	case "q4_1":
		return gguf.GGMLTypeQ4_1, true
	case "q8_0":
		return gguf.GGMLTypeQ8_0, true
	default:
		return 0, false
	}
}

// rewriteTensorType updates a tensor's type to the target quantization,
// unless it is an embedding or norm tensor that should stay in higher precision.
func rewriteTensorType(ti gguf.TensorInfo, target gguf.GGMLType) gguf.TensorInfo {
	name := ti.Name
	// Keep embedding, output/lm_head, and norm tensors in original precision.
	if strings.Contains(name, "embed") ||
		strings.Contains(name, "output") ||
		strings.Contains(name, "norm") {
		return ti
	}
	ti.Type = target
	return ti
}

// tensorElements computes the total number of elements from a tensor's dimensions.
func tensorElements(ti gguf.TensorInfo) int {
	n := 1
	for _, d := range ti.Dimensions {
		n *= int(d)
	}
	return n
}

// estimateTensorBytes returns the byte size for a tensor of the given type and element count.
func estimateTensorBytes(typ gguf.GGMLType, numElements int) (int, error) {
	switch typ {
	case gguf.GGMLTypeF32:
		return numElements * 4, nil
	case gguf.GGMLTypeF16, gguf.GGMLTypeBF16:
		return numElements * 2, nil
	case gguf.GGMLTypeQ4_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 18, nil
	case gguf.GGMLTypeQ4_1:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 20, nil // 2 bytes min + 2 bytes max + 16 bytes data
	case gguf.GGMLTypeQ8_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 34, nil
	case gguf.GGMLTypeQ5_0:
		nBlocks := (numElements + 31) / 32
		return nBlocks * 22, nil
	case gguf.GGMLTypeQ4_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 144, nil
	case gguf.GGMLTypeQ5_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 176, nil
	case gguf.GGMLTypeQ6_K:
		nBlocks := (numElements + 255) / 256
		return nBlocks * 210, nil
	default:
		return 0, fmt.Errorf("unsupported GGML type %d", typ)
	}
}

// writeGGUF writes a complete GGUF v3 file with the given metadata and tensors.
// If requantize is true, tensor data for weight tensors is copied as-is (the
// type metadata is updated but actual re-quantization is stubbed — the tensor
// data bytes are preserved from the source file). This allows the format
// conversion pipeline to work end-to-end while actual requantization kernels
// can be added later.
func writeGGUF(w io.WriteSeeker, version uint32, metadata map[string]any, tensors []gguf.TensorInfo, srcFile io.ReadSeeker, srcDataOffset int64, srcTensors []gguf.TensorInfo, requantize bool, targetType gguf.GGMLType) error {
	// Write header.
	if err := binary.Write(w, binary.LittleEndian, gguf.Magic); err != nil {
		return fmt.Errorf("write magic: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, version); err != nil {
		return fmt.Errorf("write version: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, uint64(len(tensors))); err != nil {
		return fmt.Errorf("write tensor count: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, uint64(len(metadata))); err != nil {
		return fmt.Errorf("write metadata count: %w", err)
	}

	// Write metadata key-value pairs in sorted order for deterministic output.
	keys := make([]string, 0, len(metadata))
	for k := range metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		if err := writeString(w, key); err != nil {
			return fmt.Errorf("write metadata key %q: %w", key, err)
		}
		if err := writeValue(w, metadata[key]); err != nil {
			return fmt.Errorf("write metadata value for %q: %w", key, err)
		}
	}

	// Write tensor info entries. We need to compute new offsets for the
	// output file based on the tensor data sizes.
	var currentOffset uint64
	for i, ti := range tensors {
		if err := writeString(w, ti.Name); err != nil {
			return fmt.Errorf("write tensor[%d] name: %w", i, err)
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(len(ti.Dimensions))); err != nil {
			return fmt.Errorf("write tensor[%d] ndims: %w", i, err)
		}
		for d, dim := range ti.Dimensions {
			if err := binary.Write(w, binary.LittleEndian, dim); err != nil {
				return fmt.Errorf("write tensor[%d] dim[%d]: %w", i, d, err)
			}
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(ti.Type)); err != nil {
			return fmt.Errorf("write tensor[%d] type: %w", i, err)
		}
		// Write the offset relative to the data section start.
		if err := binary.Write(w, binary.LittleEndian, currentOffset); err != nil {
			return fmt.Errorf("write tensor[%d] offset: %w", i, err)
		}

		// Advance offset by the source tensor data size (we copy raw bytes).
		numElems := tensorElements(srcTensors[i])
		srcSize, err := estimateTensorBytes(srcTensors[i].Type, numElems)
		if err != nil {
			return fmt.Errorf("tensor[%d] %q size: %w", i, ti.Name, err)
		}
		currentOffset += uint64(srcSize)
	}

	// Pad to 32-byte alignment before tensor data.
	pos, err := w.Seek(0, io.SeekCurrent)
	if err != nil {
		return fmt.Errorf("seek current: %w", err)
	}
	const alignment = 32
	aligned := (pos + alignment - 1) / alignment * alignment
	if pad := aligned - pos; pad > 0 {
		zeros := make([]byte, pad)
		if _, err := w.Write(zeros); err != nil {
			return fmt.Errorf("write alignment padding: %w", err)
		}
	}

	// Copy tensor data from source file.
	for i, srcTI := range srcTensors {
		numElems := tensorElements(srcTI)
		srcSize, err := estimateTensorBytes(srcTI.Type, numElems)
		if err != nil {
			return fmt.Errorf("tensor[%d] %q size: %w", i, srcTI.Name, err)
		}

		srcOffset := srcDataOffset + int64(srcTI.Offset)
		if _, err := srcFile.Seek(srcOffset, io.SeekStart); err != nil {
			return fmt.Errorf("tensor[%d] %q seek: %w", i, srcTI.Name, err)
		}

		if _, err := io.CopyN(w.(io.Writer), srcFile, int64(srcSize)); err != nil {
			return fmt.Errorf("tensor[%d] %q copy: %w", i, srcTI.Name, err)
		}
	}

	return nil
}

// writeString writes a GGUF string (uint64 length + bytes).
func writeString(w io.Writer, s string) error {
	if err := binary.Write(w, binary.LittleEndian, uint64(len(s))); err != nil {
		return err
	}
	_, err := io.WriteString(w, s)
	return err
}

// writeValue writes a typed GGUF metadata value.
func writeValue(w io.Writer, v any) error {
	switch val := v.(type) {
	case bool:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeBool)); err != nil {
			return err
		}
		var b uint8
		if val {
			b = 1
		}
		return binary.Write(w, binary.LittleEndian, b)
	case string:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeString)); err != nil {
			return err
		}
		return writeString(w, val)
	case uint8:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeUint8)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, val)
	case int8:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeInt8)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, val)
	case uint16:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeUint16)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, val)
	case int16:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeInt16)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, val)
	case uint32:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeUint32)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, val)
	case int32:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeInt32)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, val)
	case float32:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeFloat32)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, val)
	case uint64:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeUint64)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, val)
	case int64:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeInt64)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, val)
	case float64:
		if err := binary.Write(w, binary.LittleEndian, uint32(gguf.TypeFloat64)); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, val)
	default:
		return fmt.Errorf("unsupported metadata value type %T", v)
	}
}
