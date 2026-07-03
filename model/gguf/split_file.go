package gguf

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/zerfoo/ztensor/tensor"
)

// SplitFile represents a collection of parsed GGUF shards that together form
// a single model. Metadata comes from shard 0; tensors are distributed across
// all shards.
type SplitFile struct {
	// File is the merged view: metadata from shard 0, tensors from all shards.
	// DataOffset is not meaningful for split files — use ShardIndex instead.
	File *File

	// Shards holds the parsed header for each shard, indexed by shard number.
	Shards []*File

	// ShardPaths holds the file paths for each shard.
	ShardPaths []string

	// ShardIndex maps tensor name to the shard index that contains its data.
	ShardIndex map[string]int
}

// ParseSplit detects whether path is a split GGUF file and parses all shards.
// If path is a single (non-split) file, it returns nil, nil and the caller
// should fall back to Parse. Split files follow the naming convention:
//
//	Model-00001-of-00003.gguf
//	Model-00002-of-00003.gguf
//	Model-00003-of-00003.gguf
//
// The path may point to any shard; all sibling shards are discovered automatically.
func ParseSplit(path string) (*SplitFile, error) {
	paths, err := discoverShards(path)
	if err != nil {
		return nil, err
	}
	if paths == nil {
		return nil, nil // not a split file
	}

	shards := make([]*File, len(paths))
	shardIndex := make(map[string]int)
	var allTensors []TensorInfo

	for i, p := range paths {
		f, err := os.Open(filepath.Clean(p))
		if err != nil {
			return nil, fmt.Errorf("open shard %d (%s): %w", i, p, err)
		}
		gf, err := Parse(f)
		_ = f.Close()
		if err != nil {
			return nil, fmt.Errorf("parse shard %d (%s): %w", i, p, err)
		}
		shards[i] = gf

		for _, ti := range gf.Tensors {
			shardIndex[ti.Name] = i
			allTensors = append(allTensors, ti)
		}
	}

	// Build merged File using shard 0's metadata.
	merged := &File{
		Version:  shards[0].Version,
		Metadata: shards[0].Metadata,
		Tensors:  allTensors,
	}

	return &SplitFile{
		File:       merged,
		Shards:     shards,
		ShardPaths: paths,
		ShardIndex: shardIndex,
	}, nil
}

// discoverShards finds all shard paths for a split GGUF file.
// Returns nil if the path is not a split file (no "-NNNNN-of-NNNNN" pattern).
func discoverShards(path string) ([]string, error) {
	base := filepath.Base(path)

	// Look for the pattern "-NNNNN-of-NNNNN.gguf".
	idx := strings.LastIndex(base, "-of-")
	if idx < 0 {
		return nil, nil
	}

	suffix := base[idx+4:]
	if !strings.HasSuffix(suffix, ".gguf") {
		return nil, nil
	}

	// Find where the shard number starts (e.g., "-00001-of-").
	prefix := base[:idx]
	dashIdx := strings.LastIndex(prefix, "-")
	if dashIdx < 0 {
		return nil, nil
	}

	dir := filepath.Dir(path)
	modelPrefix := prefix[:dashIdx+1] // e.g., "Model-"
	totalSuffix := base[idx:]          // e.g., "-of-00003.gguf"

	// Parse total count from suffix.
	countStr := strings.TrimSuffix(suffix, ".gguf")
	var totalShards int
	if _, err := fmt.Sscanf(countStr, "%d", &totalShards); err != nil {
		return nil, nil
	}
	if totalShards < 2 || totalShards > 999 {
		return nil, nil
	}

	// Build paths for all shards and verify they exist.
	paths := make([]string, totalShards)
	for i := range totalShards {
		shardNum := fmt.Sprintf("%05d", i+1)
		name := modelPrefix + shardNum + totalSuffix
		p := filepath.Join(dir, name)
		if _, err := os.Stat(p); err != nil {
			return nil, fmt.Errorf("missing shard %d: %s", i+1, p)
		}
		paths[i] = p
	}

	return paths, nil
}

// LoadTensorsMmapSplit creates tensors backed by mmap'd regions from multiple
// shard files. Each shard is independently mmap'd. Tensor data references the
// correct shard's mapped region.
func LoadTensorsMmapSplit(sf *SplitFile, mappedShards [][]byte) (map[string]*tensor.TensorNumeric[float32], error) {
	result := make(map[string]*tensor.TensorNumeric[float32], len(sf.File.Tensors))

	for i := range sf.File.Tensors {
		ti := &sf.File.Tensors[i]
		shardIdx := sf.ShardIndex[ti.Name]
		shard := sf.Shards[shardIdx]
		mapped := mappedShards[shardIdx]

		numElements, err := computeNumElements(ti.Name, ti.Dimensions)
		if err != nil {
			return nil, err
		}

		shape := make([]int, len(ti.Dimensions))
		for j, d := range ti.Dimensions {
			shape[len(ti.Dimensions)-1-j] = int(d)
		}

		dataSize, err := TensorByteSize(ti.Type, int(numElements))
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}

		offset := shard.DataOffset + int64(ti.Offset)
		end := offset + int64(dataSize)
		if end > int64(len(mapped)) {
			return nil, fmt.Errorf("tensor %q: mmap region too small in shard %d (need offset %d + %d bytes, have %d)",
				ti.Name, shardIdx, offset, dataSize, len(mapped))
		}
		raw := mapped[offset:end]

		// Ternary tensors decoded eagerly (MmapStorage doesn't support TQ2_0).
		if ti.Type == GGMLTypeTQ2_0 {
			t, err := decodeTernaryTensor(shape, int(numElements), raw)
			if err != nil {
				return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
			}
			result[ti.Name] = t
			continue
		}

		qtype, err := mapGGMLType(ti.Type)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}

		s, err := tensor.NewMmapStorage(raw, int(numElements), qtype)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: mmap storage: %w", ti.Name, err)
		}

		t, err := tensor.NewWithStorage[float32](shape, s)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: create tensor: %w", ti.Name, err)
		}
		result[ti.Name] = t
	}

	return result, nil
}

// LoadTensorsSplit reads tensor data from all shards using heap allocation.
func LoadTensorsSplit(sf *SplitFile, readers []*os.File) (map[string]*tensor.TensorNumeric[float32], error) {
	result := make(map[string]*tensor.TensorNumeric[float32], len(sf.File.Tensors))

	for i := range sf.File.Tensors {
		ti := &sf.File.Tensors[i]
		shardIdx := sf.ShardIndex[ti.Name]
		shard := sf.Shards[shardIdx]
		r := readers[shardIdx]

		numElements, err := computeNumElements(ti.Name, ti.Dimensions)
		if err != nil {
			return nil, err
		}

		shape := make([]int, len(ti.Dimensions))
		for j, d := range ti.Dimensions {
			shape[len(ti.Dimensions)-1-j] = int(d)
		}

		dataSize, err := TensorByteSize(ti.Type, int(numElements))
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}

		offset := shard.DataOffset + int64(ti.Offset)
		if _, err := r.Seek(offset, 0); err != nil {
			return nil, fmt.Errorf("tensor %q: seek to offset %d in shard %d: %w", ti.Name, offset, shardIdx, err)
		}

		raw := make([]byte, dataSize)
		if _, err := readFull(r, raw); err != nil {
			return nil, fmt.Errorf("tensor %q: read %d bytes from shard %d: %w", ti.Name, dataSize, shardIdx, err)
		}

		t, err := decodeTensor(ti.Type, shape, int(numElements), raw)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}
		result[ti.Name] = t
	}

	return result, nil
}

func readFull(r *os.File, buf []byte) (int, error) {
	n := 0
	for n < len(buf) {
		nn, err := r.Read(buf[n:])
		n += nn
		if err != nil {
			return n, err
		}
	}
	return n, nil
}
