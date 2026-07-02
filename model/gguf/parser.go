// Package gguf implements a pure-Go parser for the GGUF v3 model format
// used by llama.cpp. It reads metadata key-value pairs and tensor descriptors
// from a GGUF file without loading tensor data into memory.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
)

// Magic is the GGUF file magic number ("GGUF" in little-endian).
const Magic uint32 = 0x46554747 // "GGUF" in little-endian

// GGUF metadata value types.
const (
	TypeUint8   uint32 = 0
	TypeInt8    uint32 = 1
	TypeUint16  uint32 = 2
	TypeInt16   uint32 = 3
	TypeUint32  uint32 = 4
	TypeInt32   uint32 = 5
	TypeFloat32 uint32 = 6
	TypeBool    uint32 = 7
	TypeString  uint32 = 8
	TypeArray   uint32 = 9
	TypeUint64  uint32 = 10
	TypeInt64   uint32 = 11
	TypeFloat64 uint32 = 12
)

// GGMLType identifies the quantization type of a tensor.
type GGMLType uint32

// Common GGML tensor types.
const (
	GGMLTypeF32  GGMLType = 0
	GGMLTypeF16  GGMLType = 1
	GGMLTypeQ4_0 GGMLType = 2
	GGMLTypeQ4_1 GGMLType = 3
	GGMLTypeQ5_0 GGMLType = 6
	GGMLTypeQ5_1 GGMLType = 7
	GGMLTypeQ8_0 GGMLType = 8
	GGMLTypeQ8_1 GGMLType = 9
	GGMLTypeQ2_K GGMLType = 10
	GGMLTypeQ3_K GGMLType = 11
	GGMLTypeQ4_K GGMLType = 12
	GGMLTypeQ5_K GGMLType = 13
	GGMLTypeQ6_K GGMLType = 14
	GGMLTypeQ8_K  GGMLType = 15
	GGMLTypeIQ2_XXS GGMLType = 16 // Importance-weighted 2-bit (E8 lattice codebook)
	GGMLTypeIQ3_S   GGMLType = 21 // Importance-weighted 3-bit with sub-block scales
	GGMLTypeIQ4_NL  GGMLType = 25 // Non-linear 4-bit with lookup table
	GGMLTypeBF16    GGMLType = 30
	GGMLTypeTQ2_0   GGMLType = 35 // Ternary 2-bit: 4 values per byte {-1, 0, 1}
)

// File represents a parsed GGUF file.
type File struct {
	Version    uint32
	Metadata   map[string]any
	Tensors    []TensorInfo
	DataOffset int64 // byte offset where tensor data begins
}

// TensorInfo describes a single tensor in the GGUF file.
type TensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       GGMLType
	Offset     uint64 // relative to DataOffset
}

// Parse reads a GGUF file header, metadata, and tensor info from r.
// It does not read tensor data. The returned File.DataOffset indicates
// where tensor data begins in the file.
func Parse(r io.ReadSeeker) (*File, error) {
	// Read and validate header.
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != Magic {
		return nil, fmt.Errorf("invalid GGUF magic: 0x%08X (expected 0x%08X)", magic, Magic)
	}

	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version %d (expected 2 or 3)", version)
	}

	var tensorCount, metadataKVCount uint64
	if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}
	if tensorCount > 100_000 {
		return nil, fmt.Errorf("tensor count %d exceeds maximum (100000)", tensorCount)
	}
	if err := binary.Read(r, binary.LittleEndian, &metadataKVCount); err != nil {
		return nil, fmt.Errorf("read metadata kv count: %w", err)
	}
	if metadataKVCount > 1_000_000 {
		return nil, fmt.Errorf("metadata kv count %d exceeds maximum (1000000)", metadataKVCount)
	}

	// Parse metadata key-value pairs.
	metadata := make(map[string]any, metadataKVCount)
	for i := range metadataKVCount {
		key, err := readString(r)
		if err != nil {
			return nil, fmt.Errorf("metadata[%d] key: %w", i, err)
		}
		val, err := readValue(r)
		if err != nil {
			return nil, fmt.Errorf("metadata[%d] %q value: %w", i, key, err)
		}
		metadata[key] = val
	}

	// Parse tensor info entries.
	tensors := make([]TensorInfo, tensorCount)
	for i := range tensorCount {
		name, err := readString(r)
		if err != nil {
			return nil, fmt.Errorf("tensor[%d] name: %w", i, err)
		}

		var numDims uint32
		if err := binary.Read(r, binary.LittleEndian, &numDims); err != nil {
			return nil, fmt.Errorf("tensor[%d] ndims: %w", i, err)
		}

		dims := make([]uint64, numDims)
		for d := range numDims {
			if err := binary.Read(r, binary.LittleEndian, &dims[d]); err != nil {
				return nil, fmt.Errorf("tensor[%d] dim[%d]: %w", i, d, err)
			}
		}

		var ggmlType uint32
		if err := binary.Read(r, binary.LittleEndian, &ggmlType); err != nil {
			return nil, fmt.Errorf("tensor[%d] type: %w", i, err)
		}

		var offset uint64
		if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
			return nil, fmt.Errorf("tensor[%d] offset: %w", i, err)
		}

		tensors[i] = TensorInfo{
			Name:       name,
			Dimensions: dims,
			Type:       GGMLType(ggmlType),
			Offset:     offset,
		}
	}

	// Data starts at the next alignment boundary after the header.
	pos, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("seek current: %w", err)
	}
	// GGUF aligns tensor data to 32 bytes.
	const alignment = 32
	dataOffset := (pos + alignment - 1) / alignment * alignment

	return &File{
		Version:    version,
		Metadata:   metadata,
		Tensors:    tensors,
		DataOffset: dataOffset,
	}, nil
}

// readString reads a GGUF string (uint64 length + bytes).
func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1<<20 { // sanity check: 1 MB max string
		return "", fmt.Errorf("string length %d exceeds maximum", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// readValue reads a typed GGUF metadata value.
func readValue(r io.Reader) (any, error) {
	var valueType uint32
	if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
		return nil, err
	}
	return readTypedValue(r, valueType)
}

// readTypedValue reads a value of the given type.
func readTypedValue(r io.Reader, valueType uint32) (any, error) {
	switch valueType {
	case TypeUint8:
		var v uint8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt8:
		var v int8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeUint16:
		var v uint16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt16:
		var v int16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeUint32:
		var v uint32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt32:
		var v int32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeFloat32:
		var v float32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case TypeString:
		return readString(r)
	case TypeUint64:
		var v uint64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt64:
		var v int64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeFloat64:
		var v float64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeArray:
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, fmt.Errorf("array element type: %w", err)
		}
		var length uint64
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return nil, fmt.Errorf("array length: %w", err)
		}
		if length > 1<<20 {
			return nil, fmt.Errorf("array length %d exceeds maximum", length)
		}
		arr := make([]any, length)
		for i := range length {
			v, err := readTypedValue(r, elemType)
			if err != nil {
				return nil, fmt.Errorf("array[%d]: %w", i, err)
			}
			arr[i] = v
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unknown metadata value type %d", valueType)
	}
}

// GetString returns a metadata string value.
func (f *File) GetString(key string) (string, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// GetUint32 returns a metadata value as uint32. Handles uint32 natively
// and also converts uint64, int32, and int64 values (common in HuggingFace
// GGUFs for Phi3 and Llama 3.1 which store model dimensions as uint64).
func (f *File) GetUint32(key string) (uint32, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case uint32:
		return n, true
	case uint64:
		return uint32(n), true
	case int32:
		return uint32(n), true
	case int64:
		return uint32(n), true
	default:
		return 0, false
	}
}

// GetFloat32 returns a metadata value as float32. Handles float32 natively
// and also converts float64 values (common in some GGUF converters).
func (f *File) GetFloat32(key string) (float32, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case float32:
		return n, true
	case float64:
		return float32(n), true
	default:
		return 0, false
	}
}
