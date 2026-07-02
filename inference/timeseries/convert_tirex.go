package timeseries

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	ztensorgguf "github.com/zerfoo/ztensor/gguf"
)

// TiRexConvertConfig holds configuration for converting a TiRex SafeTensors
// checkpoint to GGUF format.
type TiRexConvertConfig struct {
	// NumLayers is the number of xLSTM blocks in the model.
	NumLayers int
	// HiddenDim is the hidden dimension of the model.
	HiddenDim int
	// BlockTypes specifies the type of each block: "slstm" or "mlstm".
	// Length must equal NumLayers.
	BlockTypes []string
	// ModelName is an optional human-readable name stored in general.name.
	ModelName string
}

// Validate checks that the convert config is well-formed.
func (c *TiRexConvertConfig) Validate() error {
	if c.NumLayers <= 0 {
		return fmt.Errorf("tirex: NumLayers must be positive, got %d", c.NumLayers)
	}
	if c.HiddenDim <= 0 {
		return fmt.Errorf("tirex: HiddenDim must be positive, got %d", c.HiddenDim)
	}
	if len(c.BlockTypes) != c.NumLayers {
		return fmt.Errorf("tirex: BlockTypes length %d does not match NumLayers %d", len(c.BlockTypes), c.NumLayers)
	}
	for i, bt := range c.BlockTypes {
		if bt != "slstm" && bt != "mlstm" {
			return fmt.Errorf("tirex: BlockTypes[%d] must be \"slstm\" or \"mlstm\", got %q", i, bt)
		}
	}
	return nil
}

// MapTiRexTensorName converts a HuggingFace SafeTensors tensor name from the
// NX-AI/TiRex checkpoint to the GGUF convention used by the TiRex graph builder.
//
// HuggingFace names follow the pattern:
//
//	blocks.{layer}.{block_type}.{param}
//
// For example:
//
//	blocks.0.slstm.weight_ih  →  tirex.block.0.slstm.weight_ih
//	blocks.2.mlstm.weight_q   →  tirex.block.2.mlstm.weight_q
//
// Global tensors (not prefixed with "blocks.") are mapped with a "tirex." prefix:
//
//	input_proj.weight     →  tirex.input_proj.weight
//	output_head.weight    →  tirex.output_head.weight
//	norm.weight           →  tirex.norm.weight
func MapTiRexTensorName(hfName string) string {
	// Block-level tensors: blocks.{N}.{rest} → tirex.block.{N}.{rest}
	if strings.HasPrefix(hfName, "blocks.") {
		rest := strings.TrimPrefix(hfName, "blocks.")
		return "tirex.block." + rest
	}
	// Global tensors get "tirex." prefix.
	return "tirex." + hfName
}

// safeTensorsHeader is the JSON structure at the start of a SafeTensors file.
// Each key is a tensor name, and the value describes dtype, shape, and byte offsets.
type safeTensorsHeader map[string]safeTensorsMeta

// safeTensorsMeta describes a single tensor in the SafeTensors format.
type safeTensorsMeta struct {
	DType       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// parseSafeTensorsHeader reads the SafeTensors file header from r and returns
// the tensor metadata map and the byte offset where tensor data begins.
func parseSafeTensorsHeader(r io.ReadSeeker) (safeTensorsHeader, int64, error) {
	// First 8 bytes: little-endian uint64 header length.
	var headerLen uint64
	if err := binary.Read(r, binary.LittleEndian, &headerLen); err != nil {
		return nil, 0, fmt.Errorf("read header length: %w", err)
	}

	// Sanity check: headers > 100MB are likely corrupt.
	const maxHeaderLen = 100 * 1024 * 1024
	if headerLen > maxHeaderLen {
		return nil, 0, fmt.Errorf("header length %d exceeds maximum %d", headerLen, maxHeaderLen)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, 0, fmt.Errorf("read header JSON: %w", err)
	}

	// Parse JSON header. The "__metadata__" key is a special entry that stores
	// file-level metadata (not a tensor), so we parse into a raw map first.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, 0, fmt.Errorf("parse header JSON: %w", err)
	}

	header := make(safeTensorsHeader, len(raw))
	for key, val := range raw {
		if key == "__metadata__" {
			continue
		}
		var meta safeTensorsMeta
		if err := json.Unmarshal(val, &meta); err != nil {
			return nil, 0, fmt.Errorf("parse tensor %q metadata: %w", key, err)
		}
		header[key] = meta
	}

	dataOffset := int64(8 + headerLen)
	return header, dataOffset, nil
}

// ConvertTiRexToGGUF reads a TiRex SafeTensors checkpoint and writes a GGUF
// file with architecture metadata and converted tensor names.
func ConvertTiRexToGGUF(safeTensorsPath, ggufPath string, cfg TiRexConvertConfig) error {
	if err := cfg.Validate(); err != nil {
		return err
	}

	sf, err := os.Open(filepath.Clean(safeTensorsPath))
	if err != nil {
		return fmt.Errorf("tirex: open safetensors: %w", err)
	}
	defer func() { _ = sf.Close() }()

	header, dataOffset, err := parseSafeTensorsHeader(sf)
	if err != nil {
		return fmt.Errorf("tirex: %w", err)
	}

	gw := ztensorgguf.NewWriter()

	// Architecture metadata.
	gw.AddMetadataString("general.architecture", "tirex")
	if cfg.ModelName != "" {
		gw.AddMetadataString("general.name", cfg.ModelName)
	}
	gw.AddMetadataUint32("tirex.block_count", uint32(cfg.NumLayers))
	gw.AddMetadataUint32("tirex.hidden_dim", uint32(cfg.HiddenDim))
	gw.AddMetadataStringArray("tirex.block_types", cfg.BlockTypes)

	// Sort tensor names for deterministic output.
	names := make([]string, 0, len(header))
	for name := range header {
		names = append(names, name)
	}
	sort.Strings(names)

	// Read and convert each tensor.
	for _, hfName := range names {
		meta := header[hfName]

		ggufName := MapTiRexTensorName(hfName)

		ggufType, elemSize, err := safeTensorsDTypeToGGUF(meta.DType)
		if err != nil {
			return fmt.Errorf("tirex: tensor %q: %w", hfName, err)
		}

		dataLen := meta.DataOffsets[1] - meta.DataOffsets[0]
		if dataLen < 0 {
			return fmt.Errorf("tirex: tensor %q has invalid data offsets [%d, %d]", hfName, meta.DataOffsets[0], meta.DataOffsets[1])
		}

		// Read raw tensor data from the SafeTensors file.
		if _, err := sf.Seek(dataOffset+meta.DataOffsets[0], io.SeekStart); err != nil {
			return fmt.Errorf("tirex: seek to tensor %q data: %w", hfName, err)
		}
		data := make([]byte, dataLen)
		if _, err := io.ReadFull(sf, data); err != nil {
			return fmt.Errorf("tirex: read tensor %q data: %w", hfName, err)
		}

		// For float64 (F64) tensors, convert to float32 since GGUF does not
		// have an F64 type.
		if meta.DType == "F64" {
			data = convertF64ToF32Bytes(data)
			ggufType = ztensorgguf.TypeF32
			elemSize = 4
		}

		// Validate data length against shape.
		expectedElems := 1
		for _, d := range meta.Shape {
			expectedElems *= d
		}
		if int64(expectedElems)*int64(elemSize) != int64(len(data)) {
			return fmt.Errorf("tirex: tensor %q shape %v expects %d bytes but got %d",
				hfName, meta.Shape, int64(expectedElems)*int64(elemSize), len(data))
		}

		gw.AddTensor(ggufName, ggufType, meta.Shape, data)
	}

	out, err := os.Create(filepath.Clean(ggufPath))
	if err != nil {
		return fmt.Errorf("tirex: create GGUF file: %w", err)
	}
	defer func() { _ = out.Close() }()

	if err := gw.Write(out); err != nil {
		return fmt.Errorf("tirex: write GGUF: %w", err)
	}

	return nil
}

// safeTensorsDTypeToGGUF maps SafeTensors dtype strings to GGUF type constants
// and returns the element size in bytes.
func safeTensorsDTypeToGGUF(dtype string) (int, int, error) {
	switch dtype {
	case "F32":
		return ztensorgguf.TypeF32, 4, nil
	case "F16":
		return ztensorgguf.TypeF16, 2, nil
	case "BF16":
		return ztensorgguf.TypeBF16, 2, nil
	case "F64":
		// F64 will be converted to F32 by the caller.
		return ztensorgguf.TypeF32, 8, nil
	default:
		return 0, 0, fmt.Errorf("unsupported SafeTensors dtype %q", dtype)
	}
}

// convertF64ToF32Bytes converts raw float64 little-endian bytes to float32.
func convertF64ToF32Bytes(data []byte) []byte {
	n := len(data) / 8
	out := make([]byte, n*4)
	for i := range n {
		bits := binary.LittleEndian.Uint64(data[i*8:])
		f64 := math.Float64frombits(bits)
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(float32(f64)))
	}
	return out
}
