package timeseries

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	ztensorgguf "github.com/zerfoo/ztensor/gguf"
)

// MoiraiConvertConfig holds configuration for converting a Moirai-2
// SafeTensors checkpoint to GGUF format.
type MoiraiConvertConfig struct {
	// NumLayers is the number of transformer encoder layers.
	NumLayers int
	// HiddenDim is the hidden dimension of the model.
	HiddenDim int
	// NumHeads is the number of attention heads.
	NumHeads int
	// NumFreqEmbeddings is the number of frequency embeddings for patching.
	NumFreqEmbeddings int
	// ModelName is an optional human-readable name stored in general.name.
	ModelName string
}

// Validate checks that the convert config is well-formed.
func (c *MoiraiConvertConfig) Validate() error {
	if c.NumLayers <= 0 {
		return fmt.Errorf("moirai: NumLayers must be positive, got %d", c.NumLayers)
	}
	if c.HiddenDim <= 0 {
		return fmt.Errorf("moirai: HiddenDim must be positive, got %d", c.HiddenDim)
	}
	if c.NumHeads <= 0 {
		return fmt.Errorf("moirai: NumHeads must be positive, got %d", c.NumHeads)
	}
	if c.NumFreqEmbeddings <= 0 {
		return fmt.Errorf("moirai: NumFreqEmbeddings must be positive, got %d", c.NumFreqEmbeddings)
	}
	return nil
}

// MapMoiraiTensorName converts a HuggingFace SafeTensors tensor name from the
// Salesforce/moirai-2-* checkpoint to the GGUF convention used by the Moirai
// graph builder.
//
// HuggingFace names follow the pattern:
//
//	model.encoder.layers.{N}.self_attn.{param}  →  moirai.enc.layer.{N}.self_attn.{param}
//	model.encoder.layers.{N}.mlp.{param}        →  moirai.enc.layer.{N}.mlp.{param}
//	model.encoder.layers.{N}.norm1.{param}       →  moirai.enc.layer.{N}.norm1.{param}
//	model.encoder.layers.{N}.norm2.{param}       →  moirai.enc.layer.{N}.norm2.{param}
//	model.{param}                                →  moirai.{param}
func MapMoiraiTensorName(hfName string) string {
	// Encoder layer tensors: model.encoder.layers.{N}.{rest}
	if strings.HasPrefix(hfName, "model.encoder.layers.") {
		rest := strings.TrimPrefix(hfName, "model.encoder.layers.")
		// rest is "{N}.{rest}" — split on first dot to get layer number.
		dotIdx := strings.Index(rest, ".")
		if dotIdx < 0 {
			return "moirai.enc.layer." + rest
		}
		layerNum := rest[:dotIdx]
		layerRest := rest[dotIdx+1:]
		return "moirai.enc.layer." + layerNum + "." + layerRest
	}
	// Other model-prefixed tensors: model.{rest} → moirai.{rest}
	if strings.HasPrefix(hfName, "model.") {
		rest := strings.TrimPrefix(hfName, "model.")
		return "moirai." + rest
	}
	// Fallback: add moirai. prefix.
	return "moirai." + hfName
}

// ConvertMoiraiToGGUF reads a Moirai-2 SafeTensors checkpoint and writes a
// GGUF file with architecture metadata and converted tensor names.
func ConvertMoiraiToGGUF(safeTensorsPath, ggufPath string, cfg MoiraiConvertConfig) error {
	if err := cfg.Validate(); err != nil {
		return err
	}

	sf, err := os.Open(filepath.Clean(safeTensorsPath))
	if err != nil {
		return fmt.Errorf("moirai: open safetensors: %w", err)
	}
	defer func() { _ = sf.Close() }()

	header, dataOffset, err := parseSafeTensorsHeader(sf)
	if err != nil {
		return fmt.Errorf("moirai: %w", err)
	}

	gw := ztensorgguf.NewWriter()

	// Architecture metadata.
	gw.AddMetadataString("general.architecture", "moirai")
	if cfg.ModelName != "" {
		gw.AddMetadataString("general.name", cfg.ModelName)
	}
	gw.AddMetadataUint32("moirai.num_layers", uint32(cfg.NumLayers))
	gw.AddMetadataUint32("moirai.hidden_dim", uint32(cfg.HiddenDim))
	gw.AddMetadataUint32("moirai.num_heads", uint32(cfg.NumHeads))
	gw.AddMetadataUint32("moirai.num_freq_embeddings", uint32(cfg.NumFreqEmbeddings))

	// Sort tensor names for deterministic output.
	names := make([]string, 0, len(header))
	for name := range header {
		names = append(names, name)
	}
	sort.Strings(names)

	// Read and convert each tensor.
	for _, hfName := range names {
		meta := header[hfName]

		ggufName := MapMoiraiTensorName(hfName)

		ggufType, elemSize, err := safeTensorsDTypeToGGUF(meta.DType)
		if err != nil {
			return fmt.Errorf("moirai: tensor %q: %w", hfName, err)
		}

		dataLen := meta.DataOffsets[1] - meta.DataOffsets[0]
		if dataLen < 0 {
			return fmt.Errorf("moirai: tensor %q has invalid data offsets [%d, %d]", hfName, meta.DataOffsets[0], meta.DataOffsets[1])
		}

		// Read raw tensor data from the SafeTensors file.
		if _, err := sf.Seek(dataOffset+meta.DataOffsets[0], io.SeekStart); err != nil {
			return fmt.Errorf("moirai: seek to tensor %q data: %w", hfName, err)
		}
		data := make([]byte, dataLen)
		if _, err := io.ReadFull(sf, data); err != nil {
			return fmt.Errorf("moirai: read tensor %q data: %w", hfName, err)
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
			return fmt.Errorf("moirai: tensor %q shape %v expects %d bytes but got %d",
				hfName, meta.Shape, int64(expectedElems)*int64(elemSize), len(data))
		}

		gw.AddTensor(ggufName, ggufType, meta.Shape, data)
	}

	out, err := os.Create(filepath.Clean(ggufPath))
	if err != nil {
		return fmt.Errorf("moirai: create GGUF file: %w", err)
	}
	defer func() { _ = out.Close() }()

	if err := gw.Write(out); err != nil {
		return fmt.Errorf("moirai: write GGUF: %w", err)
	}

	return nil
}
