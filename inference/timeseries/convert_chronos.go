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

// ChronosConvertConfig holds configuration for converting a Chronos-2
// (T5 encoder-decoder) SafeTensors checkpoint to GGUF format.
type ChronosConvertConfig struct {
	// NumEncoderLayers is the number of encoder blocks.
	NumEncoderLayers int
	// NumDecoderLayers is the number of decoder blocks.
	NumDecoderLayers int
	// DModel is the model hidden dimension.
	DModel int
	// NumHeads is the number of attention heads.
	NumHeads int
	// DFF is the feed-forward intermediate dimension.
	DFF int
	// VocabSize is the size of the token vocabulary.
	VocabSize int
	// ModelName is an optional human-readable name stored in general.name.
	ModelName string
}

// Validate checks that the convert config is well-formed.
func (c *ChronosConvertConfig) Validate() error {
	if c.NumEncoderLayers <= 0 {
		return fmt.Errorf("chronos: NumEncoderLayers must be positive, got %d", c.NumEncoderLayers)
	}
	if c.NumDecoderLayers <= 0 {
		return fmt.Errorf("chronos: NumDecoderLayers must be positive, got %d", c.NumDecoderLayers)
	}
	if c.DModel <= 0 {
		return fmt.Errorf("chronos: DModel must be positive, got %d", c.DModel)
	}
	if c.NumHeads <= 0 {
		return fmt.Errorf("chronos: NumHeads must be positive, got %d", c.NumHeads)
	}
	if c.DFF <= 0 {
		return fmt.Errorf("chronos: DFF must be positive, got %d", c.DFF)
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("chronos: VocabSize must be positive, got %d", c.VocabSize)
	}
	return nil
}

// MapChronosTensorName converts a HuggingFace SafeTensors tensor name from an
// amazon/chronos-t5-* checkpoint to the GGUF convention used by the Chronos
// graph builder.
//
// T5 encoder tensors follow the pattern:
//
//	encoder.block.{N}.layer.0.SelfAttention.{q,k,v,o}.weight  →  chronos.enc.block.{N}.attn.{q,k,v,o}.weight
//	encoder.block.{N}.layer.0.layer_norm.weight                →  chronos.enc.block.{N}.attn_norm.weight
//	encoder.block.{N}.layer.1.DenseReluDense.wi.weight         →  chronos.enc.block.{N}.ffn.wi.weight
//	encoder.block.{N}.layer.1.DenseReluDense.wo.weight         →  chronos.enc.block.{N}.ffn.wo.weight
//	encoder.block.{N}.layer.1.layer_norm.weight                →  chronos.enc.block.{N}.ffn_norm.weight
//	encoder.final_layer_norm.weight                            →  chronos.enc.final_norm.weight
//
// T5 decoder tensors follow the pattern:
//
//	decoder.block.{N}.layer.0.SelfAttention.{q,k,v,o}.weight  →  chronos.dec.block.{N}.self_attn.{q,k,v,o}.weight
//	decoder.block.{N}.layer.0.layer_norm.weight                →  chronos.dec.block.{N}.self_attn_norm.weight
//	decoder.block.{N}.layer.1.EncDecAttention.{q,k,v,o}.weight→  chronos.dec.block.{N}.cross_attn.{q,k,v,o}.weight
//	decoder.block.{N}.layer.1.layer_norm.weight                →  chronos.dec.block.{N}.cross_attn_norm.weight
//	decoder.block.{N}.layer.2.DenseReluDense.wi.weight         →  chronos.dec.block.{N}.ffn.wi.weight
//	decoder.block.{N}.layer.2.DenseReluDense.wo.weight         →  chronos.dec.block.{N}.ffn.wo.weight
//	decoder.block.{N}.layer.2.layer_norm.weight                →  chronos.dec.block.{N}.ffn_norm.weight
//	decoder.final_layer_norm.weight                            →  chronos.dec.final_norm.weight
//
// Global tensors:
//
//	shared.weight                                              →  chronos.token_embd.weight
//	lm_head.weight                                             →  chronos.lm_head.weight
//	encoder.embed_tokens.weight                                →  chronos.enc.token_embd.weight
//	decoder.embed_tokens.weight                                →  chronos.dec.token_embd.weight
//
// Relative position bias:
//
//	encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight → chronos.enc.attn_rel_bias.weight
//	decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight → chronos.dec.self_attn_rel_bias.weight
func MapChronosTensorName(hfName string) string {
	// Global / embedding tensors.
	switch hfName {
	case "shared.weight":
		return "chronos.token_embd.weight"
	case "lm_head.weight":
		return "chronos.lm_head.weight"
	case "encoder.embed_tokens.weight":
		return "chronos.enc.token_embd.weight"
	case "decoder.embed_tokens.weight":
		return "chronos.dec.token_embd.weight"
	case "encoder.final_layer_norm.weight":
		return "chronos.enc.final_norm.weight"
	case "decoder.final_layer_norm.weight":
		return "chronos.dec.final_norm.weight"
	}

	// Encoder block tensors: encoder.block.{N}.layer.{L}.{rest}
	if strings.HasPrefix(hfName, "encoder.block.") {
		return mapChronosBlockTensor(hfName, "encoder", "enc", false)
	}

	// Decoder block tensors: decoder.block.{N}.layer.{L}.{rest}
	if strings.HasPrefix(hfName, "decoder.block.") {
		return mapChronosBlockTensor(hfName, "decoder", "dec", true)
	}

	// Fallback: prefix with chronos.
	return "chronos." + hfName
}

// mapChronosBlockTensor handles encoder/decoder block tensor name mapping.
// For encoder: layer.0 = self-attention, layer.1 = FFN
// For decoder: layer.0 = self-attention, layer.1 = cross-attention, layer.2 = FFN
func mapChronosBlockTensor(hfName, hfPrefix, ggufPrefix string, isDecoder bool) string {
	// Strip "{hfPrefix}.block." to get "{N}.layer.{L}.{rest}"
	stripped := strings.TrimPrefix(hfName, hfPrefix+".block.")

	// Split into block number and the rest after "layer."
	dotIdx := strings.Index(stripped, ".")
	if dotIdx < 0 {
		return "chronos." + ggufPrefix + ".block." + stripped
	}
	blockNum := stripped[:dotIdx]
	afterBlock := stripped[dotIdx+1:] // "layer.{L}.{rest}"

	if !strings.HasPrefix(afterBlock, "layer.") {
		return "chronos." + ggufPrefix + ".block." + blockNum + "." + afterBlock
	}
	layerRest := strings.TrimPrefix(afterBlock, "layer.")

	// Split layer number from the rest.
	dotIdx2 := strings.Index(layerRest, ".")
	if dotIdx2 < 0 {
		return "chronos." + ggufPrefix + ".block." + blockNum + "." + layerRest
	}
	layerNum := layerRest[:dotIdx2]
	rest := layerRest[dotIdx2+1:]

	prefix := "chronos." + ggufPrefix + ".block." + blockNum

	// Map based on layer number.
	switch {
	case layerNum == "0":
		// Self-attention layer.
		return mapChronosAttnOrNorm(prefix, rest, isDecoder, "self_attn", "attn")
	case layerNum == "1" && isDecoder:
		// Cross-attention layer (decoder only).
		return mapChronosCrossAttnOrNorm(prefix, rest)
	case (layerNum == "1" && !isDecoder) || (layerNum == "2" && isDecoder):
		// FFN layer.
		return mapChronosFFNOrNorm(prefix, rest)
	default:
		return prefix + ".layer" + layerNum + "." + rest
	}
}

func mapChronosAttnOrNorm(prefix, rest string, isDecoder bool, decoderAttnName, encoderAttnName string) string {
	attnName := encoderAttnName
	normName := attnName + "_norm"
	if isDecoder {
		attnName = decoderAttnName
		normName = attnName + "_norm"
	}

	if strings.HasPrefix(rest, "SelfAttention.relative_attention_bias.") {
		suffix := strings.TrimPrefix(rest, "SelfAttention.relative_attention_bias.")
		if isDecoder {
			return prefix + "." + attnName + "_rel_bias" + "." + suffix
		}
		// Encoder relative bias is stored at the enc level (only block 0 has it).
		encPrefix := strings.TrimSuffix(prefix, ".block.0")
		if encPrefix == prefix {
			// Not block 0, just map normally.
			return prefix + "." + attnName + "_rel_bias." + suffix
		}
		return encPrefix + ".attn_rel_bias." + suffix
	}

	if strings.HasPrefix(rest, "SelfAttention.") {
		suffix := strings.TrimPrefix(rest, "SelfAttention.")
		return prefix + "." + attnName + "." + suffix
	}

	if strings.HasPrefix(rest, "layer_norm.") {
		suffix := strings.TrimPrefix(rest, "layer_norm.")
		return prefix + "." + normName + "." + suffix
	}

	return prefix + "." + rest
}

func mapChronosCrossAttnOrNorm(prefix, rest string) string {
	if strings.HasPrefix(rest, "EncDecAttention.") {
		suffix := strings.TrimPrefix(rest, "EncDecAttention.")
		return prefix + ".cross_attn." + suffix
	}
	if strings.HasPrefix(rest, "layer_norm.") {
		suffix := strings.TrimPrefix(rest, "layer_norm.")
		return prefix + ".cross_attn_norm." + suffix
	}
	return prefix + "." + rest
}

func mapChronosFFNOrNorm(prefix, rest string) string {
	if strings.HasPrefix(rest, "DenseReluDense.") {
		suffix := strings.TrimPrefix(rest, "DenseReluDense.")
		return prefix + ".ffn." + suffix
	}
	if strings.HasPrefix(rest, "layer_norm.") {
		suffix := strings.TrimPrefix(rest, "layer_norm.")
		return prefix + ".ffn_norm." + suffix
	}
	return prefix + "." + rest
}

// ConvertChronosToGGUF reads a Chronos-2 (T5) SafeTensors checkpoint and
// writes a GGUF file with architecture metadata and converted tensor names.
func ConvertChronosToGGUF(safeTensorsPath, ggufPath string, cfg ChronosConvertConfig) error {
	if err := cfg.Validate(); err != nil {
		return err
	}

	sf, err := os.Open(filepath.Clean(safeTensorsPath))
	if err != nil {
		return fmt.Errorf("chronos: open safetensors: %w", err)
	}
	defer func() { _ = sf.Close() }()

	header, dataOffset, err := parseSafeTensorsHeader(sf)
	if err != nil {
		return fmt.Errorf("chronos: %w", err)
	}

	gw := ztensorgguf.NewWriter()

	// Architecture metadata.
	gw.AddMetadataString("general.architecture", "chronos")
	if cfg.ModelName != "" {
		gw.AddMetadataString("general.name", cfg.ModelName)
	}
	gw.AddMetadataUint32("chronos.encoder_block_count", uint32(cfg.NumEncoderLayers))
	gw.AddMetadataUint32("chronos.decoder_block_count", uint32(cfg.NumDecoderLayers))
	gw.AddMetadataUint32("chronos.d_model", uint32(cfg.DModel))
	gw.AddMetadataUint32("chronos.num_heads", uint32(cfg.NumHeads))
	gw.AddMetadataUint32("chronos.d_ff", uint32(cfg.DFF))
	gw.AddMetadataUint32("chronos.vocab_size", uint32(cfg.VocabSize))

	// Sort tensor names for deterministic output.
	names := make([]string, 0, len(header))
	for name := range header {
		names = append(names, name)
	}
	sort.Strings(names)

	// Read and convert each tensor.
	for _, hfName := range names {
		meta := header[hfName]

		ggufName := MapChronosTensorName(hfName)

		ggufType, elemSize, err := safeTensorsDTypeToGGUF(meta.DType)
		if err != nil {
			return fmt.Errorf("chronos: tensor %q: %w", hfName, err)
		}

		dataLen := meta.DataOffsets[1] - meta.DataOffsets[0]
		if dataLen < 0 {
			return fmt.Errorf("chronos: tensor %q has invalid data offsets [%d, %d]", hfName, meta.DataOffsets[0], meta.DataOffsets[1])
		}

		if _, err := sf.Seek(dataOffset+meta.DataOffsets[0], io.SeekStart); err != nil {
			return fmt.Errorf("chronos: seek to tensor %q data: %w", hfName, err)
		}
		data := make([]byte, dataLen)
		if _, err := io.ReadFull(sf, data); err != nil {
			return fmt.Errorf("chronos: read tensor %q data: %w", hfName, err)
		}

		// Convert F64 to F32 since GGUF has no F64 type.
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
			return fmt.Errorf("chronos: tensor %q shape %v expects %d bytes but got %d",
				hfName, meta.Shape, int64(expectedElems)*int64(elemSize), len(data))
		}

		gw.AddTensor(ggufName, ggufType, meta.Shape, data)
	}

	out, err := os.Create(filepath.Clean(ggufPath))
	if err != nil {
		return fmt.Errorf("chronos: create GGUF file: %w", err)
	}
	defer func() { _ = out.Close() }()

	if err := gw.Write(out); err != nil {
		return fmt.Errorf("chronos: write GGUF: %w", err)
	}

	return nil
}
