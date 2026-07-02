package transmla

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"

	"github.com/zerfoo/zerfoo/model/gguf"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
)

// kvProjPattern matches tensor names like "model.layers.N.self_attn.k_proj.weight"
// or "model.layers.N.self_attn.v_proj.weight".
var kvProjPattern = regexp.MustCompile(`^model\.layers\.(\d+)\.self_attn\.(k|v)_proj\.weight$`)

// ConvertGGUFOptions configures the TransMLA GGUF conversion.
type ConvertGGUFOptions struct {
	// Rank is the KV LoRA dimension (truncated SVD rank).
	Rank int
	// SourceArch is the original model architecture name (e.g., "llama").
	SourceArch string
	// OnLayerDone is called after each layer's K/V projection is decomposed.
	// Arguments are the layer index (0-based) and total number of layers.
	// May be nil.
	OnLayerDone func(layer, total int)
}

// ConvertGGUF reads a source GGUF file, decomposes K/V projection weights
// per layer via truncated SVD, and writes a new GGUF with:
//   - All original tensors except k_proj/v_proj weights
//   - Three new tensors per layer: transmla.{layer}.wDKV, wUK, wUV
//   - Metadata: transmla.kv_lora_dim, transmla.source_arch
//
// The source GGUF metadata is preserved in the output.
func ConvertGGUF(src io.ReadSeeker, dst io.Writer, opts ConvertGGUFOptions) error {
	if opts.Rank <= 0 {
		return fmt.Errorf("transmla: rank must be positive, got %d", opts.Rank)
	}

	f, err := gguf.Parse(src)
	if err != nil {
		return fmt.Errorf("transmla: parse source GGUF: %w", err)
	}

	// Identify which layers have K/V projections and read their raw data.
	// Map layer index -> {"k": rawBytes, "v": rawBytes} and shapes.
	type kvInfo struct {
		data   []byte
		shape  []int // outermost-first (rows, cols)
		nElems int
		typ    gguf.GGMLType
	}
	layerKV := make(map[int]map[string]kvInfo) // layer -> "k"/"v" -> info
	skipTensors := make(map[string]bool)

	for i := range f.Tensors {
		ti := &f.Tensors[i]
		m := kvProjPattern.FindStringSubmatch(ti.Name)
		if m == nil {
			continue
		}
		layerIdx, _ := strconv.Atoi(m[1])
		kv := m[2] // "k" or "v"

		// Compute element count.
		var nElems int64 = 1
		for _, d := range ti.Dimensions {
			nElems *= int64(d)
		}

		dataSize, err := gguf.TensorByteSize(ti.Type, int(nElems))
		if err != nil {
			return fmt.Errorf("transmla: tensor %q byte size: %w", ti.Name, err)
		}

		offset := f.DataOffset + int64(ti.Offset)
		if _, err := src.Seek(offset, io.SeekStart); err != nil {
			return fmt.Errorf("transmla: seek tensor %q: %w", ti.Name, err)
		}
		raw := make([]byte, dataSize)
		if _, err := io.ReadFull(src, raw); err != nil {
			return fmt.Errorf("transmla: read tensor %q: %w", ti.Name, err)
		}

		// GGUF stores dims innermost-first; reverse to outermost-first.
		shape := make([]int, len(ti.Dimensions))
		for j, d := range ti.Dimensions {
			shape[len(ti.Dimensions)-1-j] = int(d)
		}

		if layerKV[layerIdx] == nil {
			layerKV[layerIdx] = make(map[string]kvInfo)
		}
		layerKV[layerIdx][kv] = kvInfo{
			data:   raw,
			shape:  shape,
			nElems: int(nElems),
			typ:    ti.Type,
		}
		skipTensors[ti.Name] = true
	}

	w := ztensorgguf.NewWriter()

	// Copy original metadata.
	for key, val := range f.Metadata {
		if err := addMetadata(w, key, val); err != nil {
			return fmt.Errorf("transmla: copy metadata %q: %w", key, err)
		}
	}

	// Add TransMLA metadata.
	w.AddMetadataUint32("transmla.kv_lora_dim", uint32(opts.Rank))
	if opts.SourceArch != "" {
		w.AddMetadataString("transmla.source_arch", opts.SourceArch)
	}

	// Copy non-K/V tensors as raw bytes.
	for i := range f.Tensors {
		ti := &f.Tensors[i]
		if skipTensors[ti.Name] {
			continue
		}

		var nElems int64 = 1
		for _, d := range ti.Dimensions {
			nElems *= int64(d)
		}
		dataSize, err := gguf.TensorByteSize(ti.Type, int(nElems))
		if err != nil {
			return fmt.Errorf("transmla: tensor %q byte size: %w", ti.Name, err)
		}

		offset := f.DataOffset + int64(ti.Offset)
		if _, err := src.Seek(offset, io.SeekStart); err != nil {
			return fmt.Errorf("transmla: seek tensor %q: %w", ti.Name, err)
		}
		raw := make([]byte, dataSize)
		if _, err := io.ReadFull(src, raw); err != nil {
			return fmt.Errorf("transmla: read tensor %q: %w", ti.Name, err)
		}

		// Reverse dims for writer's convention (outermost-first).
		shape := make([]int, len(ti.Dimensions))
		for j, d := range ti.Dimensions {
			shape[len(ti.Dimensions)-1-j] = int(d)
		}

		w.AddTensor(ti.Name, int(ti.Type), shape, raw)
	}

	// Decompose K/V projections and add TransMLA tensors.
	totalLayers := len(layerKV)
	for layerIdx, kv := range layerKV {
		kInfo, kOK := kv["k"]
		vInfo, vOK := kv["v"]
		if !kOK || !vOK {
			return fmt.Errorf("transmla: layer %d missing k or v projection", layerIdx)
		}

		// Dequantize K and V to float32, then convert to float64 for SVD.
		kF32, err := dequantizeToFloat32(kInfo.typ, kInfo.data, kInfo.nElems)
		if err != nil {
			return fmt.Errorf("transmla: layer %d k_proj dequantize: %w", layerIdx, err)
		}
		vF32, err := dequantizeToFloat32(vInfo.typ, vInfo.data, vInfo.nElems)
		if err != nil {
			return fmt.Errorf("transmla: layer %d v_proj dequantize: %w", layerIdx, err)
		}

		// K shape: [dK, dModel], V shape: [dV, dModel] (outermost-first).
		dK := kInfo.shape[0]
		dModel := kInfo.shape[1]
		dV := vInfo.shape[0]

		// Convert to [][]float64 for SVD.
		wK := flat32ToMatrix64(kF32, dK, dModel)
		wV := flat32ToMatrix64(vF32, dV, dModel)

		wDKV, wUK, wUV, err := DecomposeKVProjection(wK, wV, opts.Rank)
		if err != nil {
			return fmt.Errorf("transmla: layer %d decompose: %w", layerIdx, err)
		}

		// wDKV: [dModel, rank], wUK: [dK, rank], wUV: [dV, rank]
		prefix := fmt.Sprintf("transmla.%d.", layerIdx)
		w.AddTensorF32(prefix+"wDKV", []int{dModel, opts.Rank}, matrix64ToFlat32(wDKV))
		w.AddTensorF32(prefix+"wUK", []int{dK, opts.Rank}, matrix64ToFlat32(wUK))
		w.AddTensorF32(prefix+"wUV", []int{dV, opts.Rank}, matrix64ToFlat32(wUV))

		if opts.OnLayerDone != nil {
			opts.OnLayerDone(layerIdx, totalLayers)
		}
	}

	return w.Write(dst)
}

// dequantizeToFloat32 converts raw GGUF tensor bytes to a flat float32 slice.
func dequantizeToFloat32(typ gguf.GGMLType, raw []byte, nElems int) ([]float32, error) {
	switch typ {
	case gguf.GGMLTypeF32:
		data := make([]float32, nElems)
		for i := range nElems {
			data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
		}
		return data, nil
	case gguf.GGMLTypeF16:
		data := make([]float32, nElems)
		for i := range nElems {
			bits := binary.LittleEndian.Uint16(raw[i*2 : i*2+2])
			data[i] = float16ToFloat32(bits)
		}
		return data, nil
	case gguf.GGMLTypeBF16:
		data := make([]float32, nElems)
		for i := range nElems {
			bits := binary.LittleEndian.Uint16(raw[i*2 : i*2+2])
			data[i] = math.Float32frombits(uint32(bits) << 16)
		}
		return data, nil
	default:
		return nil, fmt.Errorf("unsupported tensor type %d for TransMLA dequantization (only F32, F16, BF16 supported)", typ)
	}
}

// float16ToFloat32 converts IEEE 754 half-precision bits to float32.
func float16ToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) & 1
	exp := uint32(bits>>10) & 0x1F
	mant := uint32(bits) & 0x3FF

	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal: normalize.
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
	case exp == 31:
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7F800000) // Inf
		}
		return math.Float32frombits((sign << 31) | 0x7FC00000) // NaN
	default:
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
	}
}

// flat32ToMatrix64 reshapes a flat float32 slice into a [][]float64 matrix.
func flat32ToMatrix64(flat []float32, rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := range rows {
		m[i] = make([]float64, cols)
		for j := range cols {
			m[i][j] = float64(flat[i*cols+j])
		}
	}
	return m
}

// matrix64ToFlat32 flattens a [][]float64 matrix into a flat float32 slice.
func matrix64ToFlat32(m [][]float64) []float32 {
	if len(m) == 0 {
		return nil
	}
	cols := len(m[0])
	flat := make([]float32, len(m)*cols)
	for i, row := range m {
		for j, v := range row {
			flat[i*cols+j] = float32(v)
		}
	}
	return flat
}

// addMetadata copies a single GGUF metadata value to the writer.
func addMetadata(w *ztensorgguf.Writer, key string, val any) error {
	switch v := val.(type) {
	case string:
		w.AddMetadataString(key, v)
	case uint32:
		w.AddMetadataUint32(key, v)
	case int32:
		w.AddMetadataInt32(key, v)
	case float32:
		w.AddMetadataFloat32(key, v)
	case bool:
		w.AddMetadataBool(key, v)
	case uint64:
		w.AddMetadataUint64(key, v)
	case int64:
		w.AddMetadataInt64(key, v)
	case uint8:
		// GGUF TypeUint8 — store as uint32 (smallest type the writer supports).
		w.AddMetadataUint32(key, uint32(v))
	case int8:
		w.AddMetadataInt32(key, int32(v))
	case uint16:
		w.AddMetadataUint32(key, uint32(v))
	case int16:
		w.AddMetadataInt32(key, int32(v))
	case float64:
		w.AddMetadataFloat32(key, float32(v))
	case []any:
		// Array types — attempt string array or uint32 array.
		if len(v) == 0 {
			return nil // skip empty arrays
		}
		if s, ok := v[0].(string); ok {
			_ = s
			strs := make([]string, len(v))
			for i, elem := range v {
				strs[i], _ = elem.(string)
			}
			w.AddMetadataStringArray(key, strs)
		} else if _, ok := v[0].(uint32); ok {
			u32s := make([]uint32, len(v))
			for i, elem := range v {
				u32s[i], _ = elem.(uint32)
			}
			w.AddMetadataUint32Array(key, u32s)
		}
		// Other array types silently skipped for now.
	default:
		// Skip unknown types rather than failing.
	}
	return nil
}
