// Package safetensors reads bounded safetensors weight files for conversion
// into Zerfoo's GGUF model format. Runtime model loading remains GGUF-only.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

const maxHeaderBytes = 16 << 20

type tensorHeader struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"`
}

// Tensor contains one decoded F32 tensor. Shape follows the source's outermost
// dimension first convention.
type Tensor struct {
	Shape []int
	Data  []float32
}

// ReadF32 reads a safetensors file whose weight tensors are all F32. It
// rejects unsupported dtypes and malformed offsets instead of dropping them.
func ReadF32(path string) (map[string]Tensor, error) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("read safetensors: %w", err)
	}
	if len(raw) < 8 {
		return nil, fmt.Errorf("safetensors header is truncated")
	}
	headerLen := binary.LittleEndian.Uint64(raw[:8])
	if headerLen == 0 || headerLen > maxHeaderBytes || headerLen > uint64(len(raw)-8) {
		return nil, fmt.Errorf("invalid safetensors header length %d", headerLen)
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+headerLen], &entries); err != nil {
		return nil, fmt.Errorf("decode safetensors header: %w", err)
	}
	data := raw[8+headerLen:]
	out := make(map[string]Tensor, len(entries))
	for name, value := range entries {
		if name == "__metadata__" {
			continue
		}
		var h tensorHeader
		if err := json.Unmarshal(value, &h); err != nil {
			return nil, fmt.Errorf("tensor %q header: %w", name, err)
		}
		if h.DType != "F32" {
			return nil, fmt.Errorf("tensor %q has unsupported dtype %q", name, h.DType)
		}
		if len(h.Shape) == 0 || len(h.DataOffsets) != 2 || h.DataOffsets[0] < 0 || h.DataOffsets[1] < h.DataOffsets[0] || h.DataOffsets[1] > len(data) {
			return nil, fmt.Errorf("tensor %q has invalid shape or offsets", name)
		}
		count := 1
		for _, dim := range h.Shape {
			if dim <= 0 || count > math.MaxInt/dim {
				return nil, fmt.Errorf("tensor %q has invalid shape %v", name, h.Shape)
			}
			count *= dim
		}
		if count > math.MaxInt/4 || h.DataOffsets[1]-h.DataOffsets[0] != count*4 {
			return nil, fmt.Errorf("tensor %q byte count does not match shape", name)
		}
		values := make([]float32, count)
		bytes := data[h.DataOffsets[0]:h.DataOffsets[1]]
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
			if math.IsNaN(float64(values[i])) || math.IsInf(float64(values[i]), 0) {
				return nil, fmt.Errorf("tensor %q contains non-finite value", name)
			}
		}
		out[name] = Tensor{Shape: h.Shape, Data: values}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("safetensors contains no weight tensors")
	}
	return out, nil
}
