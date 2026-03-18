package lora

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// SaveAdapter writes all LoRA adapter matrices (A, B) from the given model to
// a GGUF file. Tensors are stored as float32 with naming convention
// lora.{layer}.weight_a and lora.{layer}.weight_b. Metadata includes the LoRA
// rank and alpha from the first adapter found.
func SaveAdapter[T tensor.Numeric](path string, m Model[T]) error {
	// Collect all LoRA layers.
	var adapters []*LoraLinear[T]
	for _, layer := range m.Layers() {
		if ll, ok := layer.(*LoraLinear[T]); ok {
			adapters = append(adapters, ll)
		}
	}
	if len(adapters) == 0 {
		return fmt.Errorf("lora: no LoRA adapters found in model")
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("lora: create checkpoint file: %w", err)
	}
	defer f.Close()

	// Build tensor info list. Each adapter contributes 2 tensors (A and B).
	type tensorEntry struct {
		name string
		data []float32
		rows int
		cols int
	}
	var entries []tensorEntry
	for _, ll := range adapters {
		aData := toFloat32(ll.A.Value.Data())
		aShape := ll.A.Value.Shape()
		entries = append(entries, tensorEntry{
			name: "lora." + ll.name + ".weight_a",
			data: aData,
			rows: aShape[0],
			cols: aShape[1],
		})

		bData := toFloat32(ll.B.Value.Data())
		bShape := ll.B.Value.Shape()
		entries = append(entries, tensorEntry{
			name: "lora." + ll.name + ".weight_b",
			data: bData,
			rows: bShape[0],
			cols: bShape[1],
		})
	}

	// Metadata: lora.rank, lora.alpha, general.architecture.
	rank := adapters[0].rank
	alpha := adapters[0].alpha
	metadata := []struct {
		key       string
		valueType uint32
		value     any
	}{
		{"general.architecture", gguf.TypeString, "lora"},
		{"lora.rank", gguf.TypeUint32, uint32(rank)},
		{"lora.alpha", gguf.TypeFloat32, alpha},
	}

	// Write GGUF v3 header.
	if err := binary.Write(f, binary.LittleEndian, gguf.Magic); err != nil {
		return fmt.Errorf("lora: write magic: %w", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil {
		return fmt.Errorf("lora: write version: %w", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(len(entries))); err != nil {
		return fmt.Errorf("lora: write tensor count: %w", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(len(metadata))); err != nil {
		return fmt.Errorf("lora: write metadata count: %w", err)
	}

	// Write metadata key-value pairs.
	for _, kv := range metadata {
		if err := writeGGUFString(f, kv.key); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, kv.valueType); err != nil {
			return err
		}
		switch kv.valueType {
		case gguf.TypeString:
			if err := writeGGUFString(f, kv.value.(string)); err != nil {
				return err
			}
		case gguf.TypeUint32:
			if err := binary.Write(f, binary.LittleEndian, kv.value.(uint32)); err != nil {
				return err
			}
		case gguf.TypeFloat32:
			if err := binary.Write(f, binary.LittleEndian, kv.value.(float32)); err != nil {
				return err
			}
		}
	}

	// Write tensor info entries. GGUF stores dimensions in GGML order
	// (innermost-first): for a 2D tensor with shape [rows, cols],
	// ne[0]=cols, ne[1]=rows.
	var offset uint64
	for _, e := range entries {
		if err := writeGGUFString(f, e.name); err != nil {
			return err
		}
		// n_dimensions
		if err := binary.Write(f, binary.LittleEndian, uint32(2)); err != nil {
			return err
		}
		// dimensions in GGML order: ne[0]=cols, ne[1]=rows
		if err := binary.Write(f, binary.LittleEndian, uint64(e.cols)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint64(e.rows)); err != nil {
			return err
		}
		// type = F32
		if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGMLTypeF32)); err != nil {
			return err
		}
		// offset relative to data start
		if err := binary.Write(f, binary.LittleEndian, offset); err != nil {
			return err
		}
		offset += uint64(len(e.data)) * 4
	}

	// Align to 32-byte boundary for tensor data.
	const alignment = 32
	pos, err := f.Seek(0, 1)
	if err != nil {
		return fmt.Errorf("lora: seek current: %w", err)
	}
	padLen := (alignment - pos%alignment) % alignment
	if padLen > 0 {
		pad := make([]byte, padLen)
		if _, err := f.Write(pad); err != nil {
			return fmt.Errorf("lora: write alignment padding: %w", err)
		}
	}

	// Write tensor data.
	for _, e := range entries {
		for _, v := range e.data {
			if err := binary.Write(f, binary.LittleEndian, v); err != nil {
				return fmt.Errorf("lora: write tensor data %q: %w", e.name, err)
			}
		}
	}

	return nil
}

// LoadAdapter reads a LoRA adapter GGUF checkpoint and restores the A and B
// matrices into matching LoraLinear layers in the model. Layer names are matched
// by stripping the "lora." prefix and ".weight_a"/".weight_b" suffix.
func LoadAdapter[T tensor.Numeric](path string, m Model[T]) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("lora: open checkpoint: %w", err)
	}
	defer f.Close()

	gf, err := gguf.Parse(f)
	if err != nil {
		return fmt.Errorf("lora: parse GGUF: %w", err)
	}

	tensors, err := gguf.LoadTensors(gf, f)
	if err != nil {
		return fmt.Errorf("lora: load tensors: %w", err)
	}

	// Build map of layer name -> LoraLinear for quick lookup.
	loraLayers := make(map[string]*LoraLinear[T])
	for _, layer := range m.Layers() {
		if ll, ok := layer.(*LoraLinear[T]); ok {
			loraLayers[ll.name] = ll
		}
	}

	// Restore tensors into matching layers.
	restored := 0
	for tName, tData := range tensors {
		if !strings.HasPrefix(tName, "lora.") {
			continue
		}
		rest := strings.TrimPrefix(tName, "lora.")

		var layerName string
		var isA bool
		switch {
		case strings.HasSuffix(rest, ".weight_a"):
			layerName = strings.TrimSuffix(rest, ".weight_a")
			isA = true
		case strings.HasSuffix(rest, ".weight_b"):
			layerName = strings.TrimSuffix(rest, ".weight_b")
			isA = false
		default:
			continue
		}

		ll, ok := loraLayers[layerName]
		if !ok {
			return fmt.Errorf("lora: checkpoint tensor %q has no matching layer %q in model", tName, layerName)
		}

		// Convert float32 tensor data to type T and create a new tensor.
		f32Data := tData.Data()
		tSlice := make([]T, len(f32Data))
		for i, v := range f32Data {
			tSlice[i] = T(v)
		}

		shape := tData.Shape()
		newTensor, err := tensor.New[T](shape, tSlice)
		if err != nil {
			return fmt.Errorf("lora: create tensor for %q: %w", tName, err)
		}

		if isA {
			ll.A.Value = newTensor
		} else {
			ll.B.Value = newTensor
		}
		restored++
	}

	if restored == 0 {
		return fmt.Errorf("lora: no adapter tensors found in checkpoint")
	}

	return nil
}

// writeGGUFString writes a GGUF string (uint64 length + bytes).
func writeGGUFString(f *os.File, s string) error {
	if err := binary.Write(f, binary.LittleEndian, uint64(len(s))); err != nil {
		return fmt.Errorf("lora: write string length: %w", err)
	}
	if _, err := f.Write([]byte(s)); err != nil {
		return fmt.Errorf("lora: write string data: %w", err)
	}
	return nil
}

// toFloat32 converts a slice of any Numeric type to float32.
func toFloat32[T tensor.Numeric](data []T) []float32 {
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = float32(v)
	}
	return out
}

// float32BitsEqual returns true if two float32 values have identical bit patterns,
// handling NaN correctly.
func float32BitsEqual(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b)
}
