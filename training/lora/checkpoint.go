package lora

import (
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/model/gguf"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
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

	w := ztensorgguf.NewWriter()

	// Metadata: general.architecture, lora.rank, lora.alpha.
	rank := adapters[0].rank
	alpha := adapters[0].alpha
	w.AddMetadataString("general.architecture", "lora")
	w.AddMetadataUint32("lora.rank", uint32(rank))
	w.AddMetadataFloat32("lora.alpha", alpha)

	// Add tensors. Each adapter contributes 2 tensors (A and B).
	for _, ll := range adapters {
		aData := toFloat32(ll.A.Value.Data())
		aShape := ll.A.Value.Shape()
		w.AddTensorF32("lora."+ll.name+".weight_a", aShape, aData)

		bData := toFloat32(ll.B.Value.Data())
		bShape := ll.B.Value.Shape()
		w.AddTensorF32("lora."+ll.name+".weight_b", bShape, bData)
	}

	if err := w.Write(f); err != nil {
		return fmt.Errorf("lora: write GGUF: %w", err)
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
