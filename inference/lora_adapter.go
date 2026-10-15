package inference

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

type adapterPair struct {
	a, b *tensor.TensorNumeric[float32]
}

// ApplyLoRAAdapter merges a GGUF LoRA adapter into canonical model tensors
// before graph construction. It is architecture-neutral and never mutates the
// mapped base tensor storage. Standard A=[rank,input], B=[output,rank] layout
// yields W' = W + (alpha/rank)*B*A.
func ApplyLoRAAdapter(base map[string]*tensor.TensorNumeric[float32], path string) error {
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return fmt.Errorf("open LoRA adapter: %w", err)
	}
	defer func() { _ = f.Close() }()
	file, err := gguf.Parse(f)
	if err != nil {
		return fmt.Errorf("parse LoRA adapter: %w", err)
	}
	arch, _ := file.GetString("general.architecture")
	rank, rankOK := file.GetUint32("lora.rank")
	alpha, alphaOK := file.GetFloat32("lora.alpha")
	if arch != "lora" || !rankOK || rank == 0 || !alphaOK || alpha <= 0 || math.IsNaN(float64(alpha)) || math.IsInf(float64(alpha), 0) {
		return fmt.Errorf("invalid LoRA adapter metadata")
	}
	weights, err := gguf.LoadTensors(file, f)
	if err != nil {
		return fmt.Errorf("load LoRA adapter tensors: %w", err)
	}
	pairs := make(map[string]*adapterPair)
	for name, weight := range weights {
		if !strings.HasPrefix(name, "lora.") {
			return fmt.Errorf("unexpected adapter tensor %q", name)
		}
		rest := strings.TrimPrefix(name, "lora.")
		var layer string
		var isA bool
		switch {
		case strings.HasSuffix(rest, ".weight_a"):
			layer = strings.TrimSuffix(rest, ".weight_a")
			isA = true
		case strings.HasSuffix(rest, ".weight_b"):
			layer = strings.TrimSuffix(rest, ".weight_b")
		default:
			return fmt.Errorf("unexpected adapter tensor %q", name)
		}
		pair := pairs[layer]
		if pair == nil {
			pair = &adapterPair{}
			pairs[layer] = pair
		}
		if isA {
			pair.a = weight
		} else {
			pair.b = weight
		}
	}
	if len(pairs) == 0 {
		return fmt.Errorf("LoRA adapter has no tensor pairs")
	}
	names := make([]string, 0, len(pairs))
	for layer, pair := range pairs {
		if pair.a == nil || pair.b == nil {
			return fmt.Errorf("LoRA layer %q is missing A or B", layer)
		}
		aShape, bShape := pair.a.Shape(), pair.b.Shape()
		if len(aShape) != 2 || len(bShape) != 2 || aShape[0] != int(rank) || bShape[1] != int(rank) {
			return fmt.Errorf("LoRA layer %q has invalid rank", layer)
		}
		baseName := layer + ".weight"
		if _, ok := base[baseName]; !ok {
			baseName = layer
		}
		baseWeight, ok := base[baseName]
		if !ok {
			return fmt.Errorf("LoRA layer %q has no base weight", layer)
		}
		shape := baseWeight.Shape()
		if len(shape) != 2 || shape[0] != bShape[0] || shape[1] != aShape[1] {
			return fmt.Errorf("LoRA layer %q shape does not match base %v", layer, shape)
		}
		for _, values := range [][]float32{pair.a.Data(), pair.b.Data()} {
			for _, value := range values {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return fmt.Errorf("LoRA layer %q has non-finite weight", layer)
				}
			}
		}
		names = append(names, layer)
	}
	sort.Strings(names)
	scale := alpha / float32(rank)
	for _, layer := range names {
		pair := pairs[layer]
		baseName := layer + ".weight"
		if _, ok := base[baseName]; !ok {
			baseName = layer
		}
		shape := base[baseName].Shape()
		rows, cols := shape[0], shape[1]
		data := append([]float32(nil), base[baseName].Data()...)
		a, b := pair.a.Data(), pair.b.Data()
		for row := 0; row < rows; row++ {
			for k := 0; k < int(rank); k++ {
				coefficient := scale * b[row*int(rank)+k]
				for col := 0; col < cols; col++ {
					data[row*cols+col] += coefficient * a[k*cols+col]
				}
			}
		}
		for _, value := range data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("LoRA layer %q produced non-finite weight", layer)
			}
		}
		merged, tensorErr := tensor.New(shape, data)
		if tensorErr != nil {
			return fmt.Errorf("merge LoRA layer %q: %w", layer, tensorErr)
		}
		base[baseName] = merged
	}
	return nil
}
