package lora

import (
	"fmt"
	"strings"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// Named is implemented by graph nodes that have a name.
type Named interface {
	Name() string
}

// LinearInfo provides the dimensions of a Linear layer for LoRA wrapping.
type LinearInfo interface {
	InputFeatures() int
	OutputFeatures() int
}

// Layer is a named node with dimension information, implemented by Linear layers.
type Layer[T tensor.Numeric] interface {
	graph.Node[T]
	Named
}

// Model provides access to named layers for LoRA injection.
// Implementations must support replacing layers by name.
type Model[T tensor.Numeric] interface {
	// Layers returns all named layers in the model.
	Layers() []Layer[T]

	// ReplaceLayer replaces the layer with the given name.
	// Returns an error if the name is not found.
	ReplaceLayer(name string, replacement Layer[T]) error
}

// InjectLoRA walks the model, finds all Linear layers whose names match
// targetModules, and wraps them with LoraLinear. Base model parameters
// are frozen (only LoRA A and B matrices are trainable).
func InjectLoRA[T tensor.Numeric](
	m Model[T],
	rank int,
	alpha float32,
	targetModules []string,
	engine compute.Engine[T],
) error {
	if rank <= 0 {
		return fmt.Errorf("lora: rank must be positive, got %d", rank)
	}
	if len(targetModules) == 0 {
		return fmt.Errorf("lora: targetModules must not be empty")
	}

	targets := make(map[string]bool, len(targetModules))
	for _, t := range targetModules {
		targets[t] = true
	}

	layers := m.Layers()
	replaced := 0

	for _, layer := range layers {
		name := layer.Name()
		if !matchesTarget(name, targets) {
			continue
		}
		if layer.OpType() != "Linear" {
			continue
		}

		// Extract dimensions from the Linear layer's weight parameter.
		params := layer.Parameters()
		if len(params) == 0 {
			return fmt.Errorf("lora: layer %q has no parameters", name)
		}
		weightShape := params[0].Value.Shape()
		if len(weightShape) != 2 {
			return fmt.Errorf("lora: layer %q weight shape %v is not 2D", name, weightShape)
		}
		dIn, dOut := weightShape[0], weightShape[1]

		loraLayer, err := NewLoraLinear[T](name, layer, rank, alpha, engine, dIn, dOut)
		if err != nil {
			return fmt.Errorf("lora: failed to create LoraLinear for %q: %w", name, err)
		}

		if err := m.ReplaceLayer(name, loraLayer); err != nil {
			return fmt.Errorf("lora: failed to replace layer %q: %w", name, err)
		}
		replaced++
	}

	if replaced == 0 {
		return fmt.Errorf("lora: no matching Linear layers found for targets %v", targetModules)
	}
	return nil
}

// matchesTarget checks if a layer name matches any target module pattern.
// Supports both exact match and suffix match (e.g., "q_proj" matches
// "layers.0.self_attn.q_proj").
func matchesTarget(name string, targets map[string]bool) bool {
	if targets[name] {
		return true
	}
	for t := range targets {
		if strings.HasSuffix(name, "."+t) || strings.HasSuffix(name, "_"+t) {
			return true
		}
	}
	return false
}

// TrainableParamCount returns the count of trainable parameters (those from
// LoRA layers). After InjectLoRA, only LoRA A and B matrices are trainable.
func TrainableParamCount[T tensor.Numeric](m Model[T]) int {
	count := 0
	for _, layer := range m.Layers() {
		if _, ok := layer.(*LoraLinear[T]); ok {
			for _, p := range layer.Parameters() {
				count += tensorSize(p.Value)
			}
		}
	}
	return count
}

// TotalParamCount returns the total number of parameters in the model.
func TotalParamCount[T tensor.Numeric](m Model[T]) int {
	count := 0
	for _, layer := range m.Layers() {
		for _, p := range layer.Parameters() {
			count += tensorSize(p.Value)
		}
		// For LoRA layers, also count the frozen base parameters.
		if ll, ok := layer.(*LoraLinear[T]); ok {
			for _, p := range ll.base.Parameters() {
				count += tensorSize(p.Value)
			}
		}
	}
	return count
}

// tensorSize returns the total number of elements in a tensor.
func tensorSize[T tensor.Numeric](t *tensor.TensorNumeric[T]) int {
	shape := t.Shape()
	size := 1
	for _, s := range shape {
		size *= s
	}
	return size
}
