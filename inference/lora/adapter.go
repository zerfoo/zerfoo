// Package lora implements loading and validation of LoRA (Low-Rank Adaptation)
// adapter weights from GGUF files.
//
// A LoRA adapter contains delta matrices A and B per layer such that
// W_adapted = W_base + scaleFactor * B @ A, where scaleFactor = alpha / rank.
// Tensors are identified by naming convention: "lora_a.<layer_name>" and
// "lora_b.<layer_name>" with rank and alpha stored in GGUF metadata.
package lora

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/zerfoo/zerfoo/model/gguf"
)

// Adapter holds LoRA delta matrices and scaling parameters loaded from a GGUF file.
type Adapter struct {
	Rank        int
	Alpha       float64
	ScaleFactor float64
	Layers      map[string]*Layer
}

// Layer holds the A and B delta matrices for a single adapted layer.
// A has shape [rank, inDim], B has shape [outDim, rank].
// The adapted weight is: W_adapted = W_base + scaleFactor * B @ A.
type Layer struct {
	A          [][]float32
	B          [][]float32
	TargetName string
}

// LoadAdapter parses a GGUF file containing LoRA adapter weights and returns
// a validated Adapter. The GGUF file must contain metadata keys "lora.rank"
// (uint32) and "lora.alpha" (float32), and tensor pairs named
// "lora_a.<layer_name>" and "lora_b.<layer_name>".
func LoadAdapter(path string, r io.ReadSeeker) (*Adapter, error) {
	f, err := gguf.Parse(r)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF %s: %w", path, err)
	}

	rank, ok := f.GetUint32("lora.rank")
	if !ok {
		return nil, fmt.Errorf("GGUF %s: missing metadata key \"lora.rank\"", path)
	}
	if rank == 0 {
		return nil, fmt.Errorf("GGUF %s: lora.rank must be > 0", path)
	}

	alpha, ok := f.GetFloat32("lora.alpha")
	if !ok {
		return nil, fmt.Errorf("GGUF %s: missing metadata key \"lora.alpha\"", path)
	}

	tensors, err := gguf.LoadTensors(f, r)
	if err != nil {
		return nil, fmt.Errorf("load tensors %s: %w", path, err)
	}

	// Group tensors into A/B pairs by layer name.
	aMap := make(map[string]gguf.TensorInfo)
	bMap := make(map[string]gguf.TensorInfo)
	for i := range f.Tensors {
		ti := &f.Tensors[i]
		switch {
		case strings.HasPrefix(ti.Name, "lora_a."):
			name := strings.TrimPrefix(ti.Name, "lora_a.")
			aMap[name] = *ti
		case strings.HasPrefix(ti.Name, "lora_b."):
			name := strings.TrimPrefix(ti.Name, "lora_b.")
			bMap[name] = *ti
		}
	}

	// Validate that every A has a matching B and vice versa.
	for name := range aMap {
		if _, ok := bMap[name]; !ok {
			return nil, fmt.Errorf("GGUF %s: lora_a.%s has no matching lora_b.%s", path, name, name)
		}
	}
	for name := range bMap {
		if _, ok := aMap[name]; !ok {
			return nil, fmt.Errorf("GGUF %s: lora_b.%s has no matching lora_a.%s", path, name, name)
		}
	}

	if len(aMap) == 0 {
		return nil, fmt.Errorf("GGUF %s: no LoRA tensor pairs found", path)
	}

	scaleFactor := float64(alpha) / float64(rank)
	layers := make(map[string]*Layer, len(aMap))

	for name, aTI := range aMap {
		bTI := bMap[name]

		// Validate A shape: [rank, inDim] (2D).
		if len(aTI.Dimensions) != 2 {
			return nil, fmt.Errorf("GGUF %s: lora_a.%s: expected 2 dimensions, got %d", path, name, len(aTI.Dimensions))
		}
		// Validate B shape: [outDim, rank] (2D).
		if len(bTI.Dimensions) != 2 {
			return nil, fmt.Errorf("GGUF %s: lora_b.%s: expected 2 dimensions, got %d", path, name, len(bTI.Dimensions))
		}

		// GGUF stores dimensions in GGML order (innermost-first: ne[0]=cols, ne[1]=rows).
		// After reversal in LoadTensors, shape is [rows, cols] (PyTorch order).
		// A shape: [rank, inDim], B shape: [outDim, rank].
		aTensor := tensors["lora_a."+name]
		bTensor := tensors["lora_b."+name]
		aShape := aTensor.Shape()
		bShape := bTensor.Shape()

		if aShape[0] != int(rank) {
			return nil, fmt.Errorf("GGUF %s: lora_a.%s: expected first dimension %d (rank), got %d", path, name, rank, aShape[0])
		}
		if bShape[1] != int(rank) {
			return nil, fmt.Errorf("GGUF %s: lora_b.%s: expected second dimension %d (rank), got %d", path, name, rank, bShape[1])
		}

		inDim := aShape[1]
		outDim := bShape[0]

		// Build 2D slices from flat tensor data.
		aData := aTensor.Data()
		bData := bTensor.Data()

		aRows := make([][]float32, int(rank))
		for i := range aRows {
			aRows[i] = aData[i*inDim : (i+1)*inDim]
		}

		bRows := make([][]float32, outDim)
		for i := range bRows {
			bRows[i] = bData[i*int(rank) : (i+1)*int(rank)]
		}

		layers[name] = &Layer{
			A:          aRows,
			B:          bRows,
			TargetName: name,
		}
	}

	return &Adapter{
		Rank:        int(rank),
		Alpha:       float64(alpha),
		ScaleFactor: scaleFactor,
		Layers:      layers,
	}, nil
}

// LayerNames returns the sorted list of layer names that have LoRA adaptations.
func (a *Adapter) LayerNames() []string {
	names := make([]string, 0, len(a.Layers))
	for name := range a.Layers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// HasLayer reports whether the adapter has a LoRA adaptation for the given layer name.
func (a *Adapter) HasLayer(name string) bool {
	_, ok := a.Layers[name]
	return ok
}
