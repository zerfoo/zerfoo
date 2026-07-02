package dsl

import "fmt"

// ModelGraph is the validated internal representation of a model definition.
// Use Build to instantiate a runnable Model from it.
type ModelGraph struct {
	name       string
	layers     []LayerDef
	layerIndex map[string]int
	children   map[string][]string
	parents    map[string][]string
	order      []string
	inputs     []string
	outputs    []string
}

// Name returns the model name.
func (g *ModelGraph) Name() string { return g.name }

// Order returns the topological execution order of layer names.
func (g *ModelGraph) Order() []string { return g.order }

// Inputs returns the names of input layers (layers with no parents).
func (g *ModelGraph) Inputs() []string { return g.inputs }

// Outputs returns the names of output layers (layers with no children).
func (g *ModelGraph) Outputs() []string { return g.outputs }

// Build instantiates a runnable Model from the graph.
// inputDim and outputDim specify the dimensions of the model's input and output vectors.
func (g *ModelGraph) Build(inputDim, outputDim int) (*Model, error) {
	if inputDim <= 0 {
		return nil, fmt.Errorf("modeldsl: inputDim must be positive, got %d", inputDim)
	}
	if outputDim <= 0 {
		return nil, fmt.Errorf("modeldsl: outputDim must be positive, got %d", outputDim)
	}

	// Resolve dimensions for each layer by propagating through the graph.
	dims := make(map[string]int, len(g.layers))
	for _, name := range g.order {
		def := g.layers[g.layerIndex[name]]
		parentNames := g.parents[name]

		var inDim int
		if len(parentNames) == 0 {
			inDim = inputDim
		} else {
			inDim = dims[parentNames[0]]
		}

		outDim, err := resolveOutputDim(def, inDim, outputDim, g.children[name])
		if err != nil {
			return nil, fmt.Errorf("modeldsl: layer %q: %w", name, err)
		}
		dims[name] = outDim
	}

	// Build executable layers.
	execLayers := make(map[string]execLayer, len(g.layers))
	for _, name := range g.order {
		def := g.layers[g.layerIndex[name]]
		parentNames := g.parents[name]

		var inDim int
		if len(parentNames) == 0 {
			inDim = inputDim
		} else {
			inDim = dims[parentNames[0]]
		}

		layer, err := buildLayer(def, inDim, dims[name])
		if err != nil {
			return nil, fmt.Errorf("modeldsl: layer %q: %w", name, err)
		}
		execLayers[name] = layer
	}

	return &Model{
		graph:      g,
		execLayers: execLayers,
		dims:       dims,
		inputDim:   inputDim,
		outputDim:  outputDim,
	}, nil
}

// resolveOutputDim determines the output dimension for a layer.
func resolveOutputDim(def LayerDef, inDim, modelOutputDim int, children []string) (int, error) {
	switch def.Type {
	case LayerLinear:
		if v, ok := def.Params["output_dim"]; ok {
			d, err := toInt(v)
			if err != nil {
				return 0, fmt.Errorf("invalid output_dim: %w", err)
			}
			if d <= 0 {
				return 0, fmt.Errorf("output_dim must be positive, got %d", d)
			}
			return d, nil
		}
		// If this is the last layer (no children), use model output dim.
		if len(children) == 0 {
			return modelOutputDim, nil
		}
		return inDim, nil

	case LayerAttention:
		// Attention preserves dimension.
		return inDim, nil

	case LayerRMSNorm, LayerSiLU, LayerSoftmax:
		// Element-wise ops preserve dimension.
		return inDim, nil

	default:
		return 0, fmt.Errorf("unsupported layer type %q", def.Type)
	}
}

func toInt(v any) (int, error) {
	switch val := v.(type) {
	case int:
		return val, nil
	case int64:
		return int(val), nil
	case float64:
		return int(val), nil
	default:
		return 0, fmt.Errorf("expected numeric, got %T", v)
	}
}
