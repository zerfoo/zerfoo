// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
	"github.com/zerfoo/zmf"
)

// debugONNX returns true when ZERFOO_DEBUG_ONNX=1 is set.
func debugONNX() bool { return os.Getenv("ZERFOO_DEBUG_ONNX") == "1" }

// BuildOption configures optional behavior for BuildFromZMF.
type BuildOption func(*buildConfig)

type buildConfig struct {
	resolver         ParamResolver
	globalAttributes map[string]interface{}
}

// WithGlobalAttributes injects extra key-value pairs into every node's
// attribute map during graph construction. This is used, for example, to
// propagate rope_scaling_* config from config.json into every GQA node
// without requiring the ZMF file to carry these attributes.
func WithGlobalAttributes(attrs map[string]interface{}) BuildOption {
	return func(c *buildConfig) { c.globalAttributes = attrs }
}

// WithParamResolver supplies an architecture-aware parameter name resolver.
// When provided, the resolver adds canonical aliases to the parameter map
// so that layer builders can look up parameters by canonical name even when
// the ZMF file uses architecture-specific names.
func WithParamResolver(r ParamResolver) BuildOption {
	return func(c *buildConfig) { c.resolver = r }
}

// BuildFromZMF constructs a Zerfoo computation graph from a ZMF model definition.
// This function iterates through the nodes in the graph, instantiates the
// corresponding layers using a registered builder, and connects them into an
// executable graph.
//
//nolint:gocyclo // High-level orchestration with many cases; splitting would harm clarity.
func BuildFromZMF[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	model *zmf.Model,
	opts ...BuildOption,
) (*graph.Graph[T], error) {
	if model == nil || model.Graph == nil {
		return nil, errors.New("cannot build model from nil or empty ZMF graph")
	}

	cfg := buildConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	params, err := convertParameters[T](model.Graph.Parameters)
	if err != nil {
		return nil, err
	}

	// Apply resolver to add canonical aliases alongside original names.
	if cfg.resolver != nil {
		params = ResolveAll(cfg.resolver, params)
	}

	builder := graph.NewBuilder[T](engine)
	instantiatedNodes := make(map[string]graph.Node[T])

	// 1. Handle Graph Inputs
	// These are the entry points to the graph. We create special input nodes for them.
	// Only create input nodes for actual model inputs, not parameters.
	// For KV-cache models, auto-supply attention_mask, position_ids, and past_key_values.
	var primaryInputNode graph.Node[T]
	for _, inputProto := range model.Graph.Inputs {
		// Skip parameters - they should be embedded in layers, not treated as inputs
		if _, isParam := params[inputProto.Name]; isParam {
			continue
		}

		dims := make([]int, len(inputProto.Shape))
		for i, dim := range inputProto.Shape {
			dims[i] = int(dim)
		}

		name := inputProto.Name
		switch {
		case strings.HasPrefix(name, "past_key_values"):
			// KV cache inputs: create zero-cache nodes wired to the primary input.
			numHeads, headDim := 1, 1
			if len(dims) >= 4 {
				numHeads = dims[1]
				headDim = dims[3]
			}
			if numHeads == 0 {
				numHeads = 1
			}
			if headDim == 0 {
				headDim = 64
			}
			node := &kvCacheIONode[T]{numHeads: numHeads, headDim: headDim}
			instantiatedNodes[name] = node
			// Wire to primary input later (deferred).
		case name == "attention_mask":
			node := &maskFromInputNode[T]{}
			instantiatedNodes[name] = node
		case name == "position_ids":
			node := &positionIdsNode[T]{}
			instantiatedNodes[name] = node
		default:
			inputNode := builder.Input(dims)
			instantiatedNodes[name] = inputNode
			if primaryInputNode == nil {
				primaryInputNode = inputNode
			}
		}
	}

	// Wire auto-input nodes to the primary input so they can derive batch/seq dims.
	if primaryInputNode != nil {
		for _, inputProto := range model.Graph.Inputs {
			name := inputProto.Name
			if name == "attention_mask" || name == "position_ids" || strings.HasPrefix(name, "past_key_values") {
				if node, ok := instantiatedNodes[name]; ok {
					builder.AddNode(node, primaryInputNode)
				}
			}
		}
	}

	// 1.5. Handle Parameters as nodes (only add them if they don't conflict with layer nodes)
	// We'll add parameters on-demand during the connection phase to avoid conflicts

	// 2. First pass: Instantiate all layer nodes
	for _, nodeProto := range model.Graph.Nodes {
		// Skip if a node with this name already exists (e.g., it's an input or parameter)
		if _, exists := instantiatedNodes[nodeProto.Name]; exists {
			continue
		}

		// Special case: Constant nodes embed their value in a tensor attribute.
		// Handle them inline without a registry lookup to avoid a circular dependency
		// (layers packages cannot import the model package).
		if nodeProto.OpType == "Constant" {
			constNode, err := buildConstantNode[T](nodeProto)
			if err != nil {
				return nil, fmt.Errorf("failed to build Constant node '%s': %w", nodeProto.Name, err)
			}
			instantiatedNodes[nodeProto.Name] = constNode
			// Also register every output alias so downstream nodes can reference them.
			for _, outName := range nodeProto.Outputs {
				if outName != "" {
					instantiatedNodes[outName] = constNode
				}
			}
			continue
		}

		layerBuilder, err := GetLayerBuilder[T](nodeProto.OpType)
		if err != nil {
			return nil, err
		}

		attributes := convertAttributes(nodeProto.Attributes)
		// Promote dedicated proto fields into the attributes map so layer
		// builders can find them via the standard attributes interface.
		if len(nodeProto.Perm) > 0 {
			if _, exists := attributes["perm"]; !exists {
				attributes["perm"] = nodeProto.Perm
			}
		}
		if nodeProto.Epsilon != nil {
			if _, exists := attributes["epsilon"]; !exists {
				attributes["epsilon"] = *nodeProto.Epsilon
			}
		}
		if nodeProto.Axis != nil {
			if _, exists := attributes["axis"]; !exists {
				attributes["axis"] = int(*nodeProto.Axis)
			}
		}
		// Merge global attributes (e.g. rope_scaling) into per-node attributes.
		for k, v := range cfg.globalAttributes {
			if _, exists := attributes[k]; !exists {
				attributes[k] = v
			}
		}

		node, err := layerBuilder(engine, ops, nodeProto.Name, params, attributes)
		if err != nil {
			return nil, fmt.Errorf("failed to build node '%s' of type '%s': %w", nodeProto.Name, nodeProto.OpType, err)
		}

		instantiatedNodes[nodeProto.Name] = node
		// Register output aliases so downstream nodes can reference them by output name.
		for _, outName := range nodeProto.Outputs {
			if outName != "" && outName != nodeProto.Name {
				instantiatedNodes[outName] = node
			}
		}
	}

	// 3. Second pass: Connect the nodes
	for _, nodeProto := range model.Graph.Nodes {
		currentNode := instantiatedNodes[nodeProto.Name]

		// Filter out empty input names first
		validInputNames := make([]string, 0, len(nodeProto.Inputs))
		for _, inputName := range nodeProto.Inputs {
			if inputName != "" {
				validInputNames = append(validInputNames, inputName)
			}
		}

		// Special handling for layers with embedded weights/parameters
		actualInputNames := validInputNames
		if gatherLayer, isGather := currentNode.(interface{ HasEmbeddedWeights() bool }); isGather && gatherLayer.HasEmbeddedWeights() {
			// Skip the first input (weights) since it's embedded in the layer
			if len(actualInputNames) > 1 {
				actualInputNames = actualInputNames[1:]
			}
		} else {
			switch nodeProto.OpType {
			case "MatMul":
				if strings.Contains(nodeProto.Name, "lm_head") && len(actualInputNames) > 1 {
					weightInputName := actualInputNames[1]
					if strings.Contains(weightInputName, "embed_tokens") {
						if embedParam, exists := params[weightInputName]; exists {
							transposedTensor, err := engine.Transpose(context.Background(), embedParam.Value, []int{1, 0})
							if err != nil {
								return nil, fmt.Errorf("failed to transpose embedding weights for lm_head: %w", err)
							}

							transposedParam := &graph.Parameter[T]{
								Name:  weightInputName + "_transposed",
								Value: transposedTensor,
							}
							params[weightInputName+"_transposed"] = transposedParam
							actualInputNames[1] = weightInputName + "_transposed"
						}
					}
				}
			case "SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization":
				if len(actualInputNames) > 1 {
					actualInputNames = actualInputNames[:1]
				}
			case "Reshape":
				if len(actualInputNames) > 1 {
					shapeInputName := actualInputNames[1]
					if shapeParam, exists := params[shapeInputName]; exists {
						shapeValues := make([]int64, shapeParam.Value.Size())
						for i := 0; i < shapeParam.Value.Size(); i++ { //nolint:intrange // classic loop for generic tensor access
							val, err := shapeParam.Value.At(i)
							if err != nil {
								return nil, fmt.Errorf("failed to extract shape value at index %d: %w", i, err)
							}

							shapeValues[i] = int64(val)
						}

						if nodeProto.Attributes == nil {
							nodeProto.Attributes = make(map[string]*zmf.Attribute)
						}

						intsAttr := &zmf.Ints{Val: shapeValues}
						attr := &zmf.Attribute{Value: &zmf.Attribute_Ints{Ints: intsAttr}}
						nodeProto.Attributes["shape"] = attr

						// Rebuild the node with the extracted shape attribute.
						updatedAttrs := convertAttributes(nodeProto.Attributes)
						rebuilt, rebuildErr := GetLayerBuilder[T](nodeProto.OpType)
						if rebuildErr == nil {
							node, nodeErr := rebuilt(engine, ops, nodeProto.Name, params, updatedAttrs)
							if nodeErr == nil {
								instantiatedNodes[nodeProto.Name] = node
								for _, outName := range nodeProto.Outputs {
									if outName != "" && outName != nodeProto.Name {
										instantiatedNodes[outName] = node
									}
								}
								currentNode = node
							}
						}

						// Static shape extracted; only data input needed.
						actualInputNames = actualInputNames[:1]
					} else if resolved := resolveParam(shapeInputName, params, instantiatedNodes); resolved != nil {
						shapeValues := make([]int64, resolved.Size())
						for i := 0; i < resolved.Size(); i++ { //nolint:intrange // classic loop for generic tensor access
							val, _ := resolved.At(i)
							shapeValues[i] = int64(val)
						}

						if nodeProto.Attributes == nil {
							nodeProto.Attributes = make(map[string]*zmf.Attribute)
						}

						intsAttr := &zmf.Ints{Val: shapeValues}
						attr := &zmf.Attribute{Value: &zmf.Attribute_Ints{Ints: intsAttr}}
						nodeProto.Attributes["shape"] = attr

						// Rebuild the node with the extracted shape attribute.
						updatedAttrs := convertAttributes(nodeProto.Attributes)
						rebuilt, rebuildErr := GetLayerBuilder[T](nodeProto.OpType)
						if rebuildErr == nil {
							node, nodeErr := rebuilt(engine, ops, nodeProto.Name, params, updatedAttrs)
							if nodeErr == nil {
								instantiatedNodes[nodeProto.Name] = node
								for _, outName := range nodeProto.Outputs {
									if outName != "" && outName != nodeProto.Name {
										instantiatedNodes[outName] = node
									}
								}
								currentNode = node
							}
						}

						actualInputNames = actualInputNames[:1]
					}
					// else: shape is dynamic (from another node) — keep both
					// inputs so Reshape.Forward receives the shape tensor.
				}
			case "Unsqueeze":
				// ONNX opset 13+: axes come as a second input tensor.
				if len(actualInputNames) > 1 {
					axesParam := resolveParam(actualInputNames[1], params, instantiatedNodes)
					if axesParam != nil {
						axesValues := make([]int64, axesParam.Size())
						for i := 0; i < axesParam.Size(); i++ { //nolint:intrange // generic tensor access
							val, err := axesParam.At(i)
							if err != nil {
								return nil, fmt.Errorf("unsqueeze axes extraction failed at %d: %w", i, err)
							}
							axesValues[i] = int64(val)
						}
						if nodeProto.Attributes == nil {
							nodeProto.Attributes = make(map[string]*zmf.Attribute)
						}
						nodeProto.Attributes["axes"] = &zmf.Attribute{
							Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: axesValues}},
						}
						updatedAttrs := convertAttributes(nodeProto.Attributes)
						rebuilt, rebuildErr := GetLayerBuilder[T](nodeProto.OpType)
						if rebuildErr == nil {
							node, nodeErr := rebuilt(engine, ops, nodeProto.Name, params, updatedAttrs)
							if nodeErr == nil {
								instantiatedNodes[nodeProto.Name] = node
								for _, outName := range nodeProto.Outputs {
									if outName != "" && outName != nodeProto.Name {
										instantiatedNodes[outName] = node
									}
								}
								currentNode = node
							}
						}
					}
					actualInputNames = actualInputNames[:1]
				} else {
					// zonnx may promote the axes constant into an attribute.
					// Translate the promoted key to "axes" and rebuild.
					currentNode = rebuildWithPromotedAxes(nodeProto, engine, ops, instantiatedNodes, currentNode)
				}
			}
		}

		// Connect inputs
		inputNodes := make([]graph.Node[T], len(actualInputNames))
		for i, inputName := range actualInputNames {
			inputNode, ok := instantiatedNodes[inputName]
			if !ok {
				// Try to resolve with output suffix
				resolvedName := resolveOutputSuffix(inputName, instantiatedNodes)
				if resolvedName != "" {
					inputNode, ok = instantiatedNodes[resolvedName]
				}

				if !ok {
					// Try to create a parameter node if this input refers to a parameter
					if param, paramExists := params[inputName]; paramExists {
						paramNode := &parameterNode[T]{value: param.Value}
						instantiatedNodes[inputName] = paramNode
						inputNode = paramNode
					} else if param, paramExists := params[resolvedName]; paramExists {
						// Try parameter lookup with resolved name
						paramNode := &parameterNode[T]{value: param.Value}
						instantiatedNodes[resolvedName] = paramNode
						inputNode = paramNode
					} else {
						// Handle special cases like transposed parameters
						baseParamName := strings.TrimSuffix(inputName, "_transposed")
						if baseParamName != inputName {
							if param, paramExists := params[baseParamName]; paramExists {
								paramNode := &parameterNode[T]{value: param.Value}
								instantiatedNodes[inputName] = paramNode
								inputNode = paramNode
							} else {
								return nil, fmt.Errorf("input node '%s' (resolved: '%s') for node '%s' not found", inputName, resolvedName, nodeProto.Name)
							}
						} else {
							return nil, fmt.Errorf("input node '%s' (resolved: '%s') for node '%s' not found", inputName, resolvedName, nodeProto.Name)
						}
					}
				}
			}

			inputNodes[i] = inputNode
		}

		builder.AddNode(currentNode, inputNodes...)
	}

	// 4. Identify the final output node of the graph.
	if len(model.Graph.Outputs) == 0 {
		return nil, errors.New("graph has no defined outputs")
	}

	outputNodeName := model.Graph.Outputs[0].Name

	outputNode, ok := instantiatedNodes[outputNodeName]
	if !ok {
		// Try to resolve output suffix for the output name
		resolvedOutputName := resolveOutputSuffix(outputNodeName, instantiatedNodes)

		outputNode, ok = instantiatedNodes[resolvedOutputName]
		if !ok {
			// For Gemma models, 'logits' typically maps to the last MatMul node (lm_head)
			if outputNodeName == "logits" {
				if lmHeadNode, exists := instantiatedNodes["/lm_head/MatMul"]; exists {
					outputNode = lmHeadNode
					ok = true
				}
			}
		}

		if !ok {
			return nil, fmt.Errorf("output node '%s' not found in instantiated nodes", outputNodeName)
		}
	}

	built, err := builder.Build(outputNode)
	if err != nil {
		return nil, err
	}

	// Wire KV cache feedback: link each present.N.{key,value} output back to
	// its corresponding past_key_values.N.{key,value} input node so the graph
	// automatically feeds KV state forward between decode steps.
	for _, outputInfo := range model.Graph.Outputs {
		name := outputInfo.Name
		if !strings.HasPrefix(name, "present.") {
			continue
		}
		// present.N.key -> past_key_values.N.key
		pastName := strings.Replace(name, "present.", "past_key_values.", 1)
		kvInput, inputOK := instantiatedNodes[pastName]
		kvOutput, outputOK := instantiatedNodes[name]
		if !inputOK || !outputOK {
			continue
		}
		if stateful, ok := kvInput.(graph.StatefulInputNode[T]); ok {
			built.AddKVPair(stateful, kvOutput)
			if debugONNX() {
				log.Printf("[DEBUG_ONNX] wired KV pair: %s -> %s", name, pastName)
			}
		}
	}

	// Optimization: fold Transpose nodes with constant inputs at load time.
	// This pre-applies weight transposes so they don't run on every forward pass.
	return graph.FoldConstantTransposes(built, engine)
}

// maskFromInputNode generates an all-ones attention mask covering the full
// sequence length (past cached tokens + current tokens). It tracks the
// accumulated sequence length internally, matching KV cache growth.
type maskFromInputNode[T tensor.Numeric] struct {
	pastLen int // accumulated past sequence length from prior forward passes
}

func (m *maskFromInputNode[T]) Reset()                           { m.pastLen = 0 }
func (m *maskFromInputNode[T]) OpType() string                  { return "AutoAttentionMask" }
func (m *maskFromInputNode[T]) Attributes() map[string]any       { return nil }
func (m *maskFromInputNode[T]) OutputShape() []int               { return nil }
func (m *maskFromInputNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (m *maskFromInputNode[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := inputs[0].Shape()
	curSeqLen := shape[len(shape)-1]
	totalLen := m.pastLen + curSeqLen
	batch := 1
	if len(shape) >= 2 {
		batch = shape[0]
	}
	maskShape := []int{batch, totalLen}
	data := make([]T, batch*totalLen)
	for i := range data {
		data[i] = T(1)
	}
	if debugONNX() {
		log.Printf("[DEBUG_ONNX] maskFromInputNode: pastLen=%d curSeqLen=%d totalLen=%d maskShape=%v", m.pastLen, curSeqLen, totalLen, maskShape)
	}
	m.pastLen += curSeqLen
	return tensor.New(maskShape, data)
}

func (m *maskFromInputNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// positionIdsNode generates sequential position IDs offset by the number of
// previously generated tokens. It tracks position via an internal counter
// that advances after each forward pass.
type positionIdsNode[T tensor.Numeric] struct {
	offset int // accumulated position offset from prior forward passes
}

func (p *positionIdsNode[T]) Reset()                           { p.offset = 0 }
func (p *positionIdsNode[T]) OpType() string                  { return "AutoPositionIds" }
func (p *positionIdsNode[T]) Attributes() map[string]any       { return nil }
func (p *positionIdsNode[T]) OutputShape() []int               { return nil }
func (p *positionIdsNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (p *positionIdsNode[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := inputs[0].Shape()
	size := inputs[0].Size()
	data := make([]T, size)
	seqLen := shape[len(shape)-1]
	for i := range data {
		data[i] = T(p.offset + i%seqLen)
	}
	if debugONNX() {
		log.Printf("[DEBUG_ONNX] positionIdsNode: input_shape=%v seqLen=%d offset=%d positions=%v", shape, seqLen, p.offset, data)
	}
	p.offset += seqLen
	return tensor.New(shape, data)
}

func (p *positionIdsNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// kvCacheIONode provides past KV cache state as a graph input and receives
// present KV cache state after each forward pass via SetStored. On the first
// call it returns an empty tensor; on subsequent calls it returns the
// previously stored present KV.
type kvCacheIONode[T tensor.Numeric] struct {
	numHeads int
	headDim  int
	stored   *tensor.TensorNumeric[T] // accumulated past KV from prior forward passes
}

func (z *kvCacheIONode[T]) Reset()                           { z.stored = nil }
func (z *kvCacheIONode[T]) OpType() string                  { return "AutoKVCacheIO" }
func (z *kvCacheIONode[T]) Attributes() map[string]any       { return nil }
func (z *kvCacheIONode[T]) OutputShape() []int               { return nil }
func (z *kvCacheIONode[T]) Parameters() []*graph.Parameter[T] { return nil }

// SetStored updates the stored KV tensor for the next forward pass.
// Implements graph.StatefulInputNode.
func (z *kvCacheIONode[T]) SetStored(t *tensor.TensorNumeric[T]) {
	z.stored = t
}

func (z *kvCacheIONode[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if z.stored != nil {
		if debugONNX() {
			log.Printf("[DEBUG_ONNX] kvCacheIONode: returning stored cache shape=%v", z.stored.Shape())
		}
		return z.stored, nil
	}
	batch := 1
	if len(inputs) > 0 {
		batch = inputs[0].Shape()[0]
	}
	shape := []int{batch, z.numHeads, 0, z.headDim}
	if debugONNX() {
		log.Printf("[DEBUG_ONNX] kvCacheIONode: returning empty cache shape=%v (first call)", shape)
	}
	return tensor.New(shape, []T{})
}

func (z *kvCacheIONode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Ensure kvCacheIONode implements graph.StatefulInputNode.
var _ graph.StatefulInputNode[float32] = (*kvCacheIONode[float32])(nil)

// parameterNode is a special node type for parameters that are referenced as inputs.
type parameterNode[T tensor.Numeric] struct {
	value *tensor.TensorNumeric[T]
}

func (p *parameterNode[T]) OpType() string {
	return "Parameter"
}

func (p *parameterNode[T]) Attributes() map[string]interface{} {
	return make(map[string]interface{})
}

func (p *parameterNode[T]) OutputShape() []int {
	return p.value.Shape()
}

func (p *parameterNode[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return p.value, nil
}

func (p *parameterNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Parameters don't propagate gradients to inputs since they have no inputs
	return nil, nil
}

func (p *parameterNode[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// resolveParam looks up a tensor value for an input name that is expected to be a
// constant or parameter (e.g. Unsqueeze axes, Reshape shape). It checks the params
// map first, then tries constant nodes in the instantiated map.
func resolveParam[T tensor.Numeric](name string, params map[string]*graph.Parameter[T], nodes map[string]graph.Node[T]) *tensor.TensorNumeric[T] {
	if p, ok := params[name]; ok {
		return p.Value
	}
	// Try as a constant node (parameterNode).
	if n, ok := nodes[name]; ok {
		if pn, isPn := n.(*parameterNode[T]); isPn {
			return pn.value
		}
	}
	return nil
}

// resolveOutputSuffix removes output suffixes like "/output_0", "/output_1", ":0", ":1" etc.
// and tries to find the actual node name in the nodeMap.
func resolveOutputSuffix[T tensor.Numeric](name string, nodeMap map[string]graph.Node[T]) string {
	// Try stripping common suffixes first, including numbered outputs.
	suffixes := []string{"/output_0", ":0", "/output_1", ":1", "/output_2", ":2", "/output_3", ":3"}
	for _, suffix := range suffixes {
		if strings.HasSuffix(name, suffix) {
			baseName := strings.TrimSuffix(name, suffix)
			if _, exists := nodeMap[baseName]; exists {
				return baseName
			}

			// For patterns like "/model/layers.0/input_layernorm/output_0",
			// try to find the actual layer node by appending common layer suffixes
			layerSuffixes := []string{"/LayerNorm", "/SimplifiedLayerNormalization", "/SkipLayerNorm", "/MatMul", "/Gather", "/Shape", "/Cast", "/Reshape", "/Mul", "/Sub", "/Add", "/Concat", "/Unsqueeze", "/FastGelu"}
			for _, layerSuffix := range layerSuffixes {
				candidateName := baseName + layerSuffix
				if _, exists := nodeMap[candidateName]; exists {
					return candidateName
				}
			}
		}
	}

	// Try common layer name variations (for backward compatibility).
	layerSuffixes := []string{"/LayerNorm", "/SimplifiedLayerNormalization", "/SkipLayerNorm"}
	for _, suffix := range layerSuffixes {
		if strings.HasSuffix(name, suffix) {
			baseName := strings.TrimSuffix(name, suffix)
			if _, exists := nodeMap[baseName]; exists {
				return baseName
			}
		}
	}

	return ""
}

// getNodeNames returns a slice of all node names for debugging.
func getNodeNames[T tensor.Numeric](nodes map[string]graph.Node[T]) []string {
	names := make([]string, 0, len(nodes))
	for name := range nodes {
		names = append(names, name)
	}

	return names
}

// convertParameters converts the ZMF Tensor map to a map of graph.Parameter.
func convertParameters[T tensor.Numeric](zmfParams map[string]*zmf.Tensor) (map[string]*graph.Parameter[T], error) {
	params := make(map[string]*graph.Parameter[T])

	for name, tensorProto := range zmfParams {
		tensorValue, err := DecodeTensor[T](tensorProto)
		if err != nil {
			return nil, fmt.Errorf("failed to decode tensor for parameter '%s': %w", name, err)
		}

		newTensorFn := tensor.New[T]

		param, err := graph.NewParameter[T](name, tensorValue, newTensorFn)
		if err != nil {
			return nil, fmt.Errorf("failed to create parameter '%s': %w", name, err)
		}

		params[name] = param
	}

	return params, nil
}

// buildConstantNode creates a parameterNode from a ZMF Constant op node.
// The node must have a "value" attribute of type *zmf.Attribute_Tensor.
func buildConstantNode[T tensor.Numeric](nodeProto *zmf.Node) (*parameterNode[T], error) {
	valueAttr, ok := nodeProto.Attributes["value"]
	if !ok {
		return nil, fmt.Errorf("constant node missing required 'value' attribute")
	}

	tensorAttr, ok := valueAttr.Value.(*zmf.Attribute_Tensor)
	if !ok {
		return nil, fmt.Errorf("constant node 'value' attribute has unexpected type %T", valueAttr.Value)
	}

	decoded, err := DecodeTensor[T](tensorAttr.Tensor)
	if err != nil {
		return nil, fmt.Errorf("failed to decode Constant tensor: %w", err)
	}

	return &parameterNode[T]{value: decoded}, nil
}

// rebuildWithPromotedAxes checks if a node has a constant-promoted attribute
// that should be interpreted as "axes" (for Unsqueeze). If found, it sets
// the "axes" key and rebuilds the node.
func rebuildWithPromotedAxes[T tensor.Numeric](
	nodeProto *zmf.Node,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	nodes map[string]graph.Node[T],
	currentNode graph.Node[T],
) graph.Node[T] {
	for k, attr := range nodeProto.Attributes {
		if !isConstantPromotedAttr(k) {
			continue
		}
		if v, ok := attr.Value.(*zmf.Attribute_Ints); ok {
			if nodeProto.Attributes == nil {
				nodeProto.Attributes = make(map[string]*zmf.Attribute)
			}
			nodeProto.Attributes["axes"] = &zmf.Attribute{
				Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: v.Ints.Val}},
			}
			updatedAttrs := convertAttributes(nodeProto.Attributes)
			rebuilt, rebuildErr := GetLayerBuilder[T](nodeProto.OpType)
			if rebuildErr == nil {
				node, nodeErr := rebuilt(engine, ops, nodeProto.Name, nil, updatedAttrs)
				if nodeErr == nil {
					nodes[nodeProto.Name] = node
					for _, outName := range nodeProto.Outputs {
						if outName != "" && outName != nodeProto.Name {
							nodes[outName] = node
						}
					}
					return node
				}
			}
		}
	}
	return currentNode
}

// isConstantPromotedAttr returns true if the attribute key looks like an ONNX
// output name that was promoted from a Constant node input by the zonnx converter.
// Standard attributes have short names ("axis", "epsilon", "perm") while
// promoted constants have names like "/Constant_output_0" or "onnx::Gather_919".
func isConstantPromotedAttr(key string) bool {
	if strings.HasPrefix(key, "/") || strings.HasPrefix(key, "onnx::") {
		return true
	}
	return false
}

// convertAttributes converts ZMF attributes to a more usable map[string]interface{}.
func convertAttributes(zmfAttributes map[string]*zmf.Attribute) map[string]interface{} {
	attributes := make(map[string]interface{})

	for name, attr := range zmfAttributes {
		switch v := attr.Value.(type) {
		case *zmf.Attribute_F:
			attributes[name] = v.F
		case *zmf.Attribute_I:
			attributes[name] = int(v.I) // Cast to int for convenience
		case *zmf.Attribute_S:
			attributes[name] = v.S
		case *zmf.Attribute_Ints:
			// Convert int64 array to int array for convenience
			intValues := make([]int64, len(v.Ints.Val))
			copy(intValues, v.Ints.Val)
			attributes[name] = intValues
		case *zmf.Attribute_Floats:
			// Convert float array
			floatValues := make([]float32, len(v.Floats.Val))
			copy(floatValues, v.Floats.Val)
			attributes[name] = floatValues
		case *zmf.Attribute_Strings:
			// Convert string array
			stringValues := make([]string, len(v.Strings.Val))
			copy(stringValues, v.Strings.Val)
			attributes[name] = stringValues
		case *zmf.Attribute_Tensor:
			// Store raw ZMF tensor proto; callers that need a typed tensor (e.g. Constant
			// node handling in BuildFromZMF) can call DecodeTensor themselves.
			attributes[name] = v.Tensor
		}
	}

	return attributes
}
