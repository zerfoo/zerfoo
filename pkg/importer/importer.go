package importer

import (
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/pkg/format"
	"github.com/zerfoo/zerfoo/tensor"
	"google.golang.org/protobuf/proto"
)

// LayerConstructor defines a function that creates a graph.Node (a layer)
// from a format.Node definition and a set of loaded parameters.
type LayerConstructor[T tensor.Numeric] func(
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *format.Node,
	params map[string]*graph.Parameter[T],
) (graph.Node[T], error)

// registry holds the mapping from ONNX/ZMF op_types to our layer constructors.
var registry = make(map[string]any)

// Register adds a new layer constructor to the registry.
func Register[T tensor.Numeric](opType string, constructor LayerConstructor[T]) {
	registry[opType] = constructor
}

// LoadModel reads a .zmf file from the given path and constructs a Gemma model.
// This is a placeholder and will need to be made more generic.
func LoadModel[T tensor.Numeric](
	path string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
) (*model.Model[T], error) {
	// 1. Read the file from disk
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %w", err)
	}

	// 2. Unmarshal the protobuf data
	zmfModel := &format.Model{}
	if err := proto.Unmarshal(data, zmfModel); err != nil {
		return nil, fmt.Errorf("failed to unmarshal model protobuf: %w", err)
	}

	// 3. Load parameters into graph.Parameter objects
	loadedParams, err := loadParameters[T](zmfModel.Graph.Parameters)
	if err != nil {
		return nil, err
	}

	// 4. Build the model
	var embedding *embeddings.TokenEmbedding[T]
	builder := graph.NewBuilder[T](engine)
	nodeMap := make(map[string]graph.Node[T])

	for _, nodeDef := range zmfModel.Graph.Nodes {
		if nodeDef.OpType == "TokenEmbedding" {
			// Handle the embedding layer separately
			var err error
			embedding, err = newTokenEmbedding[T](engine, ops, nodeDef, loadedParams)
			if err != nil {
				return nil, fmt.Errorf("failed to construct embedding: %w", err)
			}
		} else {
			constructorAny, ok := registry[nodeDef.OpType]
			if !ok {
				return nil, fmt.Errorf("unknown op_type: %s", nodeDef.OpType)
			}

			constructor, ok := constructorAny.(LayerConstructor[T])
			if !ok {
				return nil, fmt.Errorf("invalid constructor type for op_type: %s", nodeDef.OpType)
			}

			node, err := constructor(engine, ops, nodeDef, loadedParams)
			if err != nil {
				return nil, fmt.Errorf("failed to construct node %s (type %s): %w", nodeDef.Name, nodeDef.OpType, err)
			}

			var inputs []graph.Node[T]
			for _, inputName := range nodeDef.Inputs {
				if inputNode, exists := nodeMap[inputName]; exists {
					inputs = append(inputs, inputNode)
				}
			}

			builder.AddNode(node, inputs...)
			nodeMap[nodeDef.Name] = node
		}
	}

	// This is a simplification. We need to identify the correct output node.
	outputNode := nodeMap[zmfModel.Graph.Nodes[len(zmfModel.Graph.Nodes)-1].Name]
	graph, err := builder.Build(outputNode)
	if err != nil {
		return nil, fmt.Errorf("failed to build graph: %w", err)
	}

	return model.NewModel(embedding, graph), nil
}

// loadParameters converts the raw tensor data from the .zmf file into
// the framework's graph.Parameter format.
func loadParameters[T tensor.Numeric](paramData map[string]*format.Tensor) (map[string]*graph.Parameter[T], error) {
	params := make(map[string]*graph.Parameter[T], len(paramData))
	for name, tenData := range paramData {
		// This is a simplification. We need to handle different data types.
		if tenData.Dtype != format.Tensor_FLOAT32 {
			return nil, fmt.Errorf("unsupported data type for parameter %s: %v", name, tenData.Dtype)
		}

		// Create the tensor
		t, err := tensor.NewFromBytes[T](tenData.Shape, tenData.Data)
		if err != nil {
			return nil, fmt.Errorf("failed to create tensor for parameter %s: %w", name, err)
		}

		// Create the graph parameter
		p, err := graph.NewParameter[T](name, t, tensor.New[T])
		if err != nil {
			return nil, fmt.Errorf("failed to create graph parameter for %s: %w", name, err)
		}
		params[name] = p
	}
	return params, nil
}
