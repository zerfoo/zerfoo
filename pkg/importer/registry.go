package importer

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zmf"
	"github.com/zerfoo/zerfoo/tensor"
)

func init() {
	Register[float32]("RMSNorm", newRMSNorm)
	Register[float32]("Dense", newDense)
	Register[float32]("ReLU", newReLU)
	Register[float32]("Sigmoid", newSigmoid)
	Register[float32]("SwiGLU", newSwiGLU)
	Register[float32]("Tanh", newTanh)
	Register[float32]("GlobalAttention", newGlobalAttention)
	// We can register for float16 and other types here as well if needed
}

// newGlobalAttention is the LayerConstructor for the GlobalAttention layer.
func newGlobalAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *zmf.Node,
	params map[string]*graph.Parameter[T],
) (graph.Node[T], error) {
	// 1. Get attributes
	modelDimAttr, ok := node.Attributes["model_dim"]
	if !ok {
		return nil, fmt.Errorf("missing attribute 'model_dim' for GlobalAttention")
	}
	modelDim := int(modelDimAttr.GetI())

	numQueryHeadsAttr, ok := node.Attributes["num_query_heads"]
	if !ok {
		return nil, fmt.Errorf("missing attribute 'num_query_heads' for GlobalAttention")
	}
	numQueryHeads := int(numQueryHeadsAttr.GetI())

	numKeyValueHeadsAttr, ok := node.Attributes["num_key_value_heads"]
	if !ok {
		return nil, fmt.Errorf("missing attribute 'num_key_value_heads' for GlobalAttention")
	}
	numKeyValueHeads := int(numKeyValueHeadsAttr.GetI())

	baseAttr, ok := node.Attributes["base"]
	if !ok {
		return nil, fmt.Errorf("missing attribute 'base' for GlobalAttention")
	}
	base := float64(baseAttr.GetF())

	maxSeqLenAttr, ok := node.Attributes["max_seq_len"]
	if !ok {
		return nil, fmt.Errorf("missing attribute 'max_seq_len' for GlobalAttention")
	}
	maxSeqLen := int(maxSeqLenAttr.GetI())

	// 2. Get parameters for Dense layers
	wqWeightsParam, ok := params[node.Inputs[0]]
	if !ok {
		return nil, fmt.Errorf("wq_weights parameter not found for GlobalAttention")
	}
	wqBiasParam, ok := params[node.Inputs[1]]
	if !ok {
		return nil, fmt.Errorf("wq_bias parameter not found for GlobalAttention")
	}
	wkWeightsParam, ok := params[node.Inputs[2]]
	if !ok {
		return nil, fmt.Errorf("wk_weights parameter not found for GlobalAttention")
	}
	wkBiasParam, ok := params[node.Inputs[3]]
	if !ok {
		return nil, fmt.Errorf("wk_bias parameter not found for GlobalAttention")
	}
	wvWeightsParam, ok := params[node.Inputs[4]]
	if !ok {
		return nil, fmt.Errorf("wv_weights parameter not found for GlobalAttention")
	}
	wvBiasParam, ok := params[node.Inputs[5]]
	if !ok {
		return nil, fmt.Errorf("wv_bias parameter not found for GlobalAttention")
	}
	woWeightsParam, ok := params[node.Inputs[6]]
	if !ok {
		return nil, fmt.Errorf("wo_weights parameter not found for GlobalAttention")
	}
	woBiasParam, ok := params[node.Inputs[7]]
	if !ok {
		return nil, fmt.Errorf("wo_bias parameter not found for GlobalAttention")
	}

	// 3. Create Dense layers
	wq := core.NewDenseFromParams[T](core.NewLinearFromParam[T](engine, wqWeightsParam), core.NewBiasFromParam[T](engine, ops, wqBiasParam))
	wk := core.NewDenseFromParams[T](core.NewLinearFromParam[T](engine, wkWeightsParam), core.NewBiasFromParam[T](engine, ops, wkBiasParam))
	wv := core.NewDenseFromParams[T](core.NewLinearFromParam[T](engine, wvWeightsParam), core.NewBiasFromParam[T](engine, ops, wvBiasParam))
	wo := core.NewDenseFromParams[T](core.NewLinearFromParam[T](engine, woWeightsParam), core.NewBiasFromParam[T](engine, ops, woBiasParam))

	// 4. Create RoPE
	rope, err := embeddings.NewRotaryPositionalEmbedding[T](context.Background(), engine, modelDim/numQueryHeads, maxSeqLen, base)
	if err != nil {
		return nil, fmt.Errorf("failed to create RotaryPositionalEmbedding: %w", err)
	}

	// 5. Create GroupedQueryAttention
	gqa, err := attention.NewGroupedQueryAttentionFromParams[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, wq, wk, wv, wo, rope)
	if err != nil {
		return nil, fmt.Errorf("failed to create GroupedQueryAttention: %w", err)
	}

	// 6. Create GlobalAttention
	layer := attention.NewGlobalAttentionFromParams[T](gqa)
	return layer, nil
}

// newTokenEmbedding is the LayerConstructor for the TokenEmbedding layer.
func newTokenEmbedding[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *zmf.Node,
	params map[string]*graph.Parameter[T],
) (*embeddings.TokenEmbedding[T], error) {
	// 1. Get parameters
	embeddingTableParamName := node.Inputs[0] // Assuming the embedding table is the first input
	embeddingTableParam, ok := params[embeddingTableParamName]
	if !ok {
		return nil, fmt.Errorf("parameter %s not found for TokenEmbedding", embeddingTableParamName)
	}

	// 2. Create the layer
	layer, err := embeddings.NewTokenEmbeddingFromParam[T](engine, embeddingTableParam)
	if err != nil {
		return nil, err
	}

	return layer, nil
}

// newSwiGLU is the LayerConstructor for the SwiGLU activation.
func newSwiGLU[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *zmf.Node,
	params map[string]*graph.Parameter[T],
) (graph.Node[T], error) {
	return activations.NewSwiGLU(engine, ops), nil
}

// newTanh is the LayerConstructor for the Tanh activation.
func newTanh[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *zmf.Node,
	params map[string]*graph.Parameter[T],
) (graph.Node[T], error) {
	return activations.NewTanh(engine, ops), nil
}

// newReLU is the LayerConstructor for the ReLU activation.
func newReLU[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *zmf.Node,
	params map[string]*graph.Parameter[T],
) (graph.Node[T], error) {
	return activations.NewReLU(engine, ops), nil
}

// newSigmoid is the LayerConstructor for the Sigmoid activation.
func newSigmoid[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *zmf.Node,
	params map[string]*graph.Parameter[T],
) (graph.Node[T], error) {
	return activations.NewSigmoid(engine, ops), nil
}

// newDense is the LayerConstructor for the Dense layer.
func newDense[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *zmf.Node,
	params map[string]*graph.Parameter[T],
) (graph.Node[T], error) {
	// 1. Get parameters
	weightsParamName := node.Inputs[0] // Assuming weights are the first input
	biasParamName := node.Inputs[1]    // Assuming bias is the second input

	weightsParam, ok := params[weightsParamName]
	if !ok {
		return nil, fmt.Errorf("weights parameter %s not found for Dense", weightsParamName)
	}
	biasParam, ok := params[biasParamName]
	if !ok {
		return nil, fmt.Errorf("bias parameter %s not found for Dense", biasParamName)
	}

	// 2. Create the sub-layers
	linearLayer := core.NewLinearFromParam[T](engine, weightsParam)
	biasLayer := core.NewBiasFromParam[T](engine, ops, biasParam)

	// 3. Create the Dense layer
	layer := core.NewDenseFromParams[T](linearLayer, biasLayer)
	layer.SetName(node.Name)
	return layer, nil
}

// newRMSNorm is the LayerConstructor for the RMSNorm layer.
func newRMSNorm[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	node *zmf.Node,
	params map[string]*graph.Parameter[T],
) (graph.Node[T], error) {
	// 1. Get attributes
	epsilonAttr, ok := node.Attributes["epsilon"]
	if !ok {
		return nil, fmt.Errorf("missing attribute 'epsilon' for RMSNorm")
	}
	epsilon := T(epsilonAttr.GetF())

	// 2. Get parameters
	gainParamName := node.Inputs[0] // Assuming the first input is the gain parameter
	gainParam, ok := params[gainParamName]
	if !ok {
		return nil, fmt.Errorf("parameter %s not found for RMSNorm", gainParamName)
	}

	// 3. Create the layer
	layer, err := normalization.NewRMSNormFromParam[T](engine, ops, epsilon, gainParam)
	if err != nil {
		return nil, err
	}

	layer.SetName(node.Name)
	return layer, nil
}
