// Package core provides core neural network layer implementations.
package core

import (
	"errors"
	"fmt"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func init() {
	model.RegisterLayer("FFN", buildFFN[float16.Float16])
	// Add registrations for other supported types if needed.
}

func buildFFN[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	// Extract attributes
	inputDim, ok := attributes["input_dim"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: input_dim")
	}
	hiddenDim, ok := attributes["hidden_dim"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: hidden_dim")
	}
	outputDim, ok := attributes["output_dim"].(int)
	if !ok {
		return nil, errors.New("missing or invalid attribute: output_dim")
	}
	withBias, ok := attributes["with_bias"].(bool)
	if !ok {
		withBias = true // Default to true
	}

	// FFN is composed of Dense layers, but the ZMF will provide the flattened parameters.
	// We need to reconstruct the Dense layers from these parameters.
	w1, ok := params[name+"_w1_weights"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_w1_weights", name)
	}
	w2, ok := params[name+"_w2_weights"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_w2_weights", name)
	}
	w3, ok := params[name+"_w3_weights"]
	if !ok {
		return nil, fmt.Errorf("missing required parameter: %s_w3_weights", name)
	}

	var opts []FFNOption[T]
	if !withBias {
		opts = append(opts, WithFFNBias[T](false))
	}

	ffn, err := NewFFN[T](name, engine, ops, inputDim, hiddenDim, outputDim, opts...)
	if err != nil {
		return nil, err
	}

	// Manually set the weights of the dense layers inside FFN
	ffn.w1.linear.weights = w1
	ffn.w2.linear.weights = w2
	ffn.w3.linear.weights = w3

	if withBias {
		w1Bias, ok := params[name+"_w1_biases"]
		if !ok {
			return nil, fmt.Errorf("missing required parameter: %s_w1_biases", name)
		}
		w2Bias, ok := params[name+"_w2_biases"]
		if !ok {
			return nil, fmt.Errorf("missing required parameter: %s_w2_biases", name)
		}
		w3Bias, ok := params[name+"_w3_biases"]
		if !ok {
			return nil, fmt.Errorf("missing required parameter: %s_w3_biases", name)
		}
		ffn.w1.bias.biases = w1Bias
		ffn.w2.bias.biases = w2Bias
		ffn.w3.bias.biases = w3Bias
	}

	return ffn, nil
}
