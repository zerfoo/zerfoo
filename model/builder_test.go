// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/zmf"
)

// TestBuildFromZMF_SingleNode tests building a simple model with one RMSNorm node.
func TestBuildFromZMF_SingleNode(t *testing.T) {
	// 1. Setup: Create engine, ops, and a sample ZMF model.
	engine := compute.NewCPUEngine[tensor.Float16]()
	ops := numeric.NewFloat16Ops()

	// Create a sample gain tensor and encode it.
	gainTensor, _ := tensor.New[tensor.Float16]([]int{128}, nil)
	encodedGain, err := EncodeTensor(gainTensor)
	if err != nil {
		t.Fatalf("Failed to encode gain tensor: %v", err)
	}

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"norm_gain": encodedGain,
			},
			Nodes: []*zmf.Node{
				{
					Name:   "norm",
					OpType: "RMSNorm",
					Attributes: map[string]*zmf.Attribute{
						"epsilon": {Value: &zmf.Attribute_F{F: 1e-5}},
					},
				},
			},
		},
	}

	// 2. Call the function to be tested.
	model, err := BuildFromZMF[tensor.Float16](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}

	// 3. Verify the result.
	if len(model) != 1 {
		t.Fatalf("Expected 1 node in the model, got %d", len(model))
	}

	node, ok := model["norm"]
	if !ok {
		t.Fatal("Model is missing the 'norm' node")
	}

	if _, ok := node.(*normalization.RMSNorm[tensor.Float16]); !ok {
		t.Errorf("Expected node 'norm' to be of type *RMSNorm, but got %T", node)
	}
}
