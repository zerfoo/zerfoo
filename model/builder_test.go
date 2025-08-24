package model

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zmf"
)

func TestBuildFromZMF_Int8(t *testing.T) {
	// Create a sample ZMF Model protobuf message with an INT8 parameter.
	sampleModel := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"param1": {
					Dtype: zmf.Tensor_INT8,
					Shape: []int64{2, 2},
					Data:  []byte{1, 2, 3, 4},
				},
			},
			Inputs: []*zmf.ValueInfo{
				{
					Name:  "input",
					Dtype: zmf.Tensor_INT8,
					Shape: []int64{1, 10},
				},
			},
			Outputs: []*zmf.ValueInfo{
				{
					Name:  "input", // just pass through for simplicity
					Dtype: zmf.Tensor_INT8,
					Shape: []int64{1, 10},
				},
			},
		},
	}

	// Create an int8 engine.
	ops := numeric.Int8Ops{}
	engine := compute.NewCPUEngine[int8](ops)

	// Build the graph.
	graph, err := BuildFromZMF[int8](engine, ops, sampleModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}

	if graph == nil {
		t.Fatal("BuildFromZMF returned a nil graph")
	}
}
