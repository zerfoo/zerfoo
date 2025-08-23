// Package transpose provides the Transpose layer for the Zerfoo ML framework.
package transpose

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestBuildTranspose(t *testing.T) {
	engine := compute.NewTestableEngine[float32](numeric.Float32Ops{})

	testCases := []struct {
		name           string
		attributes     map[string]interface{}
		expectedAxes   []int
		expectBuildErr bool
	}{
		{
			name: "With perm attribute as []any",
			attributes: map[string]interface{}{
				"perm": []any{int64(0), int64(2), int64(1)},
			},
			expectedAxes: []int{0, 2, 1},
		},
		{
			name: "With perm attribute as []int64",
			attributes: map[string]interface{}{
				"perm": []int64{0, 2, 1},
			},
			expectedAxes: []int{0, 2, 1},
		},
		{
			name:           "Without perm attribute",
			attributes:     map[string]interface{}{},
			expectedAxes:   nil,
			expectBuildErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			node, err := BuildTranspose[float32](engine, nil, "", nil, tc.attributes)
			if (err != nil) != tc.expectBuildErr {
				t.Fatalf("BuildTranspose() error = %v, expectBuildErr %v", err, tc.expectBuildErr)
			}

			if err != nil {
				return
			}

			transposeLayer, ok := node.(*Transpose[float32])
			if !ok {
				t.Fatalf("BuildTranspose() did not return a *Transpose layer")
			}

			if !reflect.DeepEqual(transposeLayer.perm, tc.expectedAxes) {
				t.Errorf("Expected axes to be %v, but got %v", tc.expectedAxes, transposeLayer.perm)
			}
		})
	}
}

func TestTransposeForward(t *testing.T) {
	engine := compute.NewTestableEngine[float32](numeric.Float32Ops{})

	inputTensor, err := tensor.New[float32]([]int{2, 3, 4}, nil)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	testCases := []struct {
		name          string
		axes          []int
		expectedShape []int
	}{
		{
			name:          "With axes",
			axes:          []int{0, 2, 1},
			expectedShape: []int{2, 4, 3},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			layer := New(engine, tc.axes)

			output, err := layer.Forward(context.Background(), inputTensor)
			if err != nil {
				t.Fatalf("Forward() error = %v", err)
			}

			if !reflect.DeepEqual(output.Shape(), tc.expectedShape) {
				t.Errorf("Expected output shape to be %v, but got %v", tc.expectedShape, output.Shape())
			}
		})
	}
}
