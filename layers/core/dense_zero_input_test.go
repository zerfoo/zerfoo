package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestDense_Backward_ZeroOutputGradient(t *testing.T) {
	ops := numeric.Float32Ops{}
	testutils.RunTests(t, []testutils.TestCase{
		{
			Name: "Zero output gradient with bias",
			Func: func(t *testing.T) {
				engine := compute.NewCPUEngine[float32](ops)

				inputSize := 4
				outputSize := 3
				batchSize := 2

				dense, err := NewDense[float32](
					"test_dense",
					engine,
					ops,
					inputSize,
					outputSize,
					WithBias[float32](true),
				)
				testutils.AssertNil(t, err, "NewDense should not return an error")
				testutils.AssertNotNil(t, dense, "Dense layer should not be nil")

				// Create a zero-sized output gradient
				outputGradient, err := tensor.New[float32]([]int{batchSize, outputSize}, make([]float32, batchSize*outputSize))
				testutils.AssertNil(t, err, "New tensor for outputGradient should not return an error")
				testutils.AssertNotNil(t, outputGradient, "Output gradient tensor should not be nil")

				inputTensor, err := tensor.New[float32]([]int{batchSize, inputSize}, make([]float32, batchSize*inputSize))
				testutils.AssertNil(t, err, "New tensor for inputTensor should not return an error")

				// This call is expected to panic
				_, err = dense.Backward(context.Background(), outputGradient, inputTensor)
				testutils.AssertNotNil(t, err, "Backward should return an error, not panic") // Expecting an error, not a panic
			},
		},
	})
}