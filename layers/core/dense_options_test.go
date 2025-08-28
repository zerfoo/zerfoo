package core

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestNewDense_WithFunctionalOptions(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// This test will initially fail because NewDense does not yet accept options.
	layer, err := NewDense[float32](
		"test_layer",
		engine,
		ops,
		10,
		5,
		WithoutBias[float32](),
	)
	testutils.AssertNoError(t, err, "expected no error when creating dense layer with options, got %v")
	testutils.AssertNotNil(t, layer, "expected layer to not be nil")
	testutils.AssertNil(t, layer.bias, "expected bias to be nil when WithBias(false) is used")
}
