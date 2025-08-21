package components

import (
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestNewUniformInitializer_FunctionalOptions(t *testing.T) {
	ops := numeric.Float32Ops{}
	scale := 0.05

	// Test with functional options
	initializer := NewUniformInitializer(
		ops,
		WithScale[float32](scale),
	)

	testutils.AssertNotNil(t, initializer, "expected UniformInitializer to not be nil")

	// Test initialization
	inputSize := 10
	outputSize := 5
	weights, err := initializer.Initialize(inputSize, outputSize)
	testutils.AssertNoError(t, err, "failed to initialize weights")
	testutils.AssertEqual(t, inputSize*outputSize, len(weights), "weight slice length mismatch")
}
