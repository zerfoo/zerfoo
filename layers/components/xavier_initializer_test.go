package components

import (
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestNewXavierInitializer_FunctionalOptions(t *testing.T) {
	ops := numeric.Float32Ops{}

	// Test with functional options (even if no specific options are defined yet)
	initializer := NewXavierInitializer(
		ops,
		// No specific options to pass yet, but demonstrating the pattern
	)

	testutils.AssertNotNil(t, initializer, "expected XavierInitializer to not be nil")

	// Test initialization
	inputSize := 10
	outputSize := 5
	weights, err := initializer.Initialize(inputSize, outputSize)
	testutils.AssertNoError(t, err, "failed to initialize weights")
	testutils.AssertEqual(t, inputSize*outputSize, len(weights), "weight slice length mismatch")
}
