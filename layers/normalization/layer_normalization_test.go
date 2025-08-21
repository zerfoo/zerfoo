package normalization

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

// TestLayerNormalization_WithEpsilon tests LayerNormalization with custom epsilon option
func TestLayerNormalization_WithEpsilon(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	// Test with custom epsilon value
	customEpsilon := float32(1e-6)
	ln, err := NewLayerNormalization[float32](engine, 4, WithLayerNormEpsilon[float32](customEpsilon))
	testutils.AssertNoError(t, err, "NewLayerNormalization with custom epsilon should not return an error")
	testutils.AssertNotNil(t, ln, "LayerNormalization should not be nil")

	// Check that epsilon is set correctly
	testutils.AssertFloatEqual(t, customEpsilon, ln.epsilon, float32(1e-9), "Epsilon should be set to custom value")
}

// TestLayerNormalization_DefaultEpsilon tests LayerNormalization with default epsilon
func TestLayerNormalization_DefaultEpsilon(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	// Test with default epsilon (no options)
	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization with default epsilon should not return an error")
	testutils.AssertNotNil(t, ln, "LayerNormalization should not be nil")

	// Check that epsilon is set to default value (1e-5)
	expectedEpsilon := float32(1e-5)
	testutils.AssertFloatEqual(t, expectedEpsilon, ln.epsilon, float32(1e-9), "Epsilon should be set to default value")
}

// TestLayerNormalization_Parameters tests that LayerNormalization returns correct parameters
func TestLayerNormalization_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	params := ln.Parameters()
	testutils.AssertEqual(t, len(params), 2, "LayerNormalization should have 2 parameters (gamma and beta)")
}
