package components

import (
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
)

func TestXavierInitializer(t *testing.T) {
	ops := numeric.Float32Ops{}
	initializer := NewXavierInitializer(ops)

	inputSize, outputSize := 10, 5
	weights, err := initializer.Initialize(inputSize, outputSize)
	if err != nil {
		t.Fatalf("Xavier initialization failed: %v", err)
	}

	if len(weights) != inputSize*outputSize {
		t.Errorf("Expected %d weights, got %d", inputSize*outputSize, len(weights))
	}

	// Check that weights are within expected range
	fanIn := float64(inputSize)
	fanOut := float64(outputSize)
	expectedLimit := math.Sqrt(6.0 / (fanIn + fanOut))

	for i, weight := range weights {
		val := math.Abs(float64(weight))
		if val > expectedLimit {
			t.Errorf("Weight %d (%f) exceeds Xavier limit (%f)", i, val, expectedLimit)
		}
	}
}

func TestHeInitializer(t *testing.T) {
	ops := numeric.Float32Ops{}
	initializer := NewHeInitializer(ops)

	inputSize, outputSize := 10, 5
	weights, err := initializer.Initialize(inputSize, outputSize)
	if err != nil {
		t.Fatalf("He initialization failed: %v", err)
	}

	if len(weights) != inputSize*outputSize {
		t.Errorf("Expected %d weights, got %d", inputSize*outputSize, len(weights))
	}

	// Check that weights have reasonable variance (not all zeros)
	var sum, sumSquares float64
	for _, weight := range weights {
		val := float64(weight)
		sum += val
		sumSquares += val * val
	}

	mean := sum / float64(len(weights))
	variance := (sumSquares / float64(len(weights))) - (mean * mean)

	// He initialization should have non-zero variance
	if variance < 1e-6 {
		t.Errorf("He initialization produced weights with too low variance: %f", variance)
	}
}

func TestUniformInitializer(t *testing.T) {
	ops := numeric.Float32Ops{}
	scale := 0.1
					initializer := NewUniformInitializer(ops, WithScale[float32](float64(0.1)))

	inputSize, outputSize := 10, 5
	weights, err := initializer.Initialize(inputSize, outputSize)
	if err != nil {
		t.Fatalf("Uniform initialization failed: %v", err)
	}

	if len(weights) != inputSize*outputSize {
		t.Errorf("Expected %d weights, got %d", inputSize*outputSize, len(weights))
	}

	// Check that all weights are within [-scale, scale]
	for i, weight := range weights {
		val := math.Abs(float64(weight))
		if val > scale {
			t.Errorf("Weight %d (%f) exceeds uniform scale (%f)", i, val, scale)
		}
	}
}

func TestInitializerConsistency(t *testing.T) {
	ops := numeric.Float32Ops{}

	// Test that different initializers produce different distributions
	xavier := NewXavierInitializer(ops)
	he := NewHeInitializer(ops)
	uniform := NewUniformInitializer(ops, WithScale[float32](0.1))

	inputSize, outputSize := 100, 50

	xavierWeights, _ := xavier.Initialize(inputSize, outputSize)
	heWeights, _ := he.Initialize(inputSize, outputSize)
	uniformWeights, _ := uniform.Initialize(inputSize, outputSize)

	// Calculate variances
	calcVariance := func(weights []float32) float64 {
		var sum, sumSquares float64
		for _, w := range weights {
			val := float64(w)
			sum += val
			sumSquares += val * val
		}
		mean := sum / float64(len(weights))

		return (sumSquares / float64(len(weights))) - (mean * mean)
	}

	xavierVar := calcVariance(xavierWeights)
	heVar := calcVariance(heWeights)
	uniformVar := calcVariance(uniformWeights)

	// He initialization should generally have higher variance than Xavier for this case
	if heVar <= xavierVar {
		t.Logf("He variance (%f) should typically be higher than Xavier variance (%f) for this configuration", heVar, xavierVar)
	}

	// All should have reasonable variance
	if xavierVar < 1e-6 || heVar < 1e-6 || uniformVar < 1e-6 {
		t.Errorf("One or more initializers produced weights with too low variance: Xavier=%f, He=%f, Uniform=%f", xavierVar, heVar, uniformVar)
	}
}
