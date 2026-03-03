package training

import (
	"testing"
)

func TestNewDefaultBackpropStrategy(t *testing.T) {
	s := NewDefaultBackpropStrategy[float32]()
	if s == nil {
		t.Fatal("NewDefaultBackpropStrategy returned nil")
	}
}

func TestNewOneStepApproximationStrategy(t *testing.T) {
	s := NewOneStepApproximationStrategy[float32]()
	if s == nil {
		t.Fatal("NewOneStepApproximationStrategy returned nil")
	}
}

func TestStrategiesImplementGradientStrategy(t *testing.T) {
	var _ GradientStrategy[float32] = (*DefaultBackpropStrategy[float32])(nil)
	var _ GradientStrategy[float32] = (*OneStepApproximationStrategy[float32])(nil)
}
