// Package types contains shared, fundamental types for the Zerfoo framework.
package types

// BackwardMode is an enum for specifying the backward pass behavior.
type BackwardMode int

const (
	// FullBackprop performs a standard, full backpropagation.
	FullBackprop BackwardMode = iota
	// OneStepApproximation performs a one-step gradient approximation for recurrent models.
	OneStepApproximation
)
