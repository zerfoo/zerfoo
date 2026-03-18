package online

import "math"

// ModelSnapshot captures model state for comparison during validation.
type ModelSnapshot struct {
	// Weights maps layer names to their weight tensors as flat float32 slices.
	Weights map[string][]float32
	// Loss is the model loss on a validation set.
	Loss float64
}

// ValidationResult is the outcome of a safety validation check.
type ValidationResult struct {
	// Pass is true if the validation passed.
	Pass bool
	// Reason describes why validation failed (empty when Pass is true).
	Reason string
}

// Validator checks whether a model update is safe to promote.
type Validator interface {
	// Validate compares the model state before and after an update and
	// returns whether the update should be accepted.
	Validate(before, after ModelSnapshot) ValidationResult
}

// ValidationConfig holds thresholds for the built-in validators.
type ValidationConfig struct {
	// MaxLossDelta is the maximum allowed increase in loss (after - before).
	MaxLossDelta float64
	// MaxWeightNorm is the maximum allowed L2 norm for any single weight tensor.
	MaxWeightNorm float64
	// MaxGradNorm is the maximum allowed gradient norm (reserved for future use).
	MaxGradNorm float64
}

// LossDeltaValidator rejects updates where the loss increases by more than
// MaxLossDelta.
type LossDeltaValidator struct {
	MaxLossDelta float64
}

// NewLossDeltaValidator returns a LossDeltaValidator with the given threshold.
func NewLossDeltaValidator(maxDelta float64) *LossDeltaValidator {
	return &LossDeltaValidator{MaxLossDelta: maxDelta}
}

// Validate rejects the update if after.Loss - before.Loss > MaxLossDelta.
func (v *LossDeltaValidator) Validate(before, after ModelSnapshot) ValidationResult {
	delta := after.Loss - before.Loss
	if delta > v.MaxLossDelta {
		return ValidationResult{
			Pass:   false,
			Reason: "loss increased beyond threshold",
		}
	}
	return ValidationResult{Pass: true}
}

// WeightNormValidator rejects updates where the L2 norm of any weight tensor
// in the updated model exceeds MaxWeightNorm.
type WeightNormValidator struct {
	MaxWeightNorm float64
}

// NewWeightNormValidator returns a WeightNormValidator with the given threshold.
func NewWeightNormValidator(maxNorm float64) *WeightNormValidator {
	return &WeightNormValidator{MaxWeightNorm: maxNorm}
}

// Validate rejects the update if any weight tensor in after has an L2 norm
// exceeding MaxWeightNorm.
func (v *WeightNormValidator) Validate(_, after ModelSnapshot) ValidationResult {
	for name, weights := range after.Weights {
		norm := l2Norm(weights)
		if norm > v.MaxWeightNorm {
			return ValidationResult{
				Pass:   false,
				Reason: "weight norm exceeded threshold for " + name,
			}
		}
	}
	return ValidationResult{Pass: true}
}

// CompositeValidator runs multiple validators and returns the first failure.
type CompositeValidator struct {
	Validators []Validator
}

// NewCompositeValidator returns a CompositeValidator that runs all provided
// validators in order.
func NewCompositeValidator(validators ...Validator) *CompositeValidator {
	return &CompositeValidator{Validators: validators}
}

// Validate runs each validator in order and returns the first failure.
// If all validators pass, it returns a passing result.
func (c *CompositeValidator) Validate(before, after ModelSnapshot) ValidationResult {
	for _, v := range c.Validators {
		result := v.Validate(before, after)
		if !result.Pass {
			return result
		}
	}
	return ValidationResult{Pass: true}
}

func l2Norm(vals []float32) float64 {
	var sum float64
	for _, v := range vals {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum)
}
