package synth

import (
	"context"
	"math"
	"math/rand"

	"github.com/zerfoo/ztensor/tensor"
)

// CrashConfig controls the behavior of CrashGenerator.
type CrashConfig struct {
	// Severity controls the magnitude of extreme scenarios on a 1.0-10.0 scale.
	// Higher values produce more extreme tail events.
	Severity float64

	// Duration is the number of timesteps per generated crash scenario.
	Duration int

	// CorrelationSpike controls how much inter-asset correlations increase
	// during the crash. A value of 0 means no additional correlation; higher
	// values push all assets toward moving together.
	CorrelationSpike float64

	// Seed controls random number generation for reproducibility.
	// A value of 0 uses a non-deterministic seed.
	Seed int64
}

// CrashGenerator extends MarketVAE to generate extreme tail scenarios
// for stress testing. It biases latent-space sampling toward the tails
// of the distribution and injects correlated shocks to simulate market
// crash dynamics.
type CrashGenerator struct {
	vae    *MarketVAE
	config CrashConfig
	rng    *rand.Rand
}

// NewCrashGenerator creates a CrashGenerator that uses the given VAE to
// produce extreme scenario samples. The severity in config is clamped to
// the [1.0, 10.0] range.
func NewCrashGenerator(vae *MarketVAE, config CrashConfig) *CrashGenerator {
	if config.Severity < 1.0 {
		config.Severity = 1.0
	}
	if config.Severity > 10.0 {
		config.Severity = 10.0
	}
	if config.Duration < 1 {
		config.Duration = 1
	}

	seed := config.Seed
	if seed == 0 {
		seed = rand.Int63()
	}

	return &CrashGenerator{
		vae:    vae,
		config: config,
		rng:    rand.New(rand.NewSource(seed)),
	}
}

// Generate produces n extreme scenario samples using the configured severity.
// Each sample is a flattened sequence of length Duration * InputDim.
func (cg *CrashGenerator) Generate(n int) [][]float64 {
	return cg.GenerateWithSeverity(n, cg.config.Severity)
}

// GenerateWithSeverity produces n extreme scenario samples at the given severity
// level (clamped to [1.0, 10.0]). Each sample is a sequence of Duration timesteps,
// returned as a flat slice of length Duration * InputDim.
//
// The generator works by:
//  1. Sampling from the tails of the latent distribution (scaled by severity).
//  2. Injecting a correlated shock component so assets move together.
//  3. Applying temporal decay so the crash impact fades over the duration.
func (cg *CrashGenerator) GenerateWithSeverity(n int, severity float64) [][]float64 {
	if severity < 1.0 {
		severity = 1.0
	}
	if severity > 10.0 {
		severity = 10.0
	}

	inputDim := cg.vae.config.InputDim
	latentDim := cg.vae.config.LatentDim
	duration := cg.config.Duration

	ctx := context.Background()
	results := make([][]float64, n)
	for i := 0; i < n; i++ {
		scenario := make([]float64, 0, duration*inputDim)

		// Generate a shared shock direction in latent space for the whole scenario.
		shockDir := cg.sampleTailLatent(latentDim, severity)

		for t := 0; t < duration; t++ {
			// Temporal decay: crash is strongest at t=0, fades exponentially.
			decay := math.Exp(-float64(t) / math.Max(float64(duration)/2.0, 1.0))

			// Sample a base latent vector biased toward the tail.
			z := cg.sampleTailLatent(latentDim, severity*decay)

			// Blend in the correlated shock direction.
			corrWeight := cg.config.CorrelationSpike * decay
			for j := 0; j < latentDim; j++ {
				z[j] = z[j]*(1-corrWeight) + shockDir[j]*corrWeight
			}

			// Decode through the VAE.
			zTensor, _ := tensor.New[float64]([]int{1, latentDim}, z)
			acts := cg.vae.decoderForward(ctx, zTensor)
			decoded := acts[len(acts)-1]

			step := make([]float64, inputDim)
			copy(step, decoded.Data())
			scenario = append(scenario, step...)
		}
		results[i] = scenario
	}
	return results
}

// sampleTailLatent samples a latent vector biased toward the tails of the
// distribution. It draws from a normal distribution and scales by the
// severity factor, then shifts away from zero by adding a signed offset.
func (cg *CrashGenerator) sampleTailLatent(dim int, severity float64) []float64 {
	z := make([]float64, dim)
	// Scale factor maps severity [1,10] to a multiplier [1, ~3].
	scale := 1.0 + (severity-1.0)*0.22

	for i := range z {
		// Draw from standard normal and push toward tails.
		sample := cg.rng.NormFloat64()
		// Add a signed tail offset proportional to severity.
		sign := 1.0
		if sample < 0 {
			sign = -1.0
		}
		z[i] = sample*scale + sign*scale
	}
	return z
}
