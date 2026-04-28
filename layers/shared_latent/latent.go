package shared_latent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// ProjectionConfig controls the training of projection matrices.
type ProjectionConfig struct {
	// LatentDim overrides the latent space dimension for training. If zero,
	// the dimension specified at construction time is used.
	LatentDim int

	// LearningRate is the step size for gradient descent. Default: 0.01.
	LearningRate float64

	// NEpochs is the number of training epochs. Default: 100.
	NEpochs int

	// AlignmentWeight controls how strongly the alignment loss (bringing
	// corresponding samples from different models close in latent space)
	// is weighted relative to reconstruction loss. Default: 1.0.
	AlignmentWeight float64
}

func (c ProjectionConfig) learningRate() float64 {
	if c.LearningRate == 0 {
		return 0.01
	}
	return c.LearningRate
}

func (c ProjectionConfig) nEpochs() int {
	if c.NEpochs == 0 {
		return 100
	}
	return c.NEpochs
}

func (c ProjectionConfig) alignmentWeight() float64 {
	if c.AlignmentWeight == 0 {
		return 1.0
	}
	return c.AlignmentWeight
}

// modelEntry holds the projection and reconstruction matrices for a single
// registered model.
type modelEntry struct {
	inputDim int
	// project: inputDim x latentDim  (row-major)
	project []float64
	// reconstruct: latentDim x inputDim  (row-major)
	reconstruct []float64
}

// LatentSpace is a shared embedding space that multiple models can project
// into and read from via learned linear projections.
type LatentSpace struct {
	mu     sync.RWMutex
	dim    int
	models map[string]*modelEntry
	rng    *rand.Rand
	engine compute.Engine[float64]
}

// NewLatentSpace creates a shared latent space with the given dimension.
func NewLatentSpace(dim int, engine compute.Engine[float64]) *LatentSpace {
	return &LatentSpace{
		dim:    dim,
		models: make(map[string]*modelEntry),
		rng:    rand.New(rand.NewSource(42)),
		engine: engine,
	}
}

// Register adds a model to the latent space with the specified input
// dimension. Projection matrices are initialized with Xavier uniform
// initialization.
func (ls *LatentSpace) Register(name string, inputDim int) {
	ls.mu.Lock()
	defer ls.mu.Unlock()

	entry := &modelEntry{inputDim: inputDim}

	// Xavier uniform initialization
	entry.project = ls.xavierInit(inputDim, ls.dim)
	entry.reconstruct = ls.xavierInit(ls.dim, inputDim)

	ls.models[name] = entry
}

// xavierInit returns a row-major matrix of shape rows x cols initialized
// with Xavier uniform: U(-limit, limit) where limit = sqrt(6/(rows+cols)).
func (ls *LatentSpace) xavierInit(rows, cols int) []float64 {
	limit := math.Sqrt(6.0 / float64(rows+cols))
	m := make([]float64, rows*cols)
	for i := range m {
		m[i] = ls.rng.Float64()*2*limit - limit
	}
	return m
}

// Project maps model features into the shared latent space.
// features must have length equal to the model's registered input dimension.
func (ls *LatentSpace) Project(ctx context.Context, name string, features []float64) ([]float64, error) {
	ls.mu.RLock()
	defer ls.mu.RUnlock()

	entry := ls.models[name]
	return ls.matMul(ctx, entry.project, features, entry.inputDim, ls.dim)
}

// Retrieve maps a latent-space vector back to a model's representation space.
func (ls *LatentSpace) Retrieve(ctx context.Context, name string, latent []float64) ([]float64, error) {
	ls.mu.RLock()
	defer ls.mu.RUnlock()

	entry := ls.models[name]
	return ls.matMul(ctx, entry.reconstruct, latent, ls.dim, entry.inputDim)
}

// TrainProjections learns projection and reconstruction matrices from
// aligned data. The data map keys are model names; values are slices of
// feature vectors. All models must have the same number of samples (aligned
// by index). Training minimises reconstruction loss plus an alignment loss
// that pulls corresponding samples from different models toward the same
// point in latent space.
func (ls *LatentSpace) TrainProjections(ctx context.Context, data map[string][][]float64, config ProjectionConfig) error {
	if len(data) < 2 {
		return errors.New("shared: at least two models required for training")
	}

	ls.mu.Lock()
	defer ls.mu.Unlock()

	// Validate aligned sample counts.
	n := -1
	for name, samples := range data {
		if _, ok := ls.models[name]; !ok {
			return fmt.Errorf("shared: model %q not registered", name)
		}
		if n == -1 {
			n = len(samples)
		} else if len(samples) != n {
			return fmt.Errorf("shared: mismatched sample counts: expected %d, got %d for model %q", n, len(samples), name)
		}
	}
	if n == 0 {
		return errors.New("shared: no training samples provided")
	}

	lr := config.learningRate()
	epochs := config.nEpochs()
	alignW := config.alignmentWeight()

	// Collect model names in deterministic order.
	names := make([]string, 0, len(data))
	for name := range data {
		names = append(names, name)
	}

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < n; i++ {
			// Compute latent representations for each model.
			latents := make(map[string][]float64, len(names))
			for _, name := range names {
				entry := ls.models[name]
				z, err := ls.matMul(ctx, entry.project, data[name][i], entry.inputDim, ls.dim)
				if err != nil {
					return err
				}
				latents[name] = z
			}

			// Compute mean latent for alignment target.
			meanLatent := make([]float64, ls.dim)
			for _, name := range names {
				for d := 0; d < ls.dim; d++ {
					meanLatent[d] += latents[name][d]
				}
			}
			invN := 1.0 / float64(len(names))
			for d := 0; d < ls.dim; d++ {
				meanLatent[d] *= invN
			}

			for _, name := range names {
				entry := ls.models[name]
				x := data[name][i]
				z := latents[name]

				// Reconstruction: xHat = reconstruct * z
				xHat, err := ls.matMul(ctx, entry.reconstruct, z, ls.dim, entry.inputDim)
				if err != nil {
					return err
				}

				// Reconstruction error: dRecon = xHat - x
				dRecon := make([]float64, entry.inputDim)
				for d := 0; d < entry.inputDim; d++ {
					dRecon[d] = xHat[d] - x[d]
				}

				// Alignment error: dAlign = z - meanLatent
				dAlign := make([]float64, ls.dim)
				for d := 0; d < ls.dim; d++ {
					dAlign[d] = z[d] - meanLatent[d]
				}

				// --- Update reconstruction matrix ---
				// grad_reconstruct[l][d] = dRecon[d] * z[l]
				for l := 0; l < ls.dim; l++ {
					for d := 0; d < entry.inputDim; d++ {
						entry.reconstruct[l*entry.inputDim+d] -= lr * dRecon[d] * z[l]
					}
				}

				// --- Update projection matrix ---
				// Reconstruction gradient through projection:
				// dL/dP[r][c] = sum_d(dRecon[d] * R[·][d]) backprop through z
				// Plus alignment gradient: dAlign[l] * x[c]
				for r := 0; r < entry.inputDim; r++ {
					for c := 0; c < ls.dim; c++ {
						// Backprop reconstruction loss through reconstruct matrix.
						reconGrad := 0.0
						for d := 0; d < entry.inputDim; d++ {
							reconGrad += dRecon[d] * entry.reconstruct[c*entry.inputDim+d]
						}
						reconGrad *= x[r]

						alignGrad := alignW * dAlign[c] * x[r]
						entry.project[r*ls.dim+c] -= lr * (reconGrad + alignGrad)
					}
				}
			}
		}

		// Decay learning rate.
		lr *= 0.999
	}

	return nil
}

// matMul computes y = M^T * x via engine.MatMul where M is rows x cols
// (row-major), x has length rows, and the result has length cols.
func (ls *LatentSpace) matMul(ctx context.Context, m []float64, x []float64, rows, cols int) ([]float64, error) {
	// Reshape x as [1, rows] and M as [rows, cols] so that
	// result = x_row * M has shape [1, cols].
	xT, err := tensor.New[float64]([]int{1, rows}, x)
	if err != nil {
		return nil, err
	}
	mT, err := tensor.New[float64]([]int{rows, cols}, m)
	if err != nil {
		return nil, err
	}
	result, err := ls.engine.MatMul(ctx, xT, mT)
	if err != nil {
		return nil, err
	}
	return result.Data(), nil
}
