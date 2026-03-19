package synth

import (
	"fmt"
	"math"
	"math/rand"
)

// VAEConfig controls the architecture and training of a MarketVAE.
type VAEConfig struct {
	// InputDim is the dimensionality of the input data.
	InputDim int

	// LatentDim is the dimensionality of the latent space.
	LatentDim int

	// HiddenDims specifies the sizes of hidden layers in the encoder and decoder.
	// The decoder uses these dimensions in reverse order.
	HiddenDims []int

	// LearningRate is the step size for gradient descent. Default: 0.001.
	LearningRate float64

	// NEpochs is the number of training epochs. Default: 100.
	NEpochs int

	// Seed controls random number generation for reproducibility.
	// A value of 0 uses a non-deterministic seed.
	Seed int64
}

// MarketVAE is a Variational Autoencoder for synthetic data generation.
// It learns a smooth latent representation of the input data distribution
// and can generate new samples by decoding points from the latent space.
type MarketVAE struct {
	config VAEConfig
	rng    *rand.Rand

	// Encoder weights: input -> hidden layers -> (mu, logvar).
	encWeights [][]float64 // [layer][row-major]
	encBiases  [][]float64

	// Separate output heads for mean and log-variance.
	muWeights     []float64
	muBias        []float64
	logvarWeights []float64
	logvarBias    []float64

	// Decoder weights: latent -> hidden layers (reversed) -> output.
	decWeights [][]float64
	decBiases  [][]float64

	trained bool
}

// NewMarketVAE creates a new VAE with the given configuration.
// Weights are initialized using Xavier/Glorot uniform initialization.
func NewMarketVAE(config VAEConfig) *MarketVAE {
	if config.LearningRate <= 0 {
		config.LearningRate = 0.001
	}
	if config.NEpochs <= 0 {
		config.NEpochs = 100
	}

	seed := config.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	rng := rand.New(rand.NewSource(seed))

	v := &MarketVAE{
		config: config,
		rng:    rng,
	}
	v.initWeights()
	return v
}

// initWeights initializes encoder and decoder weights using Xavier initialization.
func (v *MarketVAE) initWeights() {
	dims := v.encoderDims()

	// Encoder hidden layers.
	v.encWeights = make([][]float64, len(dims)-1)
	v.encBiases = make([][]float64, len(dims)-1)
	for i := 0; i < len(dims)-1; i++ {
		v.encWeights[i] = v.xavierInit(dims[i], dims[i+1])
		v.encBiases[i] = make([]float64, dims[i+1])
	}

	// Mu and logvar heads from last hidden dim to latent dim.
	lastHidden := dims[len(dims)-1]
	v.muWeights = v.xavierInit(lastHidden, v.config.LatentDim)
	v.muBias = make([]float64, v.config.LatentDim)
	v.logvarWeights = v.xavierInit(lastHidden, v.config.LatentDim)
	v.logvarBias = make([]float64, v.config.LatentDim)

	// Decoder hidden layers (reverse of encoder).
	dDims := v.decoderDims()
	v.decWeights = make([][]float64, len(dDims)-1)
	v.decBiases = make([][]float64, len(dDims)-1)
	for i := 0; i < len(dDims)-1; i++ {
		v.decWeights[i] = v.xavierInit(dDims[i], dDims[i+1])
		v.decBiases[i] = make([]float64, dDims[i+1])
	}
}

// encoderDims returns [inputDim, hidden1, hidden2, ...].
func (v *MarketVAE) encoderDims() []int {
	dims := []int{v.config.InputDim}
	dims = append(dims, v.config.HiddenDims...)
	return dims
}

// decoderDims returns [latentDim, hiddenN, ..., hidden1, inputDim].
func (v *MarketVAE) decoderDims() []int {
	dims := []int{v.config.LatentDim}
	for i := len(v.config.HiddenDims) - 1; i >= 0; i-- {
		dims = append(dims, v.config.HiddenDims[i])
	}
	dims = append(dims, v.config.InputDim)
	return dims
}

// xavierInit returns a row-major weight matrix [rows*cols] with Xavier uniform init.
func (v *MarketVAE) xavierInit(rows, cols int) []float64 {
	limit := math.Sqrt(6.0 / float64(rows+cols))
	w := make([]float64, rows*cols)
	for i := range w {
		w[i] = v.rng.Float64()*2*limit - limit
	}
	return w
}

// Train trains the VAE on the provided data using mini-batch gradient descent.
// data is shaped [n_samples][input_dim].
func (v *MarketVAE) Train(data [][]float64) error {
	if len(data) == 0 {
		return fmt.Errorf("synth: data must have at least one sample")
	}
	if len(data[0]) != v.config.InputDim {
		return fmt.Errorf("synth: data dimension %d does not match InputDim %d", len(data[0]), v.config.InputDim)
	}
	for i, row := range data {
		if len(row) != v.config.InputDim {
			return fmt.Errorf("synth: row %d has %d columns, expected %d", i, len(row), v.config.InputDim)
		}
	}

	n := len(data)
	batchSize := 32
	if batchSize > n {
		batchSize = n
	}

	for epoch := 0; epoch < v.config.NEpochs; epoch++ {
		// Shuffle indices.
		perm := v.rng.Perm(n)

		for start := 0; start < n; start += batchSize {
			end := start + batchSize
			if end > n {
				end = n
			}
			batch := make([][]float64, end-start)
			for i, idx := range perm[start:end] {
				batch[i] = data[idx]
			}
			v.trainBatch(batch)
		}
	}

	v.trained = true
	return nil
}

// trainBatch performs a single gradient descent step on a mini-batch.
func (v *MarketVAE) trainBatch(batch [][]float64) {
	batchLen := len(batch)
	lr := v.config.LearningRate

	// Accumulate gradients.
	encWGrad := make([][]float64, len(v.encWeights))
	encBGrad := make([][]float64, len(v.encBiases))
	for i := range v.encWeights {
		encWGrad[i] = make([]float64, len(v.encWeights[i]))
		encBGrad[i] = make([]float64, len(v.encBiases[i]))
	}
	muWGrad := make([]float64, len(v.muWeights))
	muBGrad := make([]float64, len(v.muBias))
	logvarWGrad := make([]float64, len(v.logvarWeights))
	logvarBGrad := make([]float64, len(v.logvarBias))

	decWGrad := make([][]float64, len(v.decWeights))
	decBGrad := make([][]float64, len(v.decBiases))
	for i := range v.decWeights {
		decWGrad[i] = make([]float64, len(v.decWeights[i]))
		decBGrad[i] = make([]float64, len(v.decBiases[i]))
	}

	for _, x := range batch {
		// Forward pass through encoder.
		encActs := v.encoderForward(x)

		// Last encoder activation.
		h := encActs[len(encActs)-1]

		// Compute mu and logvar.
		eDims := v.encoderDims()
		lastH := eDims[len(eDims)-1]
		latent := v.config.LatentDim

		mu := linearForward(h, v.muWeights, v.muBias, lastH, latent)
		logvar := linearForward(h, v.logvarWeights, v.logvarBias, lastH, latent)

		// Reparameterization trick: z = mu + exp(0.5 * logvar) * eps.
		eps := make([]float64, latent)
		z := make([]float64, latent)
		for i := 0; i < latent; i++ {
			eps[i] = v.rng.NormFloat64()
			z[i] = mu[i] + math.Exp(0.5*logvar[i])*eps[i]
		}

		// Forward pass through decoder.
		decActs := v.decoderForward(z)
		recon := decActs[len(decActs)-1]

		// Compute gradients via backpropagation.
		// dL/d_recon = 2*(recon - x) / inputDim (MSE gradient).
		dRecon := make([]float64, v.config.InputDim)
		for i := range dRecon {
			dRecon[i] = 2 * (recon[i] - x[i]) / float64(v.config.InputDim)
		}

		// Backprop through decoder.
		dDims := v.decoderDims()
		dZ := v.backpropDecoder(decActs, dRecon, dDims, decWGrad, decBGrad)

		// Backprop through reparameterization.
		dMu := make([]float64, latent)
		dLogvar := make([]float64, latent)
		for i := 0; i < latent; i++ {
			// Reconstruction gradient.
			dMu[i] = dZ[i]
			dLogvar[i] = dZ[i] * 0.5 * math.Exp(0.5*logvar[i]) * eps[i]

			// KL divergence gradient: d/dmu[0.5*(mu^2 + exp(logvar) - logvar - 1)]
			dMu[i] += mu[i] / float64(v.config.InputDim)
			dLogvar[i] += 0.5 * (math.Exp(logvar[i]) - 1) / float64(v.config.InputDim)
		}

		// Backprop through mu/logvar linear layers.
		dH_mu := linearBackward(h, dMu, v.muWeights, lastH, latent, muWGrad, muBGrad)
		dH_logvar := linearBackward(h, dLogvar, v.logvarWeights, lastH, latent, logvarWGrad, logvarBGrad)

		dH := make([]float64, lastH)
		for i := range dH {
			dH[i] = dH_mu[i] + dH_logvar[i]
		}

		// Backprop through encoder.
		v.backpropEncoder(encActs, dH, eDims, encWGrad, encBGrad)
	}

	// Apply gradients (average over batch).
	scale := lr / float64(batchLen)
	for i := range v.encWeights {
		for j := range v.encWeights[i] {
			v.encWeights[i][j] -= scale * encWGrad[i][j]
		}
		for j := range v.encBiases[i] {
			v.encBiases[i][j] -= scale * encBGrad[i][j]
		}
	}
	for j := range v.muWeights {
		v.muWeights[j] -= scale * muWGrad[j]
	}
	for j := range v.muBias {
		v.muBias[j] -= scale * muBGrad[j]
	}
	for j := range v.logvarWeights {
		v.logvarWeights[j] -= scale * logvarWGrad[j]
	}
	for j := range v.logvarBias {
		v.logvarBias[j] -= scale * logvarBGrad[j]
	}
	for i := range v.decWeights {
		for j := range v.decWeights[i] {
			v.decWeights[i][j] -= scale * decWGrad[i][j]
		}
		for j := range v.decBiases[i] {
			v.decBiases[i][j] -= scale * decBGrad[i][j]
		}
	}
}

// encoderForward runs the encoder and returns activations at each layer.
// activations[0] = input, activations[i+1] = relu(W*activations[i] + b).
func (v *MarketVAE) encoderForward(x []float64) [][]float64 {
	dims := v.encoderDims()
	acts := make([][]float64, len(dims))
	acts[0] = x
	for i := 0; i < len(dims)-1; i++ {
		acts[i+1] = reluForward(linearForward(acts[i], v.encWeights[i], v.encBiases[i], dims[i], dims[i+1]))
	}
	return acts
}

// decoderForward runs the decoder and returns activations at each layer.
// activations[0] = z, activations[i+1] = relu(W*acts[i] + b) for hidden layers.
// The final layer has no activation (linear output).
func (v *MarketVAE) decoderForward(z []float64) [][]float64 {
	dims := v.decoderDims()
	acts := make([][]float64, len(dims))
	acts[0] = z
	for i := 0; i < len(dims)-1; i++ {
		linear := linearForward(acts[i], v.decWeights[i], v.decBiases[i], dims[i], dims[i+1])
		if i < len(dims)-2 {
			acts[i+1] = reluForward(linear)
		} else {
			acts[i+1] = linear // No activation on output.
		}
	}
	return acts
}

// backpropDecoder backpropagates through the decoder, accumulating gradients.
// Returns the gradient with respect to the decoder input (z).
func (v *MarketVAE) backpropDecoder(acts [][]float64, dOutput []float64, dims []int, wGrad, bGrad [][]float64) []float64 {
	d := dOutput
	for i := len(dims) - 2; i >= 0; i-- {
		if i < len(dims)-2 {
			// Apply ReLU derivative.
			d = reluBackward(acts[i+1], d)
		}
		d = linearBackward(acts[i], d, v.decWeights[i], dims[i], dims[i+1], wGrad[i], bGrad[i])
	}
	return d
}

// backpropEncoder backpropagates through the encoder, accumulating gradients.
func (v *MarketVAE) backpropEncoder(acts [][]float64, dOutput []float64, dims []int, wGrad, bGrad [][]float64) {
	d := dOutput
	for i := len(dims) - 2; i >= 0; i-- {
		d = reluBackward(acts[i+1], d)
		d = linearBackward(acts[i], d, v.encWeights[i], dims[i], dims[i+1], wGrad[i], bGrad[i])
	}
}

// Generate produces n synthetic samples by sampling from the latent space
// and decoding. The latent samples are drawn from the standard normal prior N(0,I).
func (v *MarketVAE) Generate(n int) [][]float64 {
	result := make([][]float64, n)
	for i := 0; i < n; i++ {
		z := make([]float64, v.config.LatentDim)
		for j := range z {
			z[j] = v.rng.NormFloat64()
		}
		acts := v.decoderForward(z)
		recon := acts[len(acts)-1]
		out := make([]float64, len(recon))
		copy(out, recon)
		result[i] = out
	}
	return result
}

// Encode maps input data to latent representations by returning the
// mean vector of the encoder's posterior distribution for each sample.
func (v *MarketVAE) Encode(data [][]float64) [][]float64 {
	eDims := v.encoderDims()
	lastH := eDims[len(eDims)-1]
	latent := v.config.LatentDim

	result := make([][]float64, len(data))
	for i, x := range data {
		encActs := v.encoderForward(x)
		h := encActs[len(encActs)-1]
		mu := linearForward(h, v.muWeights, v.muBias, lastH, latent)
		out := make([]float64, len(mu))
		copy(out, mu)
		result[i] = out
	}
	return result
}

// linearForward computes y = W^T * x + b where W is [inDim x outDim] row-major.
func linearForward(x, w, b []float64, inDim, outDim int) []float64 {
	y := make([]float64, outDim)
	for j := 0; j < outDim; j++ {
		y[j] = b[j]
		for i := 0; i < inDim; i++ {
			y[j] += x[i] * w[i*outDim+j]
		}
	}
	return y
}

// linearBackward computes gradients for a linear layer and returns dInput.
// Accumulates into wGrad and bGrad.
func linearBackward(input, dOutput, weights []float64, inDim, outDim int, wGrad, bGrad []float64) []float64 {
	dInput := make([]float64, inDim)
	for j := 0; j < outDim; j++ {
		bGrad[j] += dOutput[j]
		for i := 0; i < inDim; i++ {
			wGrad[i*outDim+j] += input[i] * dOutput[j]
			dInput[i] += weights[i*outDim+j] * dOutput[j]
		}
	}
	return dInput
}

// reluForward applies ReLU element-wise.
func reluForward(x []float64) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			y[i] = v
		}
	}
	return y
}

// reluBackward applies the ReLU derivative: gradient passes through where activation > 0.
func reluBackward(activation, dOutput []float64) []float64 {
	d := make([]float64, len(dOutput))
	for i := range d {
		if activation[i] > 0 {
			d[i] = dOutput[i]
		}
	}
	return d
}
