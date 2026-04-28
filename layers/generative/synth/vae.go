package synth

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/zerfoo/training/optimizer"
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
	engine compute.Engine[float64]
	ops    numeric.Arithmetic[float64]

	// Encoder parameters: input -> hidden layers.
	encParams  []*graph.Parameter[float64] // weights
	encBParams []*graph.Parameter[float64] // biases

	// Separate output heads for mean and log-variance.
	muParam      *graph.Parameter[float64]
	muBParam     *graph.Parameter[float64]
	logvarParam  *graph.Parameter[float64]
	logvarBParam *graph.Parameter[float64]

	// Decoder parameters: latent -> hidden layers (reversed) -> output.
	decParams  []*graph.Parameter[float64]
	decBParams []*graph.Parameter[float64]

	sgd     *optimizer.SGD[float64]
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

	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)

	v := &MarketVAE{
		config: config,
		rng:    rng,
		engine: engine,
		ops:    ops,
		sgd:    optimizer.NewSGD[float64](engine, ops, float32(config.LearningRate)),
	}
	v.initWeights()
	return v
}

// initWeights initializes encoder and decoder weights using Xavier initialization.
func (v *MarketVAE) initWeights() {
	dims := v.encoderDims()

	// Encoder hidden layers.
	v.encParams = make([]*graph.Parameter[float64], len(dims)-1)
	v.encBParams = make([]*graph.Parameter[float64], len(dims)-1)
	for i := 0; i < len(dims)-1; i++ {
		v.encParams[i] = v.makeParam(fmt.Sprintf("enc.w.%d", i), dims[i+1], dims[i])
		v.encBParams[i] = v.makeBiasParam(fmt.Sprintf("enc.b.%d", i), dims[i+1])
	}

	// Mu and logvar heads from last hidden dim to latent dim.
	lastHidden := dims[len(dims)-1]
	v.muParam = v.makeParam("mu.w", v.config.LatentDim, lastHidden)
	v.muBParam = v.makeBiasParam("mu.b", v.config.LatentDim)
	v.logvarParam = v.makeParam("logvar.w", v.config.LatentDim, lastHidden)
	v.logvarBParam = v.makeBiasParam("logvar.b", v.config.LatentDim)

	// Decoder hidden layers (reverse of encoder).
	dDims := v.decoderDims()
	v.decParams = make([]*graph.Parameter[float64], len(dDims)-1)
	v.decBParams = make([]*graph.Parameter[float64], len(dDims)-1)
	for i := 0; i < len(dDims)-1; i++ {
		v.decParams[i] = v.makeParam(fmt.Sprintf("dec.w.%d", i), dDims[i+1], dDims[i])
		v.decBParams[i] = v.makeBiasParam(fmt.Sprintf("dec.b.%d", i), dDims[i+1])
	}
}

// makeParam creates a Xavier-initialized weight parameter [outDim, inDim] (row-major).
func (v *MarketVAE) makeParam(name string, outDim, inDim int) *graph.Parameter[float64] {
	limit := math.Sqrt(6.0 / float64(inDim+outDim))
	data := make([]float64, outDim*inDim)
	for i := range data {
		data[i] = v.rng.Float64()*2*limit - limit
	}
	t, _ := tensor.New[float64]([]int{outDim, inDim}, data)
	p, _ := graph.NewParameter[float64](name, t, tensor.New[float64])
	return p
}

// makeBiasParam creates a zero-initialized bias parameter [dim].
func (v *MarketVAE) makeBiasParam(name string, dim int) *graph.Parameter[float64] {
	t, _ := tensor.New[float64]([]int{dim}, make([]float64, dim))
	p, _ := graph.NewParameter[float64](name, t, tensor.New[float64])
	return p
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

// allParams returns all trainable parameters for SGD.
func (v *MarketVAE) allParams() []*graph.Parameter[float64] {
	var params []*graph.Parameter[float64]
	params = append(params, v.encParams...)
	params = append(params, v.encBParams...)
	params = append(params, v.muParam, v.muBParam, v.logvarParam, v.logvarBParam)
	params = append(params, v.decParams...)
	params = append(params, v.decBParams...)
	return params
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

	ctx := context.Background()

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
			v.trainBatch(ctx, batch)
		}
	}

	v.trained = true
	return nil
}

// trainBatch performs a single gradient descent step on a mini-batch.
func (v *MarketVAE) trainBatch(ctx context.Context, batch [][]float64) {
	params := v.allParams()
	for _, p := range params {
		p.ClearGradient()
	}

	eDims := v.encoderDims()
	lastH := eDims[len(eDims)-1]
	latent := v.config.LatentDim

	for _, x := range batch {
		// Forward pass through encoder.
		encActs := v.encoderForward(ctx, x)

		// Last encoder activation.
		h := encActs[len(encActs)-1]

		// Compute mu and logvar via functional.Linear.
		mu := v.linearFwd(ctx, h, v.muParam.Value, v.muBParam.Value)
		logvar := v.linearFwd(ctx, h, v.logvarParam.Value, v.logvarBParam.Value)

		// Reparameterization trick: z = mu + exp(0.5 * logvar) * eps.
		muData := mu.Data()
		logvarData := logvar.Data()
		eps := make([]float64, latent)
		zData := make([]float64, latent)
		for i := 0; i < latent; i++ {
			eps[i] = v.rng.NormFloat64()
			zData[i] = muData[i] + math.Exp(0.5*logvarData[i])*eps[i]
		}
		z, _ := tensor.New[float64]([]int{1, latent}, zData)

		// Forward pass through decoder.
		decActs := v.decoderForward(ctx, z)
		recon := decActs[len(decActs)-1]

		// Compute gradients via backpropagation.
		// dL/d_recon = 2*(recon - x) / inputDim (MSE gradient).
		reconData := recon.Data()
		dReconData := make([]float64, v.config.InputDim)
		for i := range dReconData {
			dReconData[i] = 2 * (reconData[i] - x[i]) / float64(v.config.InputDim)
		}
		dRecon, _ := tensor.New[float64]([]int{1, v.config.InputDim}, dReconData)

		// Backprop through decoder.
		dDims := v.decoderDims()
		dZ := v.backpropDecoder(ctx, decActs, dRecon, dDims)

		// Backprop through reparameterization.
		dZData := dZ.Data()
		dMuData := make([]float64, latent)
		dLogvarData := make([]float64, latent)
		for i := 0; i < latent; i++ {
			// Reconstruction gradient.
			dMuData[i] = dZData[i]
			dLogvarData[i] = dZData[i] * 0.5 * math.Exp(0.5*logvarData[i]) * eps[i]

			// KL divergence gradient.
			dMuData[i] += muData[i] / float64(v.config.InputDim)
			dLogvarData[i] += 0.5 * (math.Exp(logvarData[i]) - 1) / float64(v.config.InputDim)
		}
		dMu, _ := tensor.New[float64]([]int{1, latent}, dMuData)
		dLogvar, _ := tensor.New[float64]([]int{1, latent}, dLogvarData)

		// Backprop through mu/logvar linear layers.
		dH_mu := v.linearBwd(ctx, h, dMu, v.muParam, v.muBParam, lastH, latent)
		dH_logvar := v.linearBwd(ctx, h, dLogvar, v.logvarParam, v.logvarBParam, lastH, latent)

		dH, _ := v.engine.Add(ctx, dH_mu, dH_logvar)

		// Backprop through encoder.
		v.backpropEncoder(ctx, encActs, dH, eDims)
	}

	// Scale gradients by 1/batchLen.
	scale := 1.0 / float64(len(batch))
	for _, p := range params {
		p.Gradient, _ = v.engine.MulScalar(ctx, p.Gradient, scale)
	}

	// Apply gradients via SGD.
	_ = v.sgd.Step(ctx, params)
}

// linearFwd computes functional.Linear: y = x @ weight^T + bias.
// Input x is []float64 (1D), weight is [outDim, inDim], bias is [outDim].
// Returns tensor [1, outDim].
func (v *MarketVAE) linearFwd(ctx context.Context, input, weight, bias *tensor.TensorNumeric[float64]) *tensor.TensorNumeric[float64] {
	result, _ := functional.Linear(ctx, v.engine, input, weight, bias)
	return result
}

// linearBwd backpropagates through a linear layer, accumulates gradients into param.
// Returns dInput [1, inDim].
func (v *MarketVAE) linearBwd(ctx context.Context, input, dOutput *tensor.TensorNumeric[float64],
	wParam, bParam *graph.Parameter[float64], inDim, outDim int) *tensor.TensorNumeric[float64] {

	// dInput = dOutput @ weight (weight is [outDim, inDim])
	dInput, _ := v.engine.MatMul(ctx, dOutput, wParam.Value)

	// dW = dOutput^T @ input → [outDim, inDim]
	dOutputT, _ := v.engine.Transpose(ctx, dOutput, []int{1, 0})
	dW, _ := v.engine.MatMul(ctx, dOutputT, input)
	_ = wParam.AddGradient(dW)

	// dB = dOutput reshaped to [outDim]
	dBData := make([]float64, outDim)
	copy(dBData, dOutput.Data())
	dB, _ := tensor.New[float64]([]int{outDim}, dBData)
	_ = bParam.AddGradient(dB)

	return dInput
}

// encoderForward runs the encoder and returns activations at each layer.
// activations[0] = input [1, inputDim], activations[i+1] = relu(Linear(activations[i])).
func (v *MarketVAE) encoderForward(ctx context.Context, x []float64) []*tensor.TensorNumeric[float64] {
	dims := v.encoderDims()
	acts := make([]*tensor.TensorNumeric[float64], len(dims))
	t, _ := tensor.New[float64]([]int{1, dims[0]}, x)
	acts[0] = t
	for i := 0; i < len(dims)-1; i++ {
		linear := v.linearFwd(ctx, acts[i], v.encParams[i].Value, v.encBParams[i].Value)
		relu, _ := functional.ReLU(ctx, v.engine, v.ops, linear)
		acts[i+1] = relu
	}
	return acts
}

// decoderForward runs the decoder and returns activations at each layer.
// activations[0] = z [1, latentDim], activations[i+1] = relu(Linear(acts[i])) for hidden layers.
// The final layer has no activation (linear output).
func (v *MarketVAE) decoderForward(ctx context.Context, z *tensor.TensorNumeric[float64]) []*tensor.TensorNumeric[float64] {
	dims := v.decoderDims()
	acts := make([]*tensor.TensorNumeric[float64], len(dims))
	acts[0] = z
	for i := 0; i < len(dims)-1; i++ {
		linear := v.linearFwd(ctx, acts[i], v.decParams[i].Value, v.decBParams[i].Value)
		if i < len(dims)-2 {
			relu, _ := functional.ReLU(ctx, v.engine, v.ops, linear)
			acts[i+1] = relu
		} else {
			acts[i+1] = linear // No activation on output.
		}
	}
	return acts
}

// backpropDecoder backpropagates through the decoder, accumulating gradients.
// Returns the gradient with respect to the decoder input (z).
func (v *MarketVAE) backpropDecoder(ctx context.Context, acts []*tensor.TensorNumeric[float64], dOutput *tensor.TensorNumeric[float64], dims []int) *tensor.TensorNumeric[float64] {
	d := dOutput
	for i := len(dims) - 2; i >= 0; i-- {
		if i < len(dims)-2 {
			// Apply ReLU derivative: gradient passes through where activation > 0.
			d = v.reluBwd(ctx, acts[i+1], d)
		}
		d = v.linearBwd(ctx, acts[i], d, v.decParams[i], v.decBParams[i], dims[i], dims[i+1])
	}
	return d
}

// backpropEncoder backpropagates through the encoder, accumulating gradients.
func (v *MarketVAE) backpropEncoder(ctx context.Context, acts []*tensor.TensorNumeric[float64], dOutput *tensor.TensorNumeric[float64], dims []int) {
	d := dOutput
	for i := len(dims) - 2; i >= 0; i-- {
		d = v.reluBwd(ctx, acts[i+1], d)
		d = v.linearBwd(ctx, acts[i], d, v.encParams[i], v.encBParams[i], dims[i], dims[i+1])
	}
}

// reluBwd applies the ReLU derivative via Engine: gradient passes through where activation > 0.
func (v *MarketVAE) reluBwd(ctx context.Context, activation, dOutput *tensor.TensorNumeric[float64]) *tensor.TensorNumeric[float64] {
	actData := activation.Data()
	dData := dOutput.Data()
	result := make([]float64, len(dData))
	for i := range result {
		if actData[i] > 0 {
			result[i] = dData[i]
		}
	}
	t, _ := tensor.New[float64](activation.Shape(), result)
	return t
}

// Generate produces n synthetic samples by sampling from the latent space
// and decoding. The latent samples are drawn from the standard normal prior N(0,I).
func (v *MarketVAE) Generate(n int) [][]float64 {
	ctx := context.Background()
	result := make([][]float64, n)
	for i := 0; i < n; i++ {
		zData := make([]float64, v.config.LatentDim)
		for j := range zData {
			zData[j] = v.rng.NormFloat64()
		}
		z, _ := tensor.New[float64]([]int{1, v.config.LatentDim}, zData)
		acts := v.decoderForward(ctx, z)
		recon := acts[len(acts)-1]
		out := make([]float64, v.config.InputDim)
		copy(out, recon.Data())
		result[i] = out
	}
	return result
}

// Encode maps input data to latent representations by returning the
// mean vector of the encoder's posterior distribution for each sample.
func (v *MarketVAE) Encode(data [][]float64) [][]float64 {
	ctx := context.Background()
	result := make([][]float64, len(data))
	for i, x := range data {
		encActs := v.encoderForward(ctx, x)
		h := encActs[len(encActs)-1]
		mu := v.linearFwd(ctx, h, v.muParam.Value, v.muBParam.Value)
		out := make([]float64, v.config.LatentDim)
		copy(out, mu.Data())
		result[i] = out
	}
	return result
}
