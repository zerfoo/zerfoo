// Experimental — this package is not yet wired into the main framework.
//
// Package synth provides synthetic data generation using generative models.
//
// The primary entry point is MarketVAE, a Variational Autoencoder (VAE)
// that learns the distribution of input data and generates realistic
// synthetic samples. The model consists of:
//
//   - An encoder that maps input data to a latent distribution
//     parameterized by mean and log-variance vectors.
//   - A decoder that reconstructs data from latent samples using the
//     reparameterization trick (z = mu + sigma * epsilon).
//   - A loss function combining reconstruction error (MSE) with
//     KL divergence to regularize the latent space.
//
// Typical usage:
//
//	vae := synth.NewMarketVAE(synth.VAEConfig{
//	    InputDim:     10,
//	    LatentDim:    3,
//	    HiddenDims:   []int{32, 16},
//	    LearningRate: 0.001,
//	    NEpochs:      100,
//	})
//	err := vae.Train(data)
//	synthetic := vae.Generate(1000)
//	latent := vae.Encode(data)
//
// (Stability: alpha)
package synth
