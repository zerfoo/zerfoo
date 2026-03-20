// Package residual provides residual connection layers for neural networks.
//
// It implements Attention Residuals (arXiv:2603.15031) which replace fixed
// additive residual connections with softmax attention over depth, allowing
// each layer to dynamically weight contributions from all previous layers.
//
// Stability: experimental
package residual
