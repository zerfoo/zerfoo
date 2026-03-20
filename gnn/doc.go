// Experimental — this package is not yet wired into the main framework.
//
// Package gnn implements Graph Neural Network layers for node-level
// representation learning on graph-structured data.
//
// The package provides two architectures:
//
//   - GCN (Graph Convolutional Network): spectral-based convolution using
//     symmetric normalized adjacency, as described by Kipf & Welling (2017).
//     Each layer computes H' = sigma(D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H W)
//     where A_tilde = A + I adds self-loops.
//
//   - GAT (Graph Attention Network): attention-based message passing as
//     described by Velickovic et al. (2018). Attention coefficients
//     alpha_{ij} = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j])) are computed
//     over each node's neighborhood, with multi-head attention support.
//
// Both models support multi-layer stacking, dropout regularization, and
// training via gradient descent with cross-entropy loss.
package gnn
