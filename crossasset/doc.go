// Package crossasset provides a cross-attention model for multi-source feature
// processing. Each source attends to features of all other sources via scaled
// dot-product multi-head attention, enabling the model to learn inter-source
// dependencies. This is useful for scenarios where multiple correlated data
// sources (e.g., different financial instruments or sensor streams) must be
// jointly analyzed.
//
// The model architecture applies cross-attention layers where each source
// computes queries from its own features and keys/values from all sources'
// features. Layer normalization and residual connections stabilize training.
package crossasset
