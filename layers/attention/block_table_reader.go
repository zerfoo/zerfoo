package attention

import "github.com/zerfoo/ztensor/tensor"

// BlockTableReader reads key/value tensors directly from paged block tables,
// avoiding the gather-to-contiguous copy. Implementations should iterate over
// blocks and return the full KV sequence for a given layer.
type BlockTableReader[T tensor.Numeric] interface {
	// ReadKV returns the cached key and value tensors for the given layer
	// as contiguous [batch*numKVHeads, seqLen, headDim] tensors read directly
	// from blocks. Returns false if the layer has no cached data.
	ReadKV(layer int) (k, v *tensor.TensorNumeric[T], ok bool)
}
