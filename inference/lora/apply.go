package lora

// Apply computes the LoRA delta for a single layer and adds it to the base output.
//
// The LoRA formula: output = base_output + scaleFactor * (x @ A^T @ B^T)
//
// This is computed as two sequential small matmuls:
//
//	hidden = x @ A^T          (shape: [batch, rank])
//	delta  = hidden @ B^T     (shape: [batch, outDim])
//	output = base_output + scaleFactor * delta
//
// x:           input activation  [batch, inDim] (flattened row-major)
// baseOutput:  output from base model's linear layer [batch, outDim] (flattened, modified in-place)
// layer:       LoRA layer with A [rank, inDim] and B [outDim, rank]
// scaleFactor: alpha / rank
// batch:       number of batch elements
// inDim:       input dimension
// outDim:      output dimension
//
// Returns the modified baseOutput slice.
func Apply(baseOutput []float32, x []float32, layer *Layer, scaleFactor float64, batch, inDim, outDim int) []float32 {
	ApplyBatch(baseOutput, x, layer, scaleFactor, batch, inDim, outDim)
	return baseOutput
}

// ApplyBatch applies LoRA adaptation to a batch of inputs.
// x: [batch * inDim] flattened input
// baseOutput: [batch * outDim] flattened base output (modified in-place)
func ApplyBatch(baseOutput []float32, x []float32, layer *Layer, scaleFactor float64, batch, inDim, outDim int) {
	rank := len(layer.A)
	scale := float32(scaleFactor)

	hidden := make([]float32, rank)

	for b := 0; b < batch; b++ {
		xRow := x[b*inDim : (b+1)*inDim]
		outRow := baseOutput[b*outDim : (b+1)*outDim]

		// hidden = x @ A^T: [inDim] x [rank, inDim]^T → [rank]
		// Each hidden[r] = dot(xRow, A[r])
		for r := 0; r < rank; r++ {
			var sum float32
			aRow := layer.A[r]
			for k := 0; k < inDim; k++ {
				sum += xRow[k] * aRow[k]
			}
			hidden[r] = sum
		}

		// delta = hidden @ B^T, added scaled to output
		// Each outRow[o] += scale * dot(hidden, B[o])
		for o := 0; o < outDim; o++ {
			var sum float32
			bRow := layer.B[o]
			for r := 0; r < rank; r++ {
				sum += hidden[r] * bRow[r]
			}
			outRow[o] += scale * sum
		}
	}
}
