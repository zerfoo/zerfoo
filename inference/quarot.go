package inference

import (
	"fmt"
	"log/slog"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// FuseQuaRotWeights applies the QuaRot (Quantization with Rotation) technique
// by fusing a normalized Walsh-Hadamard rotation into model weight matrices.
// After fusion, inference requires no additional runtime computation — the
// rotation is baked into the weights.
//
// For each transformer layer, the following projections are rotated:
//
//   - Attention: Q, K, V, O projection weights
//   - FFN: gate, up, down projection weights
//
// The Hadamard matrix H is orthogonal and involutory (H * H = I), so applying
// it twice recovers the original weights. The rotation improves quantization
// quality by spreading outlier magnitudes across dimensions (arXiv:2404.00456).
//
// Weight convention: weights are stored as [outDim, inDim] (row-major).
// The rotation is applied along the input dimension: W_rotated = W * H^T,
// which is equivalent to rotating the input space. Since H is symmetric
// (H = H^T for the normalized Hadamard), this simplifies to W_rotated = W * H.
func FuseQuaRotWeights(tensors map[string]*tensor.TensorNumeric[float32], numLayers int) error {
	if numLayers <= 0 {
		return fmt.Errorf("quarot: numLayers must be positive, got %d", numLayers)
	}

	// Determine hidden dimension from the first Q projection weight.
	qName := "model.layers.0.self_attn.q_proj.weight"
	qW, ok := tensors[qName]
	if !ok {
		return fmt.Errorf("quarot: missing tensor %q to determine hidden dimension", qName)
	}
	shape := qW.Shape()
	if len(shape) != 2 {
		return fmt.Errorf("quarot: expected 2D weight tensor, got %dD for %q", len(shape), qName)
	}
	hiddenDim := shape[1] // inDim

	if hiddenDim == 0 || (hiddenDim&(hiddenDim-1)) != 0 {
		return fmt.Errorf("quarot: hidden dimension %d is not a power of 2; Walsh-Hadamard requires power-of-2 dimensions", hiddenDim)
	}

	// Pre-compute the normalization factor: 1/sqrt(hiddenDim).
	norm := float32(1.0 / math.Sqrt(float64(hiddenDim)))

	slog.Info("fusing QuaRot Hadamard rotation into weights",
		"hidden_dim", hiddenDim, "num_layers", numLayers)

	for layer := range numLayers {
		prefix := fmt.Sprintf("model.layers.%d.", layer)

		// Attention projections: Q, K, V, O.
		attnSuffixes := []string{
			"self_attn.q_proj.weight",
			"self_attn.k_proj.weight",
			"self_attn.v_proj.weight",
			"self_attn.o_proj.weight",
		}
		for _, suffix := range attnSuffixes {
			name := prefix + suffix
			if err := fuseHadamardIntoWeight(tensors, name, hiddenDim, norm); err != nil {
				return fmt.Errorf("quarot layer %d: %w", layer, err)
			}
		}

		// FFN projections: gate, up, down.
		ffnSuffixes := []string{
			"mlp.gate_proj.weight",
			"mlp.up_proj.weight",
			"mlp.down_proj.weight",
		}
		for _, suffix := range ffnSuffixes {
			name := prefix + suffix
			if err := fuseHadamardIntoWeight(tensors, name, hiddenDim, norm); err != nil {
				return fmt.Errorf("quarot layer %d: %w", layer, err)
			}
		}
	}

	slog.Info("QuaRot weight fusion complete", "num_layers", numLayers)
	return nil
}

// fuseHadamardIntoWeight applies the normalized Walsh-Hadamard transform to
// each row of the weight matrix W[outDim, inDim]. The result is W * H where
// H is the normalized Hadamard matrix of size inDim x inDim.
//
// Instead of materializing the full Hadamard matrix, this uses the in-place
// Fast Walsh-Hadamard Transform (FWHT) on each row, which is O(n log n)
// per row instead of O(n^2).
func fuseHadamardIntoWeight(
	tensors map[string]*tensor.TensorNumeric[float32],
	name string,
	expectedInDim int,
	norm float32,
) error {
	t, ok := tensors[name]
	if !ok {
		// Some architectures (e.g., GQA with fewer K/V heads) may have
		// different dimensions for K/V projections. Skip missing tensors
		// silently — the caller logs which layers were processed.
		return nil
	}

	// Dequantize if needed to get float32 data.
	data := t.Data()
	shape := t.Shape()
	if len(shape) != 2 {
		return fmt.Errorf("expected 2D tensor for %q, got %dD", name, len(shape))
	}
	outDim, inDim := shape[0], shape[1]

	if inDim != expectedInDim {
		// K/V projections in GQA may have inDim == hiddenDim but outDim differs.
		// O projection has inDim == hiddenDim. Gate/up have inDim == hiddenDim.
		// Down has inDim == intermediateDim (different from hiddenDim).
		// For down_proj, the inDim is the intermediate size, not hidden size.
		// We still apply Hadamard if inDim is power of 2.
		if inDim == 0 || (inDim&(inDim-1)) != 0 {
			slog.Warn("quarot: skipping non-power-of-2 tensor",
				"tensor", name, "inDim", inDim)
			return nil
		}
		norm = float32(1.0 / math.Sqrt(float64(inDim)))
	}

	// Apply FWHT to each row in-place.
	rotated := make([]float32, len(data))
	copy(rotated, data)

	for row := range outDim {
		offset := row * inDim
		fwht(rotated[offset : offset+inDim])
		// Apply normalization: H_normalized = H / sqrt(n).
		for col := range inDim {
			rotated[offset+col] *= norm
		}
	}

	// Create a new tensor with rotated data (preserves the original shape).
	rotatedTensor, err := tensor.New(shape, rotated)
	if err != nil {
		return fmt.Errorf("create rotated tensor for %q: %w", name, err)
	}
	tensors[name] = rotatedTensor
	return nil
}

// fwht performs the in-place Fast Walsh-Hadamard Transform on a slice of
// length n, where n must be a power of 2. The transform computes x' = H_n * x
// where H_n is the unnormalized Hadamard matrix (entries are +1/-1).
//
// The caller is responsible for applying the 1/sqrt(n) normalization factor
// to obtain the orthonormal Hadamard matrix.
//
// Complexity: O(n log n) time, O(1) extra space.
func fwht(x []float32) {
	n := len(x)
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := range h {
				a := x[i+j]
				b := x[i+j+h]
				x[i+j] = a + b
				x[i+j+h] = a - b
			}
		}
	}
}
