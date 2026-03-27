package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// EAGLEConfig holds configuration for EAGLE-style self-speculative decoding.
type EAGLEConfig struct {
	NumDraftTokens int // number of draft tokens to generate per step
	HiddenDim      int // hidden dimension of the model
}

// BuildEAGLEHead constructs an EAGLEHead layer from an engine and config.
// The returned head can be used with GenerateDraftTokens to produce draft
// tokens from the penultimate transformer layer's output.
func BuildEAGLEHead[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	config EAGLEConfig,
) (*core.EAGLEHead[T], error) {
	if config.HiddenDim <= 0 {
		return nil, fmt.Errorf("EAGLEConfig: HiddenDim must be positive, got %d", config.HiddenDim)
	}
	if config.NumDraftTokens <= 0 {
		return nil, fmt.Errorf("EAGLEConfig: NumDraftTokens must be positive, got %d", config.NumDraftTokens)
	}
	return core.NewEAGLEHead[T](engine, ops, config.HiddenDim)
}

// lmHeadForward applies the LM head weight to hidden states to produce logits.
// hiddenStates shape: [batch, 1, hidden], weight shape: [vocab, hidden].
// Returns logits of shape [batch, 1, vocab].
func lmHeadForward[T tensor.Numeric](
	ctx context.Context,
	engine compute.Engine[T],
	hiddenStates *tensor.TensorNumeric[T],
	lmHeadWeight *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	shape := hiddenStates.Shape()
	batch, seqLen, hidden := shape[0], shape[1], shape[2]

	flat, err := engine.Reshape(ctx, hiddenStates, []int{batch * seqLen, hidden})
	if err != nil {
		return nil, fmt.Errorf("eagle lm head reshape: %w", err)
	}

	// weight is [vocab, hidden], compute flat * weight^T = [batch*seq, vocab]
	if tb, ok := engine.(compute.TransposeBMatMuler[T]); ok {
		out, err := tb.MatMulTransposeB(ctx, flat, lmHeadWeight)
		if err != nil {
			return nil, fmt.Errorf("eagle lm head matmul: %w", err)
		}
		vocabSize := lmHeadWeight.Shape()[0]
		return engine.Reshape(ctx, out, []int{batch, seqLen, vocabSize})
	}

	wT, err := engine.Transpose(ctx, lmHeadWeight, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("eagle lm head transpose: %w", err)
	}
	out, err := engine.MatMul(ctx, flat, wT)
	if err != nil {
		return nil, fmt.Errorf("eagle lm head matmul: %w", err)
	}
	vocabSize := lmHeadWeight.Shape()[0]
	return engine.Reshape(ctx, out, []int{batch, seqLen, vocabSize})
}

// argmaxLastPos returns the argmax token ID from the last sequence position
// of a [batch, seqLen, vocab] logits tensor (batch index 0).
func argmaxLastPos[T tensor.Numeric](logits *tensor.TensorNumeric[T]) int {
	shape := logits.Shape()
	vocabSize := shape[2]
	seqLen := shape[1]
	data := logits.Data()
	offset := (seqLen - 1) * vocabSize

	maxIdx := 0
	maxVal := data[offset]
	for i := 1; i < vocabSize; i++ {
		if data[offset+i] > maxVal {
			maxVal = data[offset+i]
			maxIdx = i
		}
	}
	return maxIdx
}

// eagleTensorNames maps EAGLE head components to their GGUF tensor name
// suffixes under the "eagle." prefix. EAGLE weights can appear either as
// extra tensors in the base GGUF file (prefixed with "eagle.") or in a
// separate GGUF file (without the prefix).
var eagleTensorNames = struct {
	NormWeight string
	NormBias   string
	FC1Weight  string
	FC2Weight  string
}{
	NormWeight: "eagle.norm.weight",
	NormBias:   "eagle.norm.bias",
	FC1Weight:  "eagle.fc1.weight",
	FC2Weight:  "eagle.fc2.weight",
}

// LoadEAGLEWeights loads EAGLE head weights from a GGUF tensor map and
// returns a fully constructed EAGLEHead. It looks for tensors under the
// "eagle." prefix. If the prefix is not found, it tries unprefixed names
// (for standalone EAGLE GGUF files).
//
// The function validates that all four weight tensors are present and that
// their shapes are consistent with each other.
func LoadEAGLEWeights(
	tensors map[string]*tensor.TensorNumeric[float32],
	engine compute.Engine[float32],
	ops numeric.Arithmetic[float32],
) (*core.EAGLEHead[float32], error) {
	// Try prefixed names first (base GGUF with extra eagle tensors),
	// then unprefixed names (standalone EAGLE GGUF).
	normWeight, normBias, fc1Weight, fc2Weight, err := resolveEAGLETensors(tensors)
	if err != nil {
		return nil, fmt.Errorf("load EAGLE weights: %w", err)
	}

	// Validate shapes are consistent.
	if err := validateEAGLEShapes(normWeight, normBias, fc1Weight, fc2Weight); err != nil {
		return nil, fmt.Errorf("load EAGLE weights: %w", err)
	}

	head, err := core.NewEAGLEHeadFromWeights(engine, ops, core.EAGLEHeadWeights[float32]{
		NormGamma: normWeight,
		NormBeta:  normBias,
		FC1Weight: fc1Weight,
		FC2Weight: fc2Weight,
	})
	if err != nil {
		return nil, fmt.Errorf("load EAGLE weights: %w", err)
	}

	return head, nil
}

// HasEAGLEWeights returns true if the tensor map contains EAGLE head weights,
// either under the "eagle." prefix or as unprefixed names.
func HasEAGLEWeights(tensors map[string]*tensor.TensorNumeric[float32]) bool {
	_, ok := tensors[eagleTensorNames.NormWeight]
	if ok {
		return true
	}
	// Check unprefixed names for standalone EAGLE GGUF.
	_, ok = tensors["norm.weight"]
	return ok
}

// resolveEAGLETensors looks up the four EAGLE weight tensors, trying prefixed
// names first, then unprefixed.
func resolveEAGLETensors(
	tensors map[string]*tensor.TensorNumeric[float32],
) (normWeight, normBias, fc1Weight, fc2Weight *tensor.TensorNumeric[float32], err error) {
	type nameSet struct {
		normWeight, normBias, fc1Weight, fc2Weight string
	}

	candidates := []nameSet{
		// Prefixed: base GGUF with eagle.* extra tensors.
		{
			eagleTensorNames.NormWeight,
			eagleTensorNames.NormBias,
			eagleTensorNames.FC1Weight,
			eagleTensorNames.FC2Weight,
		},
		// Unprefixed: standalone EAGLE GGUF file.
		{"norm.weight", "norm.bias", "fc1.weight", "fc2.weight"},
	}

	for _, names := range candidates {
		nw, ok1 := tensors[names.normWeight]
		nb, ok2 := tensors[names.normBias]
		f1, ok3 := tensors[names.fc1Weight]
		f2, ok4 := tensors[names.fc2Weight]
		if ok1 && ok2 && ok3 && ok4 {
			return nw, nb, f1, f2, nil
		}
		// If some but not all are present, report the missing ones.
		if ok1 || ok2 || ok3 || ok4 {
			var missing []string
			if !ok1 {
				missing = append(missing, names.normWeight)
			}
			if !ok2 {
				missing = append(missing, names.normBias)
			}
			if !ok3 {
				missing = append(missing, names.fc1Weight)
			}
			if !ok4 {
				missing = append(missing, names.fc2Weight)
			}
			return nil, nil, nil, nil, fmt.Errorf("incomplete EAGLE weights: missing %v", missing)
		}
	}

	return nil, nil, nil, nil, fmt.Errorf("no EAGLE weights found (looked for %q and unprefixed variants)", eagleTensorNames.NormWeight)
}

// validateEAGLEShapes checks that the EAGLE weight tensor shapes are consistent.
func validateEAGLEShapes(
	normWeight, normBias, fc1Weight, fc2Weight *tensor.TensorNumeric[float32],
) error {
	// Norm weights must be 1D.
	nwShape := normWeight.Shape()
	nbShape := normBias.Shape()
	if len(nwShape) != 1 {
		return fmt.Errorf("norm.weight must be 1D, got shape %v", nwShape)
	}
	if len(nbShape) != 1 {
		return fmt.Errorf("norm.bias must be 1D, got shape %v", nbShape)
	}
	hiddenDim := nwShape[0]
	if nbShape[0] != hiddenDim {
		return fmt.Errorf("norm.weight dim %d != norm.bias dim %d", hiddenDim, nbShape[0])
	}

	// FC weights must be 2D.
	f1Shape := fc1Weight.Shape()
	f2Shape := fc2Weight.Shape()
	if len(f1Shape) != 2 {
		return fmt.Errorf("fc1.weight must be 2D, got shape %v", f1Shape)
	}
	if len(f2Shape) != 2 {
		return fmt.Errorf("fc2.weight must be 2D, got shape %v", f2Shape)
	}

	return nil
}

// GenerateDraftTokens uses an EAGLEHead to autoregressively generate draft
// token IDs from the penultimate transformer layer's hidden state.
//
// It feeds penultimateFeatures (shape [1, 1, hidden]) through the EAGLEHead,
// applies the LM head weight to get logits, takes argmax for a draft token,
// and repeats numDrafts times. Each iteration feeds the previous EAGLEHead
// output back as input for the next draft.
//
// Returns a slice of numDrafts draft token IDs.
func GenerateDraftTokens[T tensor.Numeric](
	ctx context.Context,
	eagleHead *core.EAGLEHead[T],
	engine compute.Engine[T],
	penultimateFeatures *tensor.TensorNumeric[T],
	lmHeadWeight *tensor.TensorNumeric[T],
	numDrafts int,
) ([]int, error) {
	if numDrafts <= 0 {
		return nil, fmt.Errorf("GenerateDraftTokens: numDrafts must be positive, got %d", numDrafts)
	}

	featureShape := penultimateFeatures.Shape()
	if len(featureShape) != 3 {
		return nil, fmt.Errorf("GenerateDraftTokens: penultimateFeatures must be 3D [batch, seq, hidden], got %dD", len(featureShape))
	}

	draftTokens := make([]int, 0, numDrafts)
	currentFeatures := penultimateFeatures

	for i := range numDrafts {
		// Feed through EAGLEHead to predict next hidden state.
		nextHidden, err := eagleHead.Forward(ctx, currentFeatures)
		if err != nil {
			return nil, fmt.Errorf("GenerateDraftTokens: step %d forward: %w", i, err)
		}

		// Apply LM head to get logits.
		logits, err := lmHeadForward(ctx, engine, nextHidden, lmHeadWeight)
		if err != nil {
			return nil, fmt.Errorf("GenerateDraftTokens: step %d lm head: %w", i, err)
		}

		// Argmax to get draft token.
		tokenID := argmaxLastPos(logits)
		draftTokens = append(draftTokens, tokenID)

		// Use the predicted hidden state as input for next iteration.
		currentFeatures = nextHidden
	}

	return draftTokens, nil
}
