package generate

import (
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// tryGPUArgmax attempts greedy argmax entirely on GPU, returning the token index.
// It returns (token, true) on success, or (0, false) if the fast path does not apply.
// This avoids copying ~1MB of logits back to CPU for greedy decoding.
func tryGPUArgmax[T tensor.Numeric](
	logits *tensor.TensorNumeric[T],
	engine compute.Engine[T],
	sc SamplingConfig,
) (int, bool) {
	if sc.GrammarState != nil || sc.Temperature > 0 {
		return 0, false
	}
	if sc.RepetitionPenalty > 0 && sc.RepetitionPenalty != 1.0 {
		return 0, false
	}
	shape := logits.Shape()
	seqLen := shape[1]
	if seqLen != 1 {
		return 0, false
	}
	if _, ok := logits.GetStorage().(*tensor.GPUStorage[T]); !ok {
		return 0, false
	}
	am, ok := engine.(compute.GPUArgmaxer)
	if !ok {
		return 0, false
	}
	f32t, ok := any(logits).(*tensor.TensorNumeric[float32])
	if !ok {
		return 0, false
	}
	idx, err := am.GPUArgmax(f32t)
	if err != nil {
		return 0, false
	}
	return idx, true
}

// copyLogitsToCPU copies the logits tensor data into a CPU-side []T buffer,
// handling both GPU and CPU storage transparently.
func copyLogitsToCPU[T tensor.Numeric](logits *tensor.TensorNumeric[T], seqLen, vocabSize int) ([]T, error) {
	totalElems := seqLen * vocabSize
	data := make([]T, totalElems)
	if gs, ok := logits.GetStorage().(*tensor.GPUStorage[T]); ok {
		if err := gs.CopyTo(data); err != nil {
			return nil, fmt.Errorf("copy logits from GPU: %w", err)
		}
	} else {
		copy(data, logits.Data())
	}
	return data, nil
}

// applyTemperatureAndTopP applies temperature scaling, top-K filtering, top-P
// (nucleus) filtering, and samples from the resulting distribution. If
// temperature is zero or negative, it returns argmax instead.
func applyTemperatureAndTopP(logitsF64 []float64, sc SamplingConfig, vocabSize int) int {
	if sc.Temperature <= 0 {
		return argmax(logitsF64)
	}
	applyTemperature(logitsF64, sc.Temperature)
	if sc.TopK > 0 && sc.TopK < vocabSize {
		applyTopK(logitsF64, sc.TopK)
	}
	if sc.TopP > 0 && sc.TopP < 1.0 {
		applyTopP(logitsF64, sc.TopP)
	}
	return sampleFromDistribution(logitsF64)
}
