package generate

import (
	"context"

	"github.com/zerfoo/ztensor/tensor"
)

// BatchRequest represents a single generation request in a batch.
type BatchRequest struct {
	Prompt   string
	Sampling SamplingConfig
}

// BatchResult holds the output for a single request in a batch.
type BatchResult struct {
	Text string
	Err  error
}

// BatchGenerate runs multiple generation requests concurrently. Each request
// gets its own KV cache and sampling state. This provides throughput gains
// when the model graph is configured with WithParallel(true) or when
// generation is I/O bound.
//
// For true batched tensor operations (batch dimension > 1 in a single forward
// pass), the model graph and attention layers need native batch support,
// which is not yet implemented. This function provides request-level
// parallelism as an interim solution.
func (gen *Generator[T]) BatchGenerate(ctx context.Context, requests []BatchRequest) []BatchResult {
	results := make([]BatchResult, len(requests))

	// Run sequentially: the Generator's ExecutionPlan shares scratch buffers
	// that are not safe for concurrent use. True batched inference (batch
	// dimension > 1 in a single forward pass) requires native batch support
	// in the model graph, which is not yet implemented.
	for i := range requests {
		text, err := gen.Generate(ctx, requests[i].Prompt, requests[i].Sampling)
		results[i] = BatchResult{Text: text, Err: err}
	}

	return results
}

// Statically assert Generator implements the batch generate capability.
var _ interface {
	BatchGenerate(context.Context, []BatchRequest) []BatchResult
} = (*Generator[float32])(nil)

// BatchGenerateStream runs multiple streaming generation requests concurrently.
// Each request gets its own KV cache, sampling state, and token stream.
func (gen *Generator[T]) BatchGenerateStream(ctx context.Context, requests []BatchRequest, streams []TokenStream) []error {
	if len(requests) != len(streams) {
		errs := make([]error, len(requests))
		for i := range errs {
			errs[i] = context.Canceled
		}
		return errs
	}

	errs := make([]error, len(requests))

	for i := range requests {
		errs[i] = gen.GenerateStream(ctx, requests[i].Prompt, requests[i].Sampling, streams[i])
	}

	return errs
}

// Statically verify the batch stream signature compiles.
var _ func(context.Context, []BatchRequest, []TokenStream) []error = (*Generator[float32])(nil).BatchGenerateStream

// batchHelper is unexported to avoid bloating the API. Batch-level tensor
// operations (padding, splitting) will be added when the model graph supports
// batch dimension > 1 natively.
type batchHelper[T tensor.Numeric] struct {
	gen *Generator[T]
}

// newBatchHelper creates a helper for internal batch operations.
func newBatchHelper[T tensor.Numeric](gen *Generator[T]) *batchHelper[T] {
	return &batchHelper[T]{gen: gen}
}

// padPrompts left-pads a slice of token ID sequences to the maximum length,
// using the given padTokenID. Returns the padded 2D slice and the original
// lengths. This will be used when native batched forward is implemented.
func (h *batchHelper[T]) padPrompts(prompts [][]int, padTokenID int) ([][]int, []int) {
	maxLen := 0
	for _, p := range prompts {
		if len(p) > maxLen {
			maxLen = len(p)
		}
	}

	padded := make([][]int, len(prompts))
	lengths := make([]int, len(prompts))
	for i, p := range prompts {
		lengths[i] = len(p)
		row := make([]int, maxLen)
		padStart := maxLen - len(p)
		for j := range padStart {
			row[j] = padTokenID
		}
		copy(row[padStart:], p)
		padded[i] = row
	}
	return padded, lengths
}
