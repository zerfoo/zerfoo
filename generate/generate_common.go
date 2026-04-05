package generate

import (
	"context"
	"fmt"
	"os"

	"github.com/zerfoo/ztensor/tensor"
)

// prefillResult holds the output of the shared prefill setup used by both
// Generate and GenerateStream.
type prefillResult[T tensor.Numeric] struct {
	genCtx       context.Context
	cacheProvider CacheProvider[T]
	tieredStore  *TieredKVStore[T] // nil unless tiered KV is enabled; caller must Close
	stopSet      map[int]bool
	generatedIDs []int
	nextToken    int
	decodeBuf    []T
	tokenTensor  *tensor.TensorNumeric[T]
}

// prefillSetup encodes the prompt, prepends BOS, selects a KV cache provider,
// builds the stop-token set, resets stateful graph nodes, runs the prefill
// forward pass, and samples the first token. Both Generate and GenerateStream
// call this to avoid duplicating the setup sequence.
func (gen *Generator[T]) prefillSetup(ctx context.Context, promptIDs []int, sc SamplingConfig) (*prefillResult[T], error) {
	// Prepend BOS token if configured.
	if gen.config.BOSTokenID > 0 {
		promptIDs = append([]int{gen.config.BOSTokenID}, promptIDs...)
	}

	cacheProvider, tieredStore, err := gen.selectCacheProvider()
	if err != nil {
		return nil, err
	}
	genCtx := WithCache(ctx, cacheProvider)

	stopSet := make(map[int]bool, len(sc.StopTokenIDs)+1)
	for _, id := range sc.StopTokenIDs {
		stopSet[id] = true
	}
	stopSet[gen.config.EOSTokenID] = true

	// Reset stateful auto-input nodes for this new generation sequence.
	gen.graph.ResetStatefulNodes()

	// Prefill: run the full prompt through the graph.
	prefillTensor, err := gen.idsToTensor(promptIDs)
	if err != nil {
		if tieredStore != nil {
			tieredStore.Close()
		}
		return nil, fmt.Errorf("create prefill tensor: %w", err)
	}

	logits, err := gen.graph.Forward(genCtx, prefillTensor)
	if err != nil {
		if tieredStore != nil {
			tieredStore.Close()
		}
		return nil, fmt.Errorf("prefill forward: %w", err)
	}

	generatedIDs := make([]int, 0, sc.MaxNewTokens)

	nextToken, err := gen.sampleFromLogits(logits, sc, generatedIDs)
	if err != nil {
		if tieredStore != nil {
			tieredStore.Close()
		}
		return nil, fmt.Errorf("sample after prefill: %w", err)
	}

	// Pre-allocate a [1,1] tensor for the decode loop.
	decodeBuf := []T{T(nextToken)}
	tokenTensor, tErr := tensor.New([]int{1, 1}, decodeBuf)
	if tErr != nil {
		if tieredStore != nil {
			tieredStore.Close()
		}
		return nil, fmt.Errorf("create decode tensor: %w", tErr)
	}

	return &prefillResult[T]{
		genCtx:        genCtx,
		cacheProvider: cacheProvider,
		tieredStore:   tieredStore,
		stopSet:       stopSet,
		generatedIDs:  generatedIDs,
		nextToken:     nextToken,
		decodeBuf:     decodeBuf,
		tokenTensor:   tokenTensor,
	}, nil
}

// syncGPUCounter syncs the GPU-side KV cache counter back to CPU after the
// decode loop completes. During CUDA graph replay the GPU counter advances
// independently; this brings the CPU-side seqLen back in sync.
func syncGPUCounter[T tensor.Numeric](cacheProvider CacheProvider[T]) {
	type counterSyncer interface {
		SyncCounterFromGPU() error
	}
	if cs, ok := cacheProvider.(counterSyncer); ok {
		if err := cs.SyncCounterFromGPU(); err != nil {
			fmt.Fprintf(os.Stderr, "warning: GPU counter sync failed: %v\n", err)
		}
	}
}
