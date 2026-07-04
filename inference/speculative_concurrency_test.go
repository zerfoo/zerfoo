package inference

import (
	"context"
	"sync"
	"testing"
)

// TestModel_SpeculativeGenerate_ConcurrentWithNormalGenerate guards against
// CONC-H1: SpeculativeGenerate must serialize with normal Generate on the
// same graph mutex (generate.Generator.LockGraph/UnlockGraph), because the
// underlying *graph.Graph is not concurrency-safe. Both m.Generate (via the
// session pool's graphMu) and m.SpeculativeGenerate (via LockGraph) run
// Forward on the exact same target graph/node instance here, so if the
// speculative path ever stops taking the lock, `go test -race` will catch a
// data race on the shared fixedLogitsNode state, and — absent -race —
// concurrent unsynchronized Forward calls could otherwise corrupt output.
func TestModel_SpeculativeGenerate_ConcurrentWithNormalGenerate(t *testing.T) {
	vocabSize := 8
	target := buildTestModel(t, vocabSize, []int{6, 7, 2})
	draft := buildTestModel(t, vocabSize, []int{6, 7, 2})

	const iterations = 20

	var wg sync.WaitGroup
	errCh := make(chan error, iterations*2)
	resCh := make(chan string, iterations*2)

	for range iterations {
		wg.Add(2)

		go func() {
			defer wg.Done()
			res, err := target.Generate(context.Background(), "hello world",
				WithTemperature(0), WithMaxTokens(10))
			errCh <- err
			resCh <- res
		}()

		go func() {
			defer wg.Done()
			res, err := target.SpeculativeGenerate(context.Background(), draft, "hello world", 4,
				WithTemperature(0), WithMaxTokens(10))
			errCh <- err
			resCh <- res
		}()
	}

	wg.Wait()
	close(errCh)
	close(resCh)

	for err := range errCh {
		if err != nil {
			t.Fatalf("unexpected error from concurrent generate: %v", err)
		}
	}
	for res := range resCh {
		if res == "" {
			t.Errorf("expected a well-formed, non-empty generation result, got empty string")
		}
	}
}
