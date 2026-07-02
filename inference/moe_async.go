package inference

import (
	"context"
	"fmt"
	"sync"

	"github.com/zerfoo/zerfoo/internal/xblas"
)

// ExpertWork describes a single CPU expert GEMM to be dispatched asynchronously.
// The caller provides the expert weight matrix and input; the dispatcher writes
// results into Output.
type ExpertWork struct {
	ExpertID int
	// Weight is the expert FFN weight matrix, row-major [outDim, inDim].
	Weight []float32
	// Input is the token hidden state, row-major [numTokens, inDim].
	Input []float32
	// Output is pre-allocated by the caller, row-major [numTokens, outDim].
	Output []float32
	// M, N, K are the GEMM dimensions: M=numTokens, N=outDim, K=inDim.
	M, N, K int
}

// AsyncExpertDispatcher dispatches CPU-resident expert GEMMs to a goroutine
// pool so that they execute concurrently with GPU work on shared experts.
//
// Usage:
//  1. Create with [NewAsyncExpertDispatcher].
//  2. Call [Dispatch] with CPU expert work items — returns immediately.
//  3. Run GPU shared-expert work concurrently in the calling goroutine.
//  4. Call [Wait] to block until all CPU experts finish and collect errors.
//  5. Call [Shutdown] when the dispatcher is no longer needed.
type AsyncExpertDispatcher struct {
	workers int
	pool    *sync.Pool // reuse WaitGroups

	mu      sync.Mutex
	firstErr error
	wg      sync.WaitGroup
}

// NewAsyncExpertDispatcher creates a dispatcher with the given number of
// worker goroutines. If workers <= 0, it defaults to 4.
func NewAsyncExpertDispatcher(workers int) *AsyncExpertDispatcher {
	if workers <= 0 {
		workers = 4
	}
	xblas.InitPool(workers)
	return &AsyncExpertDispatcher{
		workers: workers,
	}
}

// Dispatch submits CPU expert work items for asynchronous execution.
// It returns immediately. Call [Wait] to block until all items complete.
// The context is checked before launching each work item; if already
// cancelled, remaining items are skipped.
func (d *AsyncExpertDispatcher) Dispatch(ctx context.Context, items []ExpertWork) {
	if len(items) == 0 {
		return
	}

	tasks := make([]func(), 0, len(items))
	for i := range items {
		item := &items[i]
		if err := ctx.Err(); err != nil {
			d.setError(fmt.Errorf("moe_async: context cancelled before expert %d: %w", item.ExpertID, err))
			return
		}
		if err := validateWork(item); err != nil {
			d.setError(err)
			return
		}
		d.wg.Add(1)
		tasks = append(tasks, func() {
			defer d.wg.Done()
			if err := ctx.Err(); err != nil {
				d.setError(fmt.Errorf("moe_async: context cancelled during expert %d: %w", item.ExpertID, err))
				return
			}
			// C = Input @ Weight^T  =>  GEMM(M, N, K, Input, Weight^T, Output)
			// xblas.GemmF32 computes C = A * B where A is (M,K) and B is (K,N).
			// Weight is [outDim, inDim] i.e. (N, K) row-major, so Weight^T is (K, N).
			// We need B = Weight^T which is (K, N).
			// Since Weight is stored row-major as [N][K], reading it as a (K, N)
			// column-major matrix is equivalent to the transpose. However, xblas
			// expects row-major, so we must explicitly transpose.
			wT := transposeF32(item.N, item.K, item.Weight)
			xblas.GemmF32(item.M, item.N, item.K, item.Input, wT, item.Output)
		})
	}

	// Submit all tasks to the xblas worker pool. The pool's Submit blocks
	// until all tasks complete internally, so we wrap it in a goroutine to
	// keep Dispatch non-blocking.
	go func() {
		submitTasks(tasks)
	}()
}

// submitTasks runs each task sequentially. This is used when the xblas pool
// is not directly accessible for batch submission. Each task already calls
// wg.Done on the dispatcher's WaitGroup.
func submitTasks(tasks []func()) {
	var wg sync.WaitGroup
	wg.Add(len(tasks))
	for _, t := range tasks {
		task := t
		go func() {
			defer wg.Done()
			task()
		}()
	}
	wg.Wait()
}

// Wait blocks until all dispatched work items complete. It returns the first
// error encountered, if any.
func (d *AsyncExpertDispatcher) Wait() error {
	d.wg.Wait()
	d.mu.Lock()
	err := d.firstErr
	d.firstErr = nil
	d.mu.Unlock()
	return err
}

// Shutdown releases resources. After Shutdown, the dispatcher must not be reused.
func (d *AsyncExpertDispatcher) Shutdown() {
	d.wg.Wait()
}

// setError records the first error encountered.
func (d *AsyncExpertDispatcher) setError(err error) {
	d.mu.Lock()
	if d.firstErr == nil {
		d.firstErr = err
	}
	d.mu.Unlock()
}

// validateWork checks that an ExpertWork item has consistent dimensions.
func validateWork(w *ExpertWork) error {
	if w.M <= 0 || w.N <= 0 || w.K <= 0 {
		return fmt.Errorf("moe_async: expert %d: invalid dimensions M=%d N=%d K=%d", w.ExpertID, w.M, w.N, w.K)
	}
	if len(w.Input) < w.M*w.K {
		return fmt.Errorf("moe_async: expert %d: input length %d < M*K=%d", w.ExpertID, len(w.Input), w.M*w.K)
	}
	if len(w.Weight) < w.N*w.K {
		return fmt.Errorf("moe_async: expert %d: weight length %d < N*K=%d", w.ExpertID, len(w.Weight), w.N*w.K)
	}
	if len(w.Output) < w.M*w.N {
		return fmt.Errorf("moe_async: expert %d: output length %d < M*N=%d", w.ExpertID, len(w.Output), w.M*w.N)
	}
	return nil
}

// transposeF32 transposes a row-major (rows, cols) matrix to (cols, rows).
func transposeF32(rows, cols int, src []float32) []float32 {
	dst := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			dst[c*rows+r] = src[r*cols+c]
		}
	}
	return dst
}

// SerialExpertDispatch runs the same expert GEMM work items serially on the
// calling goroutine. This is the reference implementation used to verify that
// [AsyncExpertDispatcher] produces identical results.
func SerialExpertDispatch(items []ExpertWork) error {
	for i := range items {
		item := &items[i]
		if err := validateWork(item); err != nil {
			return err
		}
		wT := transposeF32(item.N, item.K, item.Weight)
		xblas.GemmF32(item.M, item.N, item.K, item.Input, wT, item.Output)
	}
	return nil
}
