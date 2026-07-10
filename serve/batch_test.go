package serve

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBatchScheduler_BatchesRequests(t *testing.T) {
	var batchCount atomic.Int32

	sched := NewBatchScheduler(BatchConfig{
		MaxBatchSize: 4,
		BatchTimeout: 50 * time.Millisecond,
		Handler: func(_ context.Context, reqs []BatchRequest) []BatchResult {
			batchCount.Add(1)
			results := make([]BatchResult, len(reqs))
			for i, r := range reqs {
				results[i] = BatchResult{Value: "echo:" + r.Prompt}
			}
			return results
		},
	})

	sched.Start()
	defer sched.Stop()

	// Submit 4 requests concurrently — should be batched into 1 call.
	var wg sync.WaitGroup
	results := make([]BatchResult, 4)
	for i := range 4 {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			r, err := sched.Submit(context.Background(), BatchRequest{Prompt: "hello"})
			if err != nil {
				t.Errorf("submit %d: %v", idx, err)
				return
			}
			results[idx] = r
		}(i)
	}

	wg.Wait()

	for i, r := range results {
		if r.Value != "echo:hello" {
			t.Errorf("result[%d] = %q, want %q", i, r.Value, "echo:hello")
		}
	}

	if got := batchCount.Load(); got != 1 {
		t.Errorf("batch count = %d, want 1", got)
	}
}

func TestBatchScheduler_TimeoutFires(t *testing.T) {
	var batchSizes []int
	var mu sync.Mutex

	sched := NewBatchScheduler(BatchConfig{
		MaxBatchSize: 8,
		BatchTimeout: 20 * time.Millisecond,
		Handler: func(_ context.Context, reqs []BatchRequest) []BatchResult {
			mu.Lock()
			batchSizes = append(batchSizes, len(reqs))
			mu.Unlock()
			results := make([]BatchResult, len(reqs))
			for i := range reqs {
				results[i] = BatchResult{Value: "ok"}
			}
			return results
		},
	})

	sched.Start()
	defer sched.Stop()

	// Submit 2 requests (below max batch size). Timeout should fire.
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = sched.Submit(context.Background(), BatchRequest{Prompt: "test"})
		}()
	}

	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	if len(batchSizes) != 1 || batchSizes[0] != 2 {
		t.Errorf("batch sizes = %v, want [2]", batchSizes)
	}
}

func TestBatchScheduler_MaxBatchSizeEnforced(t *testing.T) {
	var batchSizes []int
	var mu sync.Mutex

	sched := NewBatchScheduler(BatchConfig{
		MaxBatchSize: 2,
		BatchTimeout: 100 * time.Millisecond,
		Handler: func(_ context.Context, reqs []BatchRequest) []BatchResult {
			mu.Lock()
			batchSizes = append(batchSizes, len(reqs))
			mu.Unlock()
			results := make([]BatchResult, len(reqs))
			for i := range reqs {
				results[i] = BatchResult{Value: "ok"}
			}
			return results
		},
	})

	sched.Start()
	defer sched.Stop()

	// Submit 4 requests with max batch size 2 → should get 2 batches.
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = sched.Submit(context.Background(), BatchRequest{Prompt: "test"})
		}()
	}

	wg.Wait()

	mu.Lock()
	defer mu.Unlock()

	total := 0
	for _, s := range batchSizes {
		if s > 2 {
			t.Errorf("batch size %d exceeds max 2", s)
		}
		total += s
	}
	if total != 4 {
		t.Errorf("total requests processed = %d, want 4", total)
	}
}

func TestBatchScheduler_ContextCancellation(t *testing.T) {
	sched := NewBatchScheduler(BatchConfig{
		MaxBatchSize: 8,
		BatchTimeout: time.Second,
		Handler: func(_ context.Context, reqs []BatchRequest) []BatchResult {
			results := make([]BatchResult, len(reqs))
			return results
		},
	})

	sched.Start()
	defer sched.Stop()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	_, err := sched.Submit(ctx, BatchRequest{Prompt: "test"})
	if err == nil {
		t.Error("expected error from canceled context")
	}
}

func TestBatchScheduler_FirstDisconnectDoesNotCancelBatch(t *testing.T) {
	handlerCalled := make(chan struct{})
	handlerCtx := make(chan context.Context, 1)

	sched := NewBatchScheduler(BatchConfig{
		MaxBatchSize: 4,
		BatchTimeout: 50 * time.Millisecond,
		Handler: func(ctx context.Context, reqs []BatchRequest) []BatchResult {
			handlerCtx <- ctx
			close(handlerCalled)
			// Simulate work — give time for first request to cancel.
			time.Sleep(100 * time.Millisecond)
			results := make([]BatchResult, len(reqs))
			for i, r := range reqs {
				results[i] = BatchResult{Value: "done:" + r.Prompt}
			}
			return results
		},
	})

	sched.Start()
	defer sched.Stop()

	// Request 0: will be cancelled immediately after batch starts.
	ctx0, cancel0 := context.WithCancel(context.Background())
	// Requests 1 and 2: stay alive.
	ctx1 := context.Background()
	ctx2 := context.Background()

	var wg sync.WaitGroup
	errs := make([]error, 3)
	vals := make([]string, 3)

	for i, ctx := range []context.Context{ctx0, ctx1, ctx2} {
		wg.Add(1)
		go func(idx int, c context.Context) {
			defer wg.Done()
			r, err := sched.Submit(c, BatchRequest{Prompt: "req"})
			errs[idx] = err
			vals[idx] = r.Value
		}(i, ctx)
	}

	// Wait for handler to be invoked, then cancel the first request.
	<-handlerCalled
	cancel0()

	// The batch context should NOT be cancelled — two requests are still alive.
	bctx := <-handlerCtx
	// Give a moment for the cancellation goroutine to run.
	time.Sleep(20 * time.Millisecond)
	if bctx.Err() != nil {
		t.Fatalf("batch context cancelled after first request disconnect: %v", bctx.Err())
	}

	wg.Wait()

	// Requests 1 and 2 must succeed.
	for _, idx := range []int{1, 2} {
		if errs[idx] != nil {
			t.Errorf("request %d error: %v", idx, errs[idx])
		}
		if vals[idx] != "done:req" {
			t.Errorf("request %d value = %q, want %q", idx, vals[idx], "done:req")
		}
	}
}

// TestBatchScheduler_ExecuteBatch_ReapsCtxWaitGoroutines guards against
// CONC-L1: executeBatch spawns one ctx-wait goroutine per live request to
// build a merged batch context. Each goroutine used to block solely on its
// own request's ctx.Done(), so if a request's context is never
// independently canceled (e.g. context.Background(), or a long-lived
// streaming context that outlives the batch), the goroutine lingered long
// after the batch itself had already completed. The fix makes each
// goroutine also select on batchCtx.Done() so it is reaped as soon as the
// batch returns.
func TestBatchScheduler_ExecuteBatch_ReapsCtxWaitGoroutines(t *testing.T) {
	const n = 6

	sched := NewBatchScheduler(BatchConfig{
		MaxBatchSize: n,
		BatchTimeout: 200 * time.Millisecond,
		Handler: func(_ context.Context, reqs []BatchRequest) []BatchResult {
			results := make([]BatchResult, len(reqs))
			for i := range reqs {
				results[i] = BatchResult{Value: "ok"}
			}
			return results
		},
	})

	sched.Start()
	defer sched.Stop()

	// Warm up so the scheduler's own steady-state goroutines are already
	// running before we take the baseline measurement.
	if _, err := sched.Submit(context.Background(), BatchRequest{Prompt: "warmup"}); err != nil {
		t.Fatalf("warmup submit: %v", err)
	}
	runtime.Gosched()
	time.Sleep(50 * time.Millisecond)
	runtime.GC()
	baseline := runtime.NumGoroutine()

	// Submit n requests concurrently, all using context.Background() so
	// nothing ever independently cancels their context. The only thing that
	// should stop the per-request ctx-wait goroutine is the batch itself
	// completing.
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			if _, err := sched.Submit(context.Background(), BatchRequest{Prompt: "req"}); err != nil {
				t.Errorf("submit %d: %v", idx, err)
			}
		}(i)
	}
	wg.Wait()

	// The batch handler has already returned by the time Submit unblocks its
	// callers (results are delivered after the handler call), so the
	// ctx-wait goroutines should be reaped almost immediately. Poll with a
	// generous deadline; without the fix these goroutines never exit
	// (context.Background() never fires Done()), so this will time out and
	// fail instead of flaking.
	deadline := time.Now().Add(2 * time.Second)
	var final int
	for {
		runtime.Gosched()
		time.Sleep(10 * time.Millisecond)
		runtime.GC()
		final = runtime.NumGoroutine()
		if final <= baseline+1 { // small slack for test/runtime bookkeeping goroutines
			break
		}
		if time.Now().After(deadline) {
			break
		}
	}

	if final > baseline+1 {
		t.Errorf("goroutine count did not settle back to baseline after batch completed: baseline=%d, final=%d (leaked ~%d ctx-wait goroutine(s))",
			baseline, final, final-baseline)
	}
}

func TestBatchScheduler_HTTPIntegration(t *testing.T) {
	m := buildTestModel(t)

	bs := NewBatchScheduler(BatchConfig{
		MaxBatchSize: 4,
		BatchTimeout: 100 * time.Millisecond,
	})
	srv := NewServer(m, WithBatchScheduler(bs))
	bs.Start()
	defer bs.Stop()

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	const n = 4
	type result struct {
		status int
		body   string
	}
	results := make([]result, n)
	var wg sync.WaitGroup
	wg.Add(n)

	for i := range n {
		go func(idx int) {
			defer wg.Done()
			payload := `{"model":"test-model","messages":[{"role":"user","content":"hello world"}]}`
			req, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
				ts.URL+"/v1/chat/completions", strings.NewReader(payload))
			if err != nil {
				t.Errorf("request %d: %v", idx, err)
				return
			}
			req.Header.Set("Content-Type", "application/json")
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Errorf("request %d: %v", idx, err)
				return
			}
			defer resp.Body.Close()
			body, _ := io.ReadAll(resp.Body)
			results[idx] = result{status: resp.StatusCode, body: string(body)}
		}(i)
	}

	wg.Wait()

	for i, r := range results {
		if r.status != http.StatusOK {
			t.Errorf("request %d: status = %d, want %d; body: %s", i, r.status, http.StatusOK, r.body)
			continue
		}
		var resp ChatCompletionResponse
		if err := json.Unmarshal([]byte(r.body), &resp); err != nil {
			t.Errorf("request %d: unmarshal: %v", i, err)
			continue
		}
		if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
			t.Errorf("request %d: empty content in response", i)
		}
	}
}
