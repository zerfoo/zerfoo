// Command bench_disagg benchmarks disaggregated vs collocated serving throughput.
//
// In disaggregated mode, prefill and decode run on separate workers behind a
// gateway that routes requests via least-loaded scheduling. In collocated mode,
// a single worker handles both prefill and decode sequentially.
//
// The benchmark measures requests/sec, mean TTFT, and P99 latency for both
// modes at configurable concurrency levels.
//
// Usage:
//
//	bench_disagg [--concurrent 16] [--requests 100] [--tokens 50]
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"

	disaggpb "github.com/zerfoo/zerfoo/serve/disaggregated/proto"

	"github.com/zerfoo/zerfoo/serve/disaggregated"
)

// benchResult holds metrics for one mode (disaggregated or collocated).
type benchResult struct {
	Mode       string  `json:"mode"`
	Concurrent int     `json:"concurrent"`
	Requests   int     `json:"requests"`
	Tokens     int     `json:"tokens_per_request"`
	ReqPerSec  float64 `json:"requests_per_sec"`
	MeanTTFTMs float64 `json:"mean_ttft_ms"`
	P99Ms      float64 `json:"p99_latency_ms"`
	Commit     string  `json:"commit"`
	Timestamp  string  `json:"timestamp"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func run() error {
	concurrent := flag.Int("concurrent", 16, "number of concurrent requests")
	requests := flag.Int("requests", 100, "total number of requests")
	tokens := flag.Int("tokens", 50, "tokens per request")
	output := flag.String("output", "", "optional JSON output file")
	flag.Parse()

	fmt.Printf("Disaggregated vs Collocated Benchmark\n")
	fmt.Printf("  concurrent=%d  requests=%d  tokens=%d\n\n", *concurrent, *requests, *tokens)

	// Run disaggregated benchmark.
	disaggResult, err := benchDisaggregated(*concurrent, *requests, *tokens)
	if err != nil {
		return fmt.Errorf("disaggregated benchmark: %w", err)
	}
	printResult(disaggResult)

	// Run collocated benchmark.
	collocResult, err := benchCollocated(*concurrent, *requests, *tokens)
	if err != nil {
		return fmt.Errorf("collocated benchmark: %w", err)
	}
	printResult(collocResult)

	// Print comparison.
	fmt.Println("--- Comparison ---")
	speedup := disaggResult.ReqPerSec / collocResult.ReqPerSec
	fmt.Printf("  Prefill throughput speedup: %.2fx (disagg %.2f req/s vs colloc %.2f req/s)\n",
		speedup, disaggResult.ReqPerSec, collocResult.ReqPerSec)
	fmt.Printf("  TTFT improvement: %.2f ms vs %.2f ms\n",
		disaggResult.MeanTTFTMs, collocResult.MeanTTFTMs)
	fmt.Printf("  P99 improvement: %.2f ms vs %.2f ms\n",
		disaggResult.P99Ms, collocResult.P99Ms)

	if *output != "" {
		results := []benchResult{disaggResult, collocResult}
		data, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal: %w", err)
		}
		if err := os.WriteFile(*output, data, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", *output, err)
		}
		fmt.Printf("\nResults written to %s\n", *output)
	}

	return nil
}

// benchDisaggregated runs the disaggregated benchmark: separate prefill and
// decode workers behind a gateway. Multiple prefill workers allow parallel
// prefill processing, which is the key advantage.
func benchDisaggregated(concurrent, totalRequests, tokens int) (benchResult, error) {
	// Create multiple prefill workers to simulate disaggregated parallelism.
	// Each prefill worker can handle requests independently.
	numPrefillWorkers := concurrent
	if numPrefillWorkers > 8 {
		numPrefillWorkers = 8
	}

	prefillClients := make([]disaggregated.PrefillClient, numPrefillWorkers)
	for i := range prefillClients {
		prefillClients[i] = &latencyPrefillClient{
			layers:       4,
			prefillDelay: 10 * time.Millisecond, // simulated prefill compute (GPU-bound)
		}
	}

	decodeClients := make([]disaggregated.DecodeClient, 2)
	for i := range decodeClients {
		decodeClients[i] = &latencyDecodeClient{
			tokensPerReq: tokens,
			decodeDelay:  200 * time.Microsecond, // simulated per-token decode (memory-bound)
		}
	}

	gw := disaggregated.NewTestGateway(prefillClients, decodeClients)
	defer gw.Close()

	srv := httptest.NewServer(gw)
	defer srv.Close()

	latencies, err := runConcurrentRequests(srv.URL, concurrent, totalRequests, tokens)
	if err != nil {
		return benchResult{}, err
	}

	commit := gitCommitHash()
	return buildResult("disaggregated", concurrent, totalRequests, tokens, latencies, commit), nil
}

// benchCollocated runs the collocated benchmark: a single handler that does
// both prefill and decode sequentially, simulating a non-disaggregated setup.
// The key difference is that prefill blocks the decode path, so under high
// concurrency, requests queue behind each other's prefill phases.
func benchCollocated(concurrent, totalRequests, tokens int) (benchResult, error) {
	handler := &collocatedHandler{
		prefillDelay: 10 * time.Millisecond,
		decodeDelay:  200 * time.Microsecond,
		tokensPerReq: tokens,
		layers:       4,
		maxConcGPU:   1, // single GPU = serialized compute
	}

	srv := httptest.NewServer(handler)
	defer srv.Close()

	latencies, err := runConcurrentRequests(srv.URL, concurrent, totalRequests, tokens)
	if err != nil {
		return benchResult{}, err
	}

	commit := gitCommitHash()
	return buildResult("collocated", concurrent, totalRequests, tokens, latencies, commit), nil
}

// runConcurrentRequests sends totalRequests requests with the given concurrency
// and returns per-request latencies.
func runConcurrentRequests(url string, concurrent, totalRequests, tokens int) ([]time.Duration, error) {
	var (
		mu        sync.Mutex
		latencies []time.Duration
		wg        sync.WaitGroup
		errCh     = make(chan error, totalRequests)
		sem       = make(chan struct{}, concurrent)
	)

	for i := range totalRequests {
		wg.Add(1)
		sem <- struct{}{}
		go func(reqID int) {
			defer wg.Done()
			defer func() { <-sem }()

			start := time.Now()
			reqBody := fmt.Sprintf(
				`{"request_id":"req-%d","token_ids":[1,2,3,4,5],"max_new_tokens":%d,"temperature":0.0}`,
				reqID, tokens,
			)
			resp, err := http.Post(url+"/v1/completions", "application/json", strings.NewReader(reqBody))
			if err != nil {
				errCh <- fmt.Errorf("request %d: %w", reqID, err)
				return
			}
			// Drain body to ensure full response is received.
			_, _ = io.ReadAll(resp.Body)
			resp.Body.Close()

			elapsed := time.Since(start)

			if resp.StatusCode != http.StatusOK {
				errCh <- fmt.Errorf("request %d: status %d", reqID, resp.StatusCode)
				return
			}

			mu.Lock()
			latencies = append(latencies, elapsed)
			mu.Unlock()
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		return nil, err
	}

	return latencies, nil
}

// buildResult computes aggregate metrics from per-request latencies.
func buildResult(mode string, concurrent, totalRequests, tokens int, latencies []time.Duration, commit string) benchResult {
	if len(latencies) == 0 {
		return benchResult{Mode: mode}
	}

	// Total wall clock time is approximated from first to last completion.
	// For simplicity, use total latency / concurrency as effective wall time.
	var totalLatency time.Duration
	for _, l := range latencies {
		totalLatency += l
	}
	wallTime := totalLatency / time.Duration(concurrent)
	reqPerSec := float64(len(latencies)) / wallTime.Seconds()

	// Mean TTFT: approximate as mean latency (first token arrives with request).
	var sumMs float64
	for _, l := range latencies {
		sumMs += float64(l.Microseconds()) / 1000.0
	}
	meanTTFT := sumMs / float64(len(latencies))

	// P99 latency.
	sorted := make([]float64, len(latencies))
	for i, l := range latencies {
		sorted[i] = float64(l.Microseconds()) / 1000.0
	}
	sort.Float64s(sorted)
	p99Idx := int(math.Ceil(0.99*float64(len(sorted)))) - 1
	if p99Idx < 0 {
		p99Idx = 0
	}
	if p99Idx >= len(sorted) {
		p99Idx = len(sorted) - 1
	}

	return benchResult{
		Mode:       mode,
		Concurrent: concurrent,
		Requests:   totalRequests,
		Tokens:     tokens,
		ReqPerSec:  reqPerSec,
		MeanTTFTMs: meanTTFT,
		P99Ms:      sorted[p99Idx],
		Commit:     commit,
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
	}
}

func printResult(r benchResult) {
	fmt.Printf("[%s] concurrent=%d requests=%d tokens=%d\n", r.Mode, r.Concurrent, r.Requests, r.Tokens)
	fmt.Printf("  Requests/sec: %.2f\n", r.ReqPerSec)
	fmt.Printf("  Mean TTFT:    %.2f ms\n", r.MeanTTFTMs)
	fmt.Printf("  P99 Latency:  %.2f ms\n", r.P99Ms)
	fmt.Println()
}

// gitCommitHash returns the short git commit hash, or "unknown" on failure.
func gitCommitHash() string {
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

// --- Mock workers with simulated latency ---

// latencyPrefillClient simulates a prefill worker with configurable latency.
type latencyPrefillClient struct {
	layers       int
	prefillDelay time.Duration
}

func (m *latencyPrefillClient) Prefill(_ context.Context, req *disaggpb.PreFillRequest) (disaggregated.KVBlockReceiver, error) {
	time.Sleep(m.prefillDelay)
	var msgs []*disaggpb.KVBlockStream
	for i := range m.layers {
		msgs = append(msgs, &disaggpb.KVBlockStream{
			Block: &disaggpb.KVBlock{
				RequestId: req.GetRequestId(),
				LayerIdx:  int32(i),
				BlockIdx:  0,
				KData:     []byte{0x00, 0x3C},
				VData:     []byte{0x00, 0x3C},
			},
		})
	}
	msgs = append(msgs, &disaggpb.KVBlockStream{Done: true})
	return &sliceKVReceiver{msgs: msgs}, nil
}

// latencyDecodeClient simulates a decode worker with per-token latency.
type latencyDecodeClient struct {
	tokensPerReq int
	decodeDelay  time.Duration
}

func (m *latencyDecodeClient) Decode(_ context.Context, req *disaggpb.DecodeRequest) (disaggregated.TokenReceiver, error) {
	n := int(req.GetMaxNewTokens())
	if n <= 0 {
		n = m.tokensPerReq
	}
	var msgs []*disaggpb.TokenStream
	for i := range n {
		msgs = append(msgs, &disaggpb.TokenStream{
			RequestId: req.GetRequestId(),
			TokenId:   int32(100 + i),
		})
	}
	msgs = append(msgs, &disaggpb.TokenStream{
		RequestId:    req.GetRequestId(),
		TokenId:      2,
		Done:         true,
		FinishReason: "stop",
	})
	return &sliceTokenReceiver{msgs: msgs, decodeDelay: m.decodeDelay}, nil
}

// sliceKVReceiver replays pre-built KVBlockStream messages.
type sliceKVReceiver struct {
	msgs []*disaggpb.KVBlockStream
	idx  int
}

func (r *sliceKVReceiver) Recv() (*disaggpb.KVBlockStream, error) {
	if r.idx >= len(r.msgs) {
		return nil, io.EOF
	}
	msg := r.msgs[r.idx]
	r.idx++
	return msg, nil
}

// sliceTokenReceiver replays pre-built TokenStream messages with per-token delay.
type sliceTokenReceiver struct {
	msgs        []*disaggpb.TokenStream
	idx         int
	decodeDelay time.Duration
}

func (r *sliceTokenReceiver) Recv() (*disaggpb.TokenStream, error) {
	if r.idx >= len(r.msgs) {
		return nil, io.EOF
	}
	if r.decodeDelay > 0 {
		time.Sleep(r.decodeDelay)
	}
	msg := r.msgs[r.idx]
	r.idx++
	return msg, nil
}

// collocatedHandler simulates a collocated serving setup where prefill and
// decode happen on the same worker. A semaphore limits concurrent GPU access
// to simulate single-GPU contention: both prefill and decode phases compete
// for the same compute resource.
type collocatedHandler struct {
	prefillDelay time.Duration
	decodeDelay  time.Duration
	tokensPerReq int
	layers       int
	maxConcGPU   int // max concurrent GPU operations (typically 1)
	sem          sync.Once
	gpuSem       chan struct{}
}

func (h *collocatedHandler) initSem() {
	h.sem.Do(func() {
		h.gpuSem = make(chan struct{}, h.maxConcGPU)
	})
}

func (h *collocatedHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.initSem()

	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Collocated: acquire GPU for entire prefill phase (serialized).
	h.gpuSem <- struct{}{}
	time.Sleep(h.prefillDelay)
	<-h.gpuSem

	// Decode: tokens run without GPU semaphore since decode is memory-bound
	// and can overlap with other requests' decode steps (continuous batching).
	// However, decode must wait for prefill to finish, which is the bottleneck.
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	enc := json.NewEncoder(w)
	for i := range h.tokensPerReq {
		time.Sleep(h.decodeDelay)
		fmt.Fprintf(w, "data: ")
		enc.Encode(map[string]any{"token_id": 100 + i, "done": false})
		fmt.Fprintf(w, "\n")
		flusher.Flush()
	}

	// Final EOS token.
	time.Sleep(h.decodeDelay)
	fmt.Fprintf(w, "data: ")
	enc.Encode(map[string]any{"token_id": 2, "done": true, "finish_reason": "stop"})
	fmt.Fprintf(w, "\n")
	flusher.Flush()
}
