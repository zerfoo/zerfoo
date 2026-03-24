package disaggregated

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	disaggpb "github.com/zerfoo/zerfoo/serve/disaggregated/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// PrefillClient abstracts the prefill RPC. The production implementation wraps
// disaggpb.PrefillWorkerClient; tests can supply a mock.
type PrefillClient interface {
	Prefill(ctx context.Context, req *disaggpb.PreFillRequest) (KVBlockReceiver, error)
}

// KVBlockReceiver reads KVBlockStream messages from a prefill stream.
type KVBlockReceiver interface {
	Recv() (*disaggpb.KVBlockStream, error)
}

// DecodeClient abstracts the decode RPC.
type DecodeClient interface {
	Decode(ctx context.Context, req *disaggpb.DecodeRequest) (TokenReceiver, error)
}

// TokenReceiver reads TokenStream messages from a decode stream.
type TokenReceiver interface {
	Recv() (*disaggpb.TokenStream, error)
}

// workerEntry tracks a worker and its current load.
type workerEntry struct {
	addr   string
	active atomic.Int64
	// conn is non-nil only for gRPC-backed workers (nil for test mocks).
	conn *grpc.ClientConn
	// Client factories — exactly one pair is set.
	prefill PrefillClient
	decode  DecodeClient
}

// GatewayConfig holds configuration for the Gateway.
type GatewayConfig struct {
	PrefillAddrs []string
	DecodeAddrs  []string

	// HealthCheckInterval is the base interval for health checks.
	// Defaults to 1s.
	HealthCheckInterval time.Duration
	// MaxBackoff is the maximum backoff duration for health checks.
	// Defaults to 30s.
	MaxBackoff time.Duration

	// DialOptions are extra gRPC dial options for worker connections.
	DialOptions []grpc.DialOption
}

// Gateway routes incoming HTTP requests to disaggregated prefill and decode
// workers. It maintains connection pools for both worker types, routes each
// request to the least-loaded prefill worker, streams KV blocks to a decode
// worker, and multiplexes the decode token stream as an HTTP SSE response.
type Gateway struct {
	prefillWorkers []*workerEntry
	decodeWorkers  []*workerEntry

	mu      sync.RWMutex
	healthy map[string]bool // addr -> healthy

	healthInterval time.Duration
	maxBackoff     time.Duration

	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewGateway creates a new Gateway and dials all worker addresses. It starts
// background health check goroutines for each worker.
func NewGateway(cfg GatewayConfig) (*Gateway, error) {
	if len(cfg.PrefillAddrs) == 0 {
		return nil, fmt.Errorf("gateway: at least one prefill worker address required")
	}
	if len(cfg.DecodeAddrs) == 0 {
		return nil, fmt.Errorf("gateway: at least one decode worker address required")
	}

	dialOpts := cfg.DialOptions
	if len(dialOpts) == 0 {
		dialOpts = []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	}

	healthInterval := cfg.HealthCheckInterval
	if healthInterval == 0 {
		healthInterval = time.Second
	}
	maxBackoff := cfg.MaxBackoff
	if maxBackoff == 0 {
		maxBackoff = 30 * time.Second
	}

	g := &Gateway{
		healthy:        make(map[string]bool),
		healthInterval: healthInterval,
		maxBackoff:     maxBackoff,
	}

	// Dial prefill workers.
	for _, addr := range cfg.PrefillAddrs {
		conn, err := grpc.NewClient(addr, dialOpts...)
		if err != nil {
			g.Close()
			return nil, fmt.Errorf("gateway: dial prefill %s: %w", addr, err)
		}
		w := &workerEntry{
			addr:    addr,
			conn:    conn,
			prefill: &grpcPrefillClient{client: disaggpb.NewPrefillWorkerClient(conn)},
		}
		g.prefillWorkers = append(g.prefillWorkers, w)
		g.healthy[addr] = true
	}

	// Dial decode workers.
	for _, addr := range cfg.DecodeAddrs {
		conn, err := grpc.NewClient(addr, dialOpts...)
		if err != nil {
			g.Close()
			return nil, fmt.Errorf("gateway: dial decode %s: %w", addr, err)
		}
		w := &workerEntry{
			addr:   addr,
			conn:   conn,
			decode: &grpcDecodeClient{client: disaggpb.NewDecodeWorkerClient(conn)},
		}
		g.decodeWorkers = append(g.decodeWorkers, w)
		g.healthy[addr] = true
	}

	ctx, cancel := context.WithCancel(context.Background())
	g.cancel = cancel

	// Start health check loops for workers with gRPC connections.
	for _, w := range g.prefillWorkers {
		if w.conn != nil {
			g.wg.Add(1)
			go g.healthCheckLoop(ctx, w)
		}
	}
	for _, w := range g.decodeWorkers {
		if w.conn != nil {
			g.wg.Add(1)
			go g.healthCheckLoop(ctx, w)
		}
	}

	return g, nil
}

// Close shuts down health checks and closes all gRPC connections.
func (g *Gateway) Close() error {
	if g.cancel != nil {
		g.cancel()
	}
	g.wg.Wait()

	var firstErr error
	for _, w := range g.prefillWorkers {
		if w.conn != nil {
			if err := w.conn.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	for _, w := range g.decodeWorkers {
		if w.conn != nil {
			if err := w.conn.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	return firstErr
}

// leastLoaded picks the healthy worker with the lowest active request count.
// Returns nil if no healthy worker is available.
func (g *Gateway) leastLoaded(workers []*workerEntry) *workerEntry {
	g.mu.RLock()
	defer g.mu.RUnlock()

	var best *workerEntry
	bestLoad := int64(math.MaxInt64)
	for _, w := range workers {
		if !g.healthy[w.addr] {
			continue
		}
		load := w.active.Load()
		if load < bestLoad {
			bestLoad = load
			best = w
		}
	}
	return best
}

// sseEvent is an SSE data payload for streaming tokens.
type sseEvent struct {
	Token        int32  `json:"token_id"`
	Done         bool   `json:"done"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// ServeHTTP handles an incoming chat/completion request by:
//  1. Picking the least-loaded prefill worker
//  2. Streaming KV blocks from prefill
//  3. Sending KV blocks to a decode worker
//  4. Streaming decode tokens back as SSE
func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req disaggpb.PreFillRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	ctx := r.Context()

	// 1. Pick least-loaded prefill worker.
	pw := g.leastLoaded(g.prefillWorkers)
	if pw == nil {
		http.Error(w, "no healthy prefill workers", http.StatusServiceUnavailable)
		return
	}

	// 2. Run prefill and collect KV blocks.
	pw.active.Add(1)
	defer pw.active.Add(-1)

	kvStream, err := pw.prefill.Prefill(ctx, &req)
	if err != nil {
		http.Error(w, fmt.Sprintf("prefill error: %v", err), http.StatusBadGateway)
		return
	}

	var kvBlocks []*disaggpb.KVBlock
	for {
		msg, err := kvStream.Recv()
		if err != nil {
			if err == io.EOF {
				break
			}
			http.Error(w, fmt.Sprintf("prefill stream error: %v", err), http.StatusBadGateway)
			return
		}
		if msg.GetDone() {
			break
		}
		if msg.GetBlock() != nil {
			kvBlocks = append(kvBlocks, msg.GetBlock())
		}
	}

	// 3. Pick least-loaded decode worker and send KV blocks + token IDs.
	dw := g.leastLoaded(g.decodeWorkers)
	if dw == nil {
		http.Error(w, "no healthy decode workers", http.StatusServiceUnavailable)
		return
	}

	dw.active.Add(1)
	defer dw.active.Add(-1)

	decodeReq := &disaggpb.DecodeRequest{
		RequestId:    req.GetRequestId(),
		KvBlocks:     kvBlocks,
		TokenIds:     req.GetTokenIds(),
		MaxNewTokens: req.GetMaxNewTokens(),
		Temperature:  req.GetTemperature(),
	}

	tokenStream, err := dw.decode.Decode(ctx, decodeReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("decode error: %v", err), http.StatusBadGateway)
		return
	}

	// 4. Stream SSE response.
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	enc := json.NewEncoder(w)
	for {
		tok, err := tokenStream.Recv()
		if err != nil {
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", "inference failed")
			flusher.Flush()
			return
		}

		fmt.Fprintf(w, "data: ")
		enc.Encode(sseEvent{
			Token:        tok.GetTokenId(),
			Done:         tok.GetDone(),
			FinishReason: tok.GetFinishReason(),
		})
		fmt.Fprintf(w, "\n")
		flusher.Flush()

		if tok.GetDone() {
			return
		}
	}
}

// healthCheckLoop pings a worker's gRPC connection with exponential backoff.
// On failure, it marks the worker unhealthy; on success, it resets the backoff
// and marks the worker healthy.
func (g *Gateway) healthCheckLoop(ctx context.Context, w *workerEntry) {
	defer g.wg.Done()

	backoff := g.healthInterval
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}

		// Use a short-lived context for the health probe.
		probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		state := w.conn.GetState()
		_ = w.conn.WaitForStateChange(probeCtx, state)
		cancel()

		healthy := w.conn.GetState().String() == "READY" || w.conn.GetState().String() == "IDLE"

		g.mu.Lock()
		prev := g.healthy[w.addr]
		g.healthy[w.addr] = healthy
		g.mu.Unlock()

		if healthy {
			backoff = g.healthInterval
		} else {
			if prev {
				slog.Warn("gateway: worker became unhealthy", "addr", w.addr)
			}
			backoff = time.Duration(math.Min(
				float64(backoff*2),
				float64(g.maxBackoff),
			))
		}
	}
}

// grpcPrefillClient wraps disaggpb.PrefillWorkerClient to implement PrefillClient.
type grpcPrefillClient struct {
	client disaggpb.PrefillWorkerClient
}

func (c *grpcPrefillClient) Prefill(ctx context.Context, req *disaggpb.PreFillRequest) (KVBlockReceiver, error) {
	return c.client.Prefill(ctx, req)
}

// grpcDecodeClient wraps disaggpb.DecodeWorkerClient to implement DecodeClient.
type grpcDecodeClient struct {
	client disaggpb.DecodeWorkerClient
}

func (c *grpcDecodeClient) Decode(ctx context.Context, req *disaggpb.DecodeRequest) (TokenReceiver, error) {
	return c.client.Decode(ctx, req)
}
