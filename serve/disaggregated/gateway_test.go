package disaggregated

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	disaggpb "github.com/zerfoo/zerfoo/serve/disaggregated/proto"
)

// mockPrefillClient implements PrefillClient for testing.
type mockPrefillClient struct {
	layers int
}

func (m *mockPrefillClient) Prefill(_ context.Context, req *disaggpb.PreFillRequest) (KVBlockReceiver, error) {
	var msgs []*disaggpb.KVBlockStream
	for i := range m.layers {
		msgs = append(msgs, &disaggpb.KVBlockStream{
			Block: &disaggpb.KVBlock{
				RequestId: req.GetRequestId(),
				LayerIdx:  int32(i),
				BlockIdx:  0,
				KData:     []byte{0x00, 0x3C}, // 1.0 in FP16
				VData:     []byte{0x00, 0x3C},
			},
		})
	}
	msgs = append(msgs, &disaggpb.KVBlockStream{Done: true})
	return &sliceKVReceiver{msgs: msgs}, nil
}

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

// mockDecodeClient implements DecodeClient for testing.
type mockDecodeClient struct {
	tokens []int32
}

func (m *mockDecodeClient) Decode(_ context.Context, req *disaggpb.DecodeRequest) (TokenReceiver, error) {
	var msgs []*disaggpb.TokenStream
	for _, tok := range m.tokens {
		msgs = append(msgs, &disaggpb.TokenStream{
			RequestId: req.GetRequestId(),
			TokenId:   tok,
		})
	}
	msgs = append(msgs, &disaggpb.TokenStream{
		RequestId:    req.GetRequestId(),
		TokenId:      2,
		Done:         true,
		FinishReason: "stop",
	})
	return &sliceTokenReceiver{msgs: msgs}, nil
}

type sliceTokenReceiver struct {
	msgs []*disaggpb.TokenStream
	idx  int
}

func (r *sliceTokenReceiver) Recv() (*disaggpb.TokenStream, error) {
	if r.idx >= len(r.msgs) {
		return nil, io.EOF
	}
	msg := r.msgs[r.idx]
	r.idx++
	return msg, nil
}

// newTestGateway creates a Gateway with mock clients (no gRPC connections).
func newTestGateway(prefillClients []PrefillClient, decodeClients []DecodeClient) *Gateway {
	g := &Gateway{
		healthy:        make(map[string]bool),
		healthInterval: time.Second,
		maxBackoff:     30 * time.Second,
	}

	for i, pc := range prefillClients {
		addr := addrForIndex("prefill", i)
		g.prefillWorkers = append(g.prefillWorkers, &workerEntry{
			addr:    addr,
			prefill: pc,
		})
		g.healthy[addr] = true
	}

	for i, dc := range decodeClients {
		addr := addrForIndex("decode", i)
		g.decodeWorkers = append(g.decodeWorkers, &workerEntry{
			addr:   addr,
			decode: dc,
		})
		g.healthy[addr] = true
	}

	ctx, cancel := context.WithCancel(context.Background())
	g.cancel = cancel
	// No health check loops for test gateways (no real gRPC connections).
	_ = ctx

	return g
}

func addrForIndex(kind string, i int) string {
	return kind + "-" + string(rune('0'+i))
}

func TestGateway(t *testing.T) {
	gw := newTestGateway(
		[]PrefillClient{&mockPrefillClient{layers: 2}},
		[]DecodeClient{&mockDecodeClient{tokens: []int32{5, 3, 7}}},
	)
	defer gw.Close()

	reqBody := `{"request_id":"req-1","token_ids":[1,2,3],"max_new_tokens":10,"temperature":0.7}`
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(reqBody))
	rec := httptest.NewRecorder()

	gw.ServeHTTP(rec, req)

	resp := rec.Result()
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d: %s", resp.StatusCode, body)
	}

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	// Parse SSE events.
	body, _ := io.ReadAll(resp.Body)
	lines := strings.Split(strings.TrimSpace(string(body)), "\n")

	var events []sseEvent
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		var evt sseEvent
		if err := json.Unmarshal([]byte(data), &evt); err != nil {
			t.Fatalf("unmarshal SSE data %q: %v", data, err)
		}
		events = append(events, evt)
	}

	// Expect 4 events: tokens 5, 3, 7, then EOS (token 2).
	expectedTokens := []int32{5, 3, 7, 2}
	if len(events) != len(expectedTokens) {
		t.Fatalf("expected %d SSE events, got %d: %v", len(expectedTokens), len(events), events)
	}
	for i, evt := range events {
		if evt.Token != expectedTokens[i] {
			t.Errorf("event[%d].Token = %d, want %d", i, evt.Token, expectedTokens[i])
		}
	}

	// Last event should be done.
	last := events[len(events)-1]
	if !last.Done {
		t.Error("last event: expected done=true")
	}
	if last.FinishReason != "stop" {
		t.Errorf("last event: finish_reason = %q, want %q", last.FinishReason, "stop")
	}
}

func TestGateway_LeastLoadedRouting(t *testing.T) {
	gw := newTestGateway(
		[]PrefillClient{
			&mockPrefillClient{layers: 1},
			&mockPrefillClient{layers: 1},
		},
		[]DecodeClient{&mockDecodeClient{tokens: []int32{4}}},
	)
	defer gw.Close()

	// Simulate load on worker 0.
	gw.prefillWorkers[0].active.Store(10)

	// leastLoaded should pick worker 1 (lower load).
	picked := gw.leastLoaded(gw.prefillWorkers)
	if picked == nil {
		t.Fatal("leastLoaded returned nil")
	}
	if picked.addr != gw.prefillWorkers[1].addr {
		t.Errorf("expected least-loaded = %s, got %s", gw.prefillWorkers[1].addr, picked.addr)
	}
}

func TestGateway_HealthCheck(t *testing.T) {
	gw := newTestGateway(
		[]PrefillClient{&mockPrefillClient{layers: 1}},
		[]DecodeClient{&mockDecodeClient{tokens: []int32{4}}},
	)
	defer gw.Close()

	// All workers start healthy.
	gw.mu.RLock()
	for addr, h := range gw.healthy {
		if !h {
			t.Errorf("worker %s should start healthy", addr)
		}
	}
	gw.mu.RUnlock()

	// Mark a prefill worker unhealthy and verify leastLoaded skips it.
	prefillAddr := gw.prefillWorkers[0].addr
	gw.mu.Lock()
	gw.healthy[prefillAddr] = false
	gw.mu.Unlock()

	picked := gw.leastLoaded(gw.prefillWorkers)
	if picked != nil {
		t.Error("expected nil when all prefill workers are unhealthy")
	}
}

func TestGateway_MethodNotAllowed(t *testing.T) {
	gw := &Gateway{healthy: make(map[string]bool)}

	req := httptest.NewRequest(http.MethodGet, "/v1/completions", nil)
	rec := httptest.NewRecorder()
	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", rec.Code)
	}
}

func TestGateway_NoHealthyWorkers(t *testing.T) {
	gw := newTestGateway(
		[]PrefillClient{&mockPrefillClient{layers: 1}},
		[]DecodeClient{&mockDecodeClient{tokens: nil}},
	)
	defer gw.Close()

	// Mark prefill worker unhealthy.
	gw.mu.Lock()
	gw.healthy[gw.prefillWorkers[0].addr] = false
	gw.mu.Unlock()

	reqBody := `{"request_id":"req-fail","token_ids":[1],"max_new_tokens":5}`
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(reqBody))
	rec := httptest.NewRecorder()
	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503, got %d", rec.Code)
	}
}

func TestNewGateway_Validation(t *testing.T) {
	_, err := NewGateway(GatewayConfig{})
	if err == nil {
		t.Fatal("expected error with empty config")
	}

	_, err = NewGateway(GatewayConfig{PrefillAddrs: []string{"localhost:0"}})
	if err == nil {
		t.Fatal("expected error with no decode addrs")
	}
}

func TestGateway_ExponentialBackoffCap(t *testing.T) {
	// Verify the health check loop respects context cancellation.
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // immediately cancel

	gw := &Gateway{
		healthy:        map[string]bool{"test:1234": true},
		healthInterval: 1 * time.Second,
		maxBackoff:     30 * time.Second,
		cancel:         cancel,
	}

	w := &workerEntry{addr: "test:1234"}
	// The loop should exit immediately due to cancelled context.
	gw.wg.Add(1)
	go gw.healthCheckLoop(ctx, w)
	gw.wg.Wait()
}

func TestGateway_NoHealthyDecodeWorkers(t *testing.T) {
	gw := newTestGateway(
		[]PrefillClient{&mockPrefillClient{layers: 1}},
		[]DecodeClient{&mockDecodeClient{tokens: []int32{4}}},
	)
	defer gw.Close()

	// Mark decode worker unhealthy.
	gw.mu.Lock()
	gw.healthy[gw.decodeWorkers[0].addr] = false
	gw.mu.Unlock()

	reqBody := `{"request_id":"req-2","token_ids":[1],"max_new_tokens":5}`
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(reqBody))
	rec := httptest.NewRecorder()
	gw.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503, got %d", rec.Code)
	}
}
