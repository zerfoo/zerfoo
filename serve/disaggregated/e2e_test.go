package disaggregated

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestDisaggregatedE2E exercises the full request cycle through a real HTTP
// server: prompt → gateway → mock prefill worker → mock decode worker → SSE
// response with coherent token stream.
func TestDisaggregatedE2E(t *testing.T) {
	// Set up a gateway with mock workers:
	//   - Prefill returns 2 KV blocks (2 layers)
	//   - Decode streams tokens [10, 20, 30] then EOS (token 2, done=true, finish_reason="stop")
	gw := newTestGateway(
		[]PrefillClient{&mockPrefillClient{layers: 2}},
		[]DecodeClient{&mockDecodeClient{tokens: []int32{10, 20, 30}}},
	)
	defer gw.Close()

	// Start a real HTTP server backed by the gateway.
	srv := httptest.NewServer(gw)
	defer srv.Close()

	// Send a POST request to the gateway's completions endpoint.
	reqBody := `{"request_id":"e2e-1","token_ids":[1,2,3],"max_new_tokens":10,"temperature":0.7}`
	resp, err := http.Post(srv.URL+"/v1/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	// Parse the SSE stream line by line.
	var events []sseEvent
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
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
	if err := scanner.Err(); err != nil {
		t.Fatalf("reading SSE stream: %v", err)
	}

	// Expect tokens [10, 20, 30] followed by EOS (token 2).
	wantTokens := []int32{10, 20, 30, 2}
	if len(events) != len(wantTokens) {
		t.Fatalf("expected %d SSE events, got %d: %+v", len(wantTokens), len(events), events)
	}

	for i, evt := range events {
		if evt.Token != wantTokens[i] {
			t.Errorf("event[%d].Token = %d, want %d", i, evt.Token, wantTokens[i])
		}
	}

	// Verify ordering: tokens must arrive in sequence.
	for i := 0; i < len(events)-1; i++ {
		if events[i].Done {
			t.Errorf("event[%d] should not be done (only last event should be done)", i)
		}
	}

	// Verify the final event signals completion.
	last := events[len(events)-1]
	if !last.Done {
		t.Error("last event: expected done=true")
	}
	if last.FinishReason != "stop" {
		t.Errorf("last event: finish_reason = %q, want %q", last.FinishReason, "stop")
	}

	// TTFT < 500ms assertion: requires a real model and DGX hardware.
	// Skipping in mock-based tests; enable when running on DGX Spark.
}
