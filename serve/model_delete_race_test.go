package serve

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// blockingLogitsNode behaves like fixedLogitsNode but blocks on its first
// Forward call until the test releases it. This lets a test pin a chat
// completion request inside the model's forward pass for a controlled
// window, so a concurrent DELETE /v1/models/{id} can be interleaved
// deterministically instead of relying on timing luck.
type blockingLogitsNode struct {
	graph.NoParameters[float32]
	vocabSize     int
	tokenSequence []int
	callCount     int
	started       chan struct{}
	release       chan struct{}
	once          sync.Once
}

func (n *blockingLogitsNode) OpType() string                     { return "BlockingLogits" }
func (n *blockingLogitsNode) Attributes() map[string]interface{} { return nil }
func (n *blockingLogitsNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }

func (n *blockingLogitsNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *blockingLogitsNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	n.once.Do(func() {
		close(n.started)
		<-n.release
	})

	seqLen := 1
	if len(inputs) > 0 {
		shape := inputs[0].Shape()
		if len(shape) >= 2 {
			seqLen = shape[1]
		}
	}
	data := make([]float32, seqLen*n.vocabSize)
	for pos := range seqLen {
		targetToken := n.tokenSequence[n.callCount%len(n.tokenSequence)]
		offset := pos * n.vocabSize
		for j := range n.vocabSize {
			data[offset+j] = -10.0
		}
		if targetToken >= 0 && targetToken < n.vocabSize {
			data[offset+targetToken] = 10.0
		}
		if pos == seqLen-1 {
			n.callCount++
		}
	}
	return tensor.New([]int{1, seqLen, n.vocabSize}, data)
}

// TestModelDelete_DrainsInFlightRequest_NoUseAfterClose is a regression test
// for CONC-H2: serve/handlers.go used to call s.inflight.Add(1) at handler
// entry with no recheck of s.unloaded, racing DELETE /v1/models/{id}'s
// s.unloaded.Store(true) -> s.inflight.Wait() -> s.model.Close() sequence.
// A request could Add(1) after Wait() had already returned (use-after-close
// on the now-closed model), or Add(1) could land exactly as the WaitGroup
// counter transitioned 1->0 during Wait (WaitGroup misuse panic).
//
// This test pins a chat completion request inside the model's forward pass
// (holding the handler's RLock), fires a concurrent delete, and asserts:
//   - the delete cannot proceed past acquiring the write lock until the
//     in-flight request finishes (drain semantics preserved),
//   - the in-flight request completes successfully against the still-open
//     model (no use-after-close),
//   - the delete succeeds afterward with no panic anywhere in the chain.
func TestModelDelete_DrainsInFlightRequest_NoUseAfterClose(t *testing.T) {
	node := &blockingLogitsNode{
		vocabSize:     8,
		tokenSequence: []int{6, 7, 2}, // foo, bar, EOS
		started:       make(chan struct{}),
		release:       make(chan struct{}),
	}
	mdl := buildModelWithNode(t, node)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	chatDone := make(chan struct{})
	var chatStatus int
	go func() {
		defer close(chatDone)
		resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json",
			`{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`)
		chatStatus = resp.StatusCode
		_ = resp.Body.Close()
	}()

	select {
	case <-node.started:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for chat completion to enter the forward pass")
	}

	deleteDone := make(chan struct{})
	var deleteStatus int
	go func() {
		defer close(deleteDone)
		resp := doDelete(t, ts.URL+"/v1/models/test-model")
		deleteStatus = resp.StatusCode
		_ = resp.Body.Close()
	}()

	// The delete must not be able to complete while the chat request still
	// holds the model RLock — this is the drain guarantee CONC-H2 broke.
	select {
	case <-deleteDone:
		t.Fatal("delete completed before the in-flight chat request finished draining")
	case <-time.After(200 * time.Millisecond):
	}

	close(node.release)

	select {
	case <-chatDone:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for chat completion to finish")
	}
	if chatStatus != http.StatusOK {
		t.Errorf("chat completion status = %d, want 200 (must observe the model before close, not after)", chatStatus)
	}

	select {
	case <-deleteDone:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for delete to finish")
	}
	if deleteStatus != http.StatusOK {
		t.Errorf("delete status = %d, want 200", deleteStatus)
	}
}

// TestConcurrentDeleteAndChatCompletions_NoRace hammers a server with many
// concurrent chat completion requests racing a single model delete. Under
// the old sync.WaitGroup-based implementation this could either panic with
// "sync: WaitGroup misuse: Add called concurrently with Wait" (caught by
// recoveryMiddleware and surfaced as a 500) or let a request proceed against
// a closed model. With the RWMutex-based fix, every response must be either
// 200 (request won the race and ran against the still-open model) or 404
// (request lost the race to the delete) — never a 500, and no data race
// under `go test -race`.
func TestConcurrentDeleteAndChatCompletions_NoRace(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	const n = 50
	statuses := make([]int, n)

	var wg sync.WaitGroup
	wg.Add(n + 1)
	for i := range n {
		go func(i int) {
			defer wg.Done()
			resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json",
				`{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`)
			statuses[i] = resp.StatusCode
			_ = resp.Body.Close()
		}(i)
	}
	go func() {
		defer wg.Done()
		resp := doDelete(t, ts.URL+"/v1/models/test-model")
		_ = resp.Body.Close()
	}()

	wg.Wait()

	for i, status := range statuses {
		if status != http.StatusOK && status != http.StatusNotFound {
			t.Errorf("request %d: status = %d, want 200 or 404 (500 would indicate a recovered panic from the raced delete)", i, status)
		}
	}
}
