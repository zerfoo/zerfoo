package distributed

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/distributed/pb"
)

// --- reduceSession tests ---

func TestReduceSession_TwoPeersGetAveragedResult(t *testing.T) {
	rs := newReduceSession(2)

	// Peer 0 submits [2, 4, 6]
	rs.Submit(0, map[string]*pb.Tensor{
		"grad": {Shape: []int32{3}, Data: []float32{2, 4, 6}},
	})

	// Peer 1 submits [4, 6, 8]
	rs.Submit(1, map[string]*pb.Tensor{
		"grad": {Shape: []int32{3}, Data: []float32{4, 6, 8}},
	})

	result := rs.WaitForResult(context.Background())
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	got := result["grad"]
	if got == nil {
		t.Fatal("expected grad tensor in result")
	}

	// Average: [(2+4)/2, (4+6)/2, (6+8)/2] = [3, 5, 7]
	want := []float32{3, 5, 7}
	if len(got.Data) != len(want) {
		t.Fatalf("data length = %d, want %d", len(got.Data), len(want))
	}
	for i, v := range got.Data {
		if v != want[i] {
			t.Errorf("data[%d] = %f, want %f", i, v, want[i])
		}
	}
	if len(got.Shape) != 1 || got.Shape[0] != 3 {
		t.Errorf("shape = %v, want [3]", got.Shape)
	}
}

func TestReduceSession_ThreePeersMultipleTensors(t *testing.T) {
	rs := newReduceSession(3)

	rs.Submit(0, map[string]*pb.Tensor{
		"w1": {Shape: []int32{2}, Data: []float32{3, 6}},
		"b1": {Shape: []int32{1}, Data: []float32{9}},
	})
	rs.Submit(1, map[string]*pb.Tensor{
		"w1": {Shape: []int32{2}, Data: []float32{6, 9}},
		"b1": {Shape: []int32{1}, Data: []float32{12}},
	})
	rs.Submit(2, map[string]*pb.Tensor{
		"w1": {Shape: []int32{2}, Data: []float32{9, 12}},
		"b1": {Shape: []int32{1}, Data: []float32{15}},
	})

	result := rs.WaitForResult(context.Background())
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	// w1: [(3+6+9)/3, (6+9+12)/3] = [6, 9]
	w1 := result["w1"]
	if w1 == nil {
		t.Fatal("expected w1 in result")
	}
	wantW1 := []float32{6, 9}
	for i, v := range w1.Data {
		if v != wantW1[i] {
			t.Errorf("w1[%d] = %f, want %f", i, v, wantW1[i])
		}
	}

	// b1: [(9+12+15)/3] = [12]
	b1 := result["b1"]
	if b1 == nil {
		t.Fatal("expected b1 in result")
	}
	if b1.Data[0] != 12 {
		t.Errorf("b1[0] = %f, want 12", b1.Data[0])
	}
}

func TestReduceSession_TimeoutWhenPeerMissing(t *testing.T) {
	rs := newReduceSession(2)

	// Only one peer submits.
	rs.Submit(0, map[string]*pb.Tensor{
		"grad": {Shape: []int32{2}, Data: []float32{1, 2}},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	result := rs.WaitForResult(ctx)
	if result != nil {
		t.Errorf("expected nil result on timeout, got %v", result)
	}
}

func TestReduceSession_ConcurrentSubmission(t *testing.T) {
	const peers = 10
	rs := newReduceSession(int32(peers))

	var wg sync.WaitGroup
	for i := range peers {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			rs.Submit(int32(rank), map[string]*pb.Tensor{
				"grad": {Shape: []int32{2}, Data: []float32{float32(rank), float32(rank * 2)}},
			})
		}(i)
	}
	wg.Wait()

	result := rs.WaitForResult(context.Background())
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	grad := result["grad"]
	if grad == nil {
		t.Fatal("expected grad in result")
	}
	if len(grad.Data) != 2 {
		t.Fatalf("data length = %d, want 2", len(grad.Data))
	}

	// Sum of ranks 0..9 = 45, average = 4.5
	// Sum of rank*2 for 0..9 = 90, average = 9.0
	const eps = 1e-5
	if diff := grad.Data[0] - 4.5; diff > eps || diff < -eps {
		t.Errorf("data[0] = %f, want 4.5", grad.Data[0])
	}
	if diff := grad.Data[1] - 9.0; diff > eps || diff < -eps {
		t.Errorf("data[1] = %f, want 9.0", grad.Data[1])
	}
}

func TestReduceSession_NilTensorSkipped(t *testing.T) {
	rs := newReduceSession(2)

	rs.Submit(0, map[string]*pb.Tensor{
		"grad": {Shape: []int32{1}, Data: []float32{10}},
		"skip": nil,
	})
	rs.Submit(1, map[string]*pb.Tensor{
		"grad": {Shape: []int32{1}, Data: []float32{20}},
	})

	result := rs.WaitForResult(context.Background())
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if _, ok := result["skip"]; ok {
		t.Error("expected skip tensor to be absent from result")
	}
	grad := result["grad"]
	if grad == nil {
		t.Fatal("expected grad in result")
	}
	// Only one peer submitted for "grad" in the averaging, so
	// sum = 10+20 = 30, avg = 30/2 = 15. But "skip" only had nil from peer 0
	// and was absent from peer 1, so it should not appear.
	if grad.Data[0] != 15 {
		t.Errorf("grad[0] = %f, want 15", grad.Data[0])
	}
}

// --- barrierState tests ---

func TestBarrierState_AllWorkersArrive(t *testing.T) {
	bs := newBarrierState(3)

	errs := make([]error, 3)
	var wg sync.WaitGroup
	for i := range 3 {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			errs[rank] = bs.arrive(context.Background())
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("worker %d: unexpected error: %v", i, err)
		}
	}
}

func TestBarrierState_TimeoutWhenWorkerMissing(t *testing.T) {
	bs := newBarrierState(3)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	// Only 2 of 3 workers arrive.
	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := range 2 {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			errs[rank] = bs.arrive(ctx)
		}(i)
	}
	wg.Wait()

	// At least one worker should get a context error.
	anyErr := false
	for _, err := range errs {
		if err != nil {
			anyErr = true
		}
	}
	if !anyErr {
		t.Error("expected at least one worker to get context error")
	}
}

func TestBarrierState_SequentialBarriers(t *testing.T) {
	bs := newBarrierState(2)

	// Run 3 sequential barriers.
	for round := range 3 {
		var wg sync.WaitGroup
		errs := make([]error, 2)
		for i := range 2 {
			wg.Add(1)
			go func(rank int) {
				defer wg.Done()
				errs[rank] = bs.arrive(context.Background())
			}(i)
		}
		wg.Wait()

		for i, err := range errs {
			if err != nil {
				t.Errorf("round %d, worker %d: unexpected error: %v", round, i, err)
			}
		}
	}

	// Epoch should be 3 after 3 barriers.
	bs.mu.Lock()
	if bs.epoch != 3 {
		t.Errorf("epoch = %d, want 3", bs.epoch)
	}
	bs.mu.Unlock()
}

// --- NewWorkerService tests ---

func TestNewWorkerService(t *testing.T) {
	ws := NewWorkerService(0, 3, nil)
	if ws.rank != 0 {
		t.Errorf("rank = %d, want 0", ws.rank)
	}
	if ws.worldSize != 3 {
		t.Errorf("worldSize = %d, want 3", ws.worldSize)
	}
	if ws.logger == nil {
		t.Error("expected non-nil logger (should default to Nop)")
	}
	if ws.collector == nil {
		t.Error("expected non-nil collector (should default to Nop)")
	}
	if ws.barrier == nil {
		t.Error("expected non-nil barrier")
	}
}

func TestWorkerService_NewSession(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	if ws.getSession() != nil {
		t.Error("expected nil session before NewSession")
	}
	ws.NewSession()
	if ws.getSession() == nil {
		t.Error("expected non-nil session after NewSession")
	}
}

func TestWorkerService_SetLocalTensors(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	ws.NewSession()

	ws.SetLocalTensors(map[string]*pb.Tensor{
		"grad": {Shape: []int32{2}, Data: []float32{1, 2}},
	})

	// Submit from the other peer to complete the session.
	ws.getSession().Submit(1, map[string]*pb.Tensor{
		"grad": {Shape: []int32{2}, Data: []float32{3, 4}},
	})

	result := ws.getSession().WaitForResult(context.Background())
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	grad := result["grad"]
	if grad == nil {
		t.Fatal("expected grad in result")
	}
	// Average: [(1+3)/2, (2+4)/2] = [2, 3]
	if grad.Data[0] != 2 || grad.Data[1] != 3 {
		t.Errorf("grad = %v, want [2, 3]", grad.Data)
	}
}

func TestWorkerService_SetLocalTensors_NoSession(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	// Should not panic when session is nil.
	ws.SetLocalTensors(map[string]*pb.Tensor{
		"grad": {Shape: []int32{1}, Data: []float32{1}},
	})
}

func TestWorkerService_SetBroadcastTensor(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)

	tensor := &pb.Tensor{Shape: []int32{2}, Data: []float32{10, 20}}
	ws.SetBroadcastTensor("weights", tensor)

	entry := ws.getBroadcastEntry("weights")
	select {
	case <-entry.ready:
		// good, ready
	default:
		t.Error("expected broadcast entry to be ready")
	}
	if entry.tensor != tensor {
		t.Error("expected stored tensor to match")
	}
}

func TestWorkerService_GetBroadcastEntry_WaitsForSet(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)

	entry := ws.getBroadcastEntry("weights")
	select {
	case <-entry.ready:
		t.Error("expected broadcast entry to NOT be ready yet")
	default:
		// good, not ready
	}

	// Now set it.
	tensor := &pb.Tensor{Shape: []int32{1}, Data: []float32{42}}
	ws.SetBroadcastTensor("weights", tensor)

	select {
	case <-entry.ready:
		// good
	default:
		t.Error("expected broadcast entry to be ready after set")
	}
}

func TestWorkerService_ClearBroadcasts(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	ws.SetBroadcastTensor("w", &pb.Tensor{Shape: []int32{1}, Data: []float32{1}})
	ws.ClearBroadcasts()

	// After clear, getting an entry should return a new, non-ready entry.
	entry := ws.getBroadcastEntry("w")
	select {
	case <-entry.ready:
		t.Error("expected new entry to NOT be ready after clear")
	default:
		// good
	}
}

func TestWorkerService_SetCollector(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	// nil should default to Nop
	ws.SetCollector(nil)
	if ws.collector == nil {
		t.Error("expected non-nil collector after SetCollector(nil)")
	}
}

func TestWorkerService_SetBroadcastTensor_DoubleSet(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	t1 := &pb.Tensor{Shape: []int32{1}, Data: []float32{1}}
	t2 := &pb.Tensor{Shape: []int32{1}, Data: []float32{2}}

	ws.SetBroadcastTensor("w", t1)
	ws.SetBroadcastTensor("w", t2) // should not panic

	entry := ws.getBroadcastEntry("w")
	if entry.tensor != t2 {
		t.Error("expected second tensor to overwrite first")
	}
}

