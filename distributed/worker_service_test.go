package distributed

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/distributed/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
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

// --- Mock bidi stream for AllReduce tests ---

type mockAllReduceStream struct {
	grpc.ServerStream
	ctx      context.Context
	recvMsgs []*pb.AllReduceRequest
	recvIdx  int
	recvErr  error
	sentMsgs []*pb.AllReduceResponse
	sendErr  error
	mu       sync.Mutex
}

func (m *mockAllReduceStream) Context() context.Context { return m.ctx }

func (m *mockAllReduceStream) Send(resp *pb.AllReduceResponse) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.sendErr != nil {
		return m.sendErr
	}
	m.sentMsgs = append(m.sentMsgs, resp)
	return nil
}

func (m *mockAllReduceStream) Recv() (*pb.AllReduceRequest, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.recvErr != nil {
		return nil, m.recvErr
	}
	if m.recvIdx >= len(m.recvMsgs) {
		return nil, io.EOF
	}
	msg := m.recvMsgs[m.recvIdx]
	m.recvIdx++
	return msg, nil
}

func (m *mockAllReduceStream) SendHeader(metadata.MD) error { return nil }
func (m *mockAllReduceStream) SetHeader(metadata.MD) error  { return nil }
func (m *mockAllReduceStream) SetTrailer(metadata.MD)       {}
func (m *mockAllReduceStream) SendMsg(any) error { return nil }
func (m *mockAllReduceStream) RecvMsg(any) error { return nil }

// --- AllReduce RPC handler tests ---

func TestAllReduce_TwoPeers(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	ws.NewSession()

	// Root submits its own tensors.
	ws.SetLocalTensors(map[string]*pb.Tensor{
		"grad": {Shape: []int32{3}, Data: []float32{2, 4, 6}},
	})

	// Peer 1 calls AllReduce via stream.
	stream := &mockAllReduceStream{
		ctx: context.Background(),
		recvMsgs: []*pb.AllReduceRequest{
			{Name: "grad", Tensor: &pb.Tensor{Shape: []int32{3}, Data: []float32{4, 6, 8}}},
		},
	}

	err := ws.AllReduce(stream)
	if err != nil {
		t.Fatalf("AllReduce error: %v", err)
	}

	if len(stream.sentMsgs) != 1 {
		t.Fatalf("sent %d messages, want 1", len(stream.sentMsgs))
	}

	got := stream.sentMsgs[0]
	if got.Name != "grad" {
		t.Errorf("name = %s, want grad", got.Name)
	}
	// Average: [(2+4)/2, (4+6)/2, (6+8)/2] = [3, 5, 7]
	want := []float32{3, 5, 7}
	for i, v := range got.Tensor.Data {
		if v != want[i] {
			t.Errorf("data[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestAllReduce_NoSession(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	// No NewSession() called.

	stream := &mockAllReduceStream{
		ctx:      context.Background(),
		recvMsgs: []*pb.AllReduceRequest{},
	}

	err := ws.AllReduce(stream)
	if err == nil {
		t.Fatal("expected error when no session exists")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.FailedPrecondition {
		t.Errorf("expected FailedPrecondition, got %v", err)
	}
}

func TestAllReduce_RecvError(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	ws.NewSession()

	stream := &mockAllReduceStream{
		ctx:     context.Background(),
		recvErr: errors.New("connection reset"),
	}

	err := ws.AllReduce(stream)
	if err == nil {
		t.Fatal("expected error on recv failure")
	}
}

func TestAllReduce_SendError(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	ws.NewSession()

	// Root submits its own tensors.
	ws.SetLocalTensors(map[string]*pb.Tensor{
		"grad": {Shape: []int32{1}, Data: []float32{1}},
	})

	stream := &mockAllReduceStream{
		ctx: context.Background(),
		recvMsgs: []*pb.AllReduceRequest{
			{Name: "grad", Tensor: &pb.Tensor{Shape: []int32{1}, Data: []float32{2}}},
		},
		sendErr: errors.New("broken pipe"),
	}

	err := ws.AllReduce(stream)
	if err == nil {
		t.Fatal("expected error on send failure")
	}
}

// --- Barrier RPC handler tests ---

func TestBarrier_AllWorkersArrive_RPC(t *testing.T) {
	ws := NewWorkerService(0, 3, nil)

	errs := make([]error, 3)
	var wg sync.WaitGroup
	for i := range 3 {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			_, errs[rank] = ws.Barrier(context.Background(), &pb.BarrierRequest{Rank: int32(rank)})
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("worker %d: unexpected error: %v", i, err)
		}
	}
}

func TestBarrier_InvalidRank(t *testing.T) {
	ws := NewWorkerService(0, 3, nil)

	tests := []struct {
		name string
		rank int32
	}{
		{"negative", -1},
		{"too_large", 3},
		{"way_too_large", 100},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ws.Barrier(context.Background(), &pb.BarrierRequest{Rank: tc.rank})
			if err == nil {
				t.Fatal("expected error for invalid rank")
			}
			st, ok := status.FromError(err)
			if !ok || st.Code() != codes.InvalidArgument {
				t.Errorf("expected InvalidArgument, got %v", err)
			}
		})
	}
}

func TestBarrier_Timeout_RPC(t *testing.T) {
	ws := NewWorkerService(0, 3, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	// Only 1 of 3 workers arrives.
	_, err := ws.Barrier(ctx, &pb.BarrierRequest{Rank: 0})
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

// --- Broadcast RPC handler tests ---

func TestBroadcast_SetThenRetrieve(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)

	tensor := &pb.Tensor{Shape: []int32{2}, Data: []float32{10, 20}}
	ws.SetBroadcastTensor("weights", tensor)

	resp, err := ws.Broadcast(context.Background(), &pb.BroadcastRequest{Name: "weights"})
	if err != nil {
		t.Fatalf("Broadcast error: %v", err)
	}
	if resp.Tensor == nil {
		t.Fatal("expected tensor in response")
	}
	if resp.Tensor.Data[0] != 10 || resp.Tensor.Data[1] != 20 {
		t.Errorf("data = %v, want [10, 20]", resp.Tensor.Data)
	}
}

func TestBroadcast_WaitForTensor(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)

	// Set tensor after a short delay.
	go func() {
		time.Sleep(10 * time.Millisecond)
		ws.SetBroadcastTensor("weights", &pb.Tensor{Shape: []int32{1}, Data: []float32{42}})
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	resp, err := ws.Broadcast(ctx, &pb.BroadcastRequest{Name: "weights"})
	if err != nil {
		t.Fatalf("Broadcast error: %v", err)
	}
	if resp.Tensor.Data[0] != 42 {
		t.Errorf("data[0] = %f, want 42", resp.Tensor.Data[0])
	}
}

func TestBroadcast_EmptyName(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)

	_, err := ws.Broadcast(context.Background(), &pb.BroadcastRequest{Name: ""})
	if err == nil {
		t.Fatal("expected error for empty name")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", err)
	}
}

func TestBroadcast_Timeout(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	// Tensor never set, should timeout.
	_, err := ws.Broadcast(ctx, &pb.BroadcastRequest{Name: "missing"})
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

// --- AllReduce input validation tests ---

func TestAllReduce_EmptyTensorName(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	ws.NewSession()

	stream := &mockAllReduceStream{
		ctx: context.Background(),
		recvMsgs: []*pb.AllReduceRequest{
			{Name: "", Tensor: &pb.Tensor{Shape: []int32{1}, Data: []float32{1}}},
		},
	}

	err := ws.AllReduce(stream)
	if err == nil {
		t.Fatal("expected error for empty tensor name")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", err)
	}
}

func TestAllReduce_NilTensor(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	ws.NewSession()

	stream := &mockAllReduceStream{
		ctx: context.Background(),
		recvMsgs: []*pb.AllReduceRequest{
			{Name: "grad", Tensor: nil},
		},
	}

	err := ws.AllReduce(stream)
	if err == nil {
		t.Fatal("expected error for nil tensor")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", err)
	}
}

func TestAllReduce_InvalidTensorShape(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)
	ws.NewSession()

	stream := &mockAllReduceStream{
		ctx: context.Background(),
		recvMsgs: []*pb.AllReduceRequest{
			{Name: "grad", Tensor: &pb.Tensor{Shape: []int32{2, 3}, Data: []float32{1}}},
		},
	}

	err := ws.AllReduce(stream)
	if err == nil {
		t.Fatal("expected error for shape/data mismatch")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", err)
	}
}

// --- Broadcast input validation tests ---

func TestBroadcast_InvalidTensor(t *testing.T) {
	ws := NewWorkerService(0, 2, nil)

	_, err := ws.Broadcast(context.Background(), &pb.BroadcastRequest{
		Name:   "w",
		Tensor: &pb.Tensor{Shape: []int32{2}, Data: []float32{1, 2, 3}}, // mismatch
	})
	if err == nil {
		t.Fatal("expected error for invalid tensor")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", err)
	}
}

// --- validateTensor tests ---

func TestValidateTensor(t *testing.T) {
	tests := []struct {
		name    string
		tensor  *pb.Tensor
		wantErr bool
	}{
		{"valid_1d", &pb.Tensor{Shape: []int32{3}, Data: []float32{1, 2, 3}}, false},
		{"valid_2d", &pb.Tensor{Shape: []int32{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}, false},
		{"nil_tensor", nil, true},
		{"empty_shape", &pb.Tensor{Shape: []int32{}, Data: []float32{1}}, true},
		{"zero_dim", &pb.Tensor{Shape: []int32{0, 3}, Data: []float32{}}, true},
		{"negative_dim", &pb.Tensor{Shape: []int32{-1, 3}, Data: []float32{}}, true},
		{"shape_data_mismatch", &pb.Tensor{Shape: []int32{2, 3}, Data: []float32{1, 2}}, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateTensor(tc.tensor, "test")
			if (err != nil) != tc.wantErr {
				t.Errorf("validateTensor() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

