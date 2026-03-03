package testutils

import (
	"errors"
	"net"
	"testing"
)

func TestCustomMockStrategy_InitAndRank(t *testing.T) {
	mock := &CustomMockStrategy[float32]{}
	mock.ReturnInit(nil)
	mock.ReturnRank(3)

	if err := mock.Init(0, 4, "localhost:5000"); err != nil {
		t.Errorf("Init: unexpected error: %v", err)
	}

	if r := mock.Rank(); r != 3 {
		t.Errorf("Rank: got %d, want 3", r)
	}

	mock.AssertExpectations(t)
}

func TestCustomMockStrategy_SizeAndBarrier(t *testing.T) {
	mock := &CustomMockStrategy[float32]{}
	mock.ReturnSize(4)
	mock.ReturnBarrier(nil)
	mock.ReturnBarrier(nil)

	if s := mock.Size(); s != 4 {
		t.Errorf("Size: got %d, want 4", s)
	}

	if err := mock.Barrier(); err != nil {
		t.Errorf("Barrier 1: unexpected error: %v", err)
	}

	if err := mock.Barrier(); err != nil {
		t.Errorf("Barrier 2: unexpected error: %v", err)
	}
}

func TestCustomMockStrategy_InitError(t *testing.T) {
	mock := &CustomMockStrategy[float32]{}
	testErr := errors.New("init failed")
	mock.ReturnInit(testErr)

	if err := mock.Init(0, 2, "localhost"); !errors.Is(err, testErr) {
		t.Errorf("expected %v, got %v", testErr, err)
	}
}

func TestCustomMockStrategy_AllReduceGradients(t *testing.T) {
	mock := &CustomMockStrategy[float32]{}
	mock.ReturnAllReduceGradients(nil)

	err := mock.AllReduceGradients(nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCustomMockStrategy_BroadcastTensor(t *testing.T) {
	mock := &CustomMockStrategy[float32]{}
	mock.ReturnBroadcastTensor(nil)

	err := mock.BroadcastTensor(nil, 0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(mock.BroadcastTensorArgs) == 0 {
		t.Error("expected BroadcastTensorArgs to be populated")
	}
}

func TestCustomMockStrategy_Shutdown(t *testing.T) {
	mock := &CustomMockStrategy[float32]{}
	mock.Shutdown()
	// Verify no panic.
}

func TestCustomMockStrategy_AssertNotCalled(t *testing.T) {
	mock := &CustomMockStrategy[float32]{}
	mock.AssertNotCalled(t, "Init")
	mock.AssertNotCalled(t, "AllReduceGradients")
	mock.AssertNotCalled(t, "Barrier")
	mock.AssertNotCalled(t, "BroadcastTensor")
	mock.AssertNotCalled(t, "Rank")
	mock.AssertNotCalled(t, "Size")
	mock.AssertNotCalled(t, "Shutdown")
}

func TestCustomMockStrategy_AssertNotCalled_Unknown(t *testing.T) {
	mock := &CustomMockStrategy[float32]{}
	ft := &testing.T{}
	mock.AssertNotCalled(ft, "UnknownMethod")
}

func TestCustomMockStrategy_FluentAPI(t *testing.T) {
	mock := &CustomMockStrategy[float32]{}
	// Verify fluent chaining returns the same mock.
	result := mock.OnInit(0, 2, "addr").ReturnInit(nil).OnceInit()
	if result != mock {
		t.Error("fluent API should return same mock")
	}

	result = mock.OnRank().ReturnRank(1).OnceRank()
	if result != mock {
		t.Error("fluent API should return same mock")
	}

	result = mock.OnSize().ReturnSize(2).OnceSize()
	if result != mock {
		t.Error("fluent API should return same mock")
	}

	result = mock.OnBarrier().ReturnBarrier(nil).TwiceBarrier()
	if result != mock {
		t.Error("fluent API should return same mock")
	}

	result = mock.OnAllReduceGradients(nil).ReturnAllReduceGradients(nil).OnceAllReduceGradients()
	if result != mock {
		t.Error("fluent API should return same mock")
	}

	result = mock.OnBroadcastTensor(nil, 0).ReturnBroadcastTensor(nil).OnceBroadcastTensor()
	if result != mock {
		t.Error("fluent API should return same mock")
	}
}

type mockAddr struct{}

func (mockAddr) Network() string { return "tcp" }
func (mockAddr) String() string  { return "127.0.0.1:0" }

func TestCustomMockListener(t *testing.T) {
	ml := &CustomMockListener{}
	testErr := errors.New("accept fail")
	ml.OnAccept(testErr)
	ml.OnClose(nil)
	ml.OnAddr(mockAddr{})

	_, err := ml.Accept()
	if !errors.Is(err, testErr) {
		t.Errorf("Accept: expected %v, got %v", testErr, err)
	}

	if err := ml.Close(); err != nil {
		t.Errorf("Close: unexpected error: %v", err)
	}

	addr := ml.Addr()
	if addr == nil {
		t.Error("expected non-nil address")
	}

	ml.AssertExpectations(t)
}

var _ net.Listener = (*CustomMockListener)(nil)

func TestCustomMockGrpcServer(t *testing.T) {
	ms := &CustomMockGrpcServer{}
	ms.OnServe(nil)
	ms.RegisterService(nil, nil)
	ms.Stop()
	ms.GracefulStop()

	if err := ms.Serve(nil); err != nil {
		t.Errorf("Serve: unexpected error: %v", err)
	}

	ms.AssertExpectations(t)
}

func TestCustomMockLogger(t *testing.T) {
	ml := &CustomMockLogger{}
	ml.OnPrintf()
	ml.Printf("test %s %d", "hello", 42)

	if ml.printfCalls != 1 {
		t.Errorf("expected 1 Printf call, got %d", ml.printfCalls)
	}

	ml.AssertExpectations(t)
}

func TestCustomMockDistributedServiceClient(t *testing.T) {
	mc := &CustomMockDistributedServiceClient{}
	mc.OnAllReduce().ReturnAllReduce(nil, nil)

	ctx := t.Context()

	client, err := mc.AllReduce(ctx)
	if err != nil {
		t.Errorf("AllReduce: unexpected error: %v", err)
	}

	if client != nil {
		t.Error("expected nil client")
	}

	resp, err := mc.Barrier(ctx, nil)
	if err != nil {
		t.Errorf("Barrier: unexpected error: %v", err)
	}

	if resp == nil {
		t.Error("expected non-nil BarrierResponse")
	}

	bResp, err := mc.Broadcast(ctx, nil)
	if err != nil {
		t.Errorf("Broadcast: unexpected error: %v", err)
	}

	if bResp == nil {
		t.Error("expected non-nil BroadcastResponse")
	}

	mc.AssertExpectations(t)
}

func TestMockClientFactory(t *testing.T) {
	client := MockClientFactory(nil)
	if client == nil {
		t.Error("expected non-nil client")
	}

	if _, ok := client.(*CustomMockDistributedServiceClient); !ok {
		t.Errorf("expected *CustomMockDistributedServiceClient, got %T", client)
	}
}
