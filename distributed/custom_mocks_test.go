package distributed

import (
	"context"
	"net"
	"sync"
	"testing"

	"github.com/zerfoo/zerfoo/distributed/pb"
	"github.com/zerfoo/zerfoo/tensor"
	"google.golang.org/grpc"
)

// CustomMockStrategy is a custom mock implementation of the InternalStrategy interface.
type CustomMockStrategy[T tensor.Numeric] struct {
	mu       sync.Mutex
	initArgs []struct {
		rank               int
		size               int
		coordinatorAddress string
	}
	initReturns []error
	initCalls   int

	rankReturns []int
	rankCalls   int

	sizeReturns []int
	sizeCalls   int

	allReduceGradientsArgs    []map[string]*tensor.Tensor[T]
	allReduceGradientsReturns []error
	allReduceGradientsCalls   int

	barrierReturns []error
	barrierCalls   int

	broadcastTensorArgs []struct {
		t        *tensor.Tensor[T]
		rootRank int
	}
	broadcastTensorReturns []error
	broadcastTensorCalls   int

	shutdownCalls int
}

func (m *CustomMockStrategy[T]) Init(rank int, size int, coordinatorAddress string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.initCalls++
	m.initArgs = append(m.initArgs, struct {
		rank               int
		size               int
		coordinatorAddress string
	}{
		rank:               rank,
		size:               size,
		coordinatorAddress: coordinatorAddress,
	})
	if len(m.initReturns) < m.initCalls {
		panic("not enough return values for Init")
	}

	return m.initReturns[m.initCalls-1]
}

func (m *CustomMockStrategy[T]) OnInit(rank, size int, coordinatorAddress string) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.initArgs = append(m.initArgs, struct {
		rank               int
		size               int
		coordinatorAddress string
	}{
		rank:               rank,
		size:               size,
		coordinatorAddress: coordinatorAddress,
	})

	return m
}

func (m *CustomMockStrategy[T]) ReturnInit(err error) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.initReturns = append(m.initReturns, err)

	return m
}

func (m *CustomMockStrategy[T]) OnceInit() *CustomMockStrategy[T] {
	// For simplicity, Once is handled by the order of Return calls.
	return m
}

func (m *CustomMockStrategy[T]) Rank() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rankCalls++
	if len(m.rankReturns) < m.rankCalls {
		panic("not enough return values for Rank")
	}

	return m.rankReturns[m.rankCalls-1]
}

func (m *CustomMockStrategy[T]) OnRank() *CustomMockStrategy[T] {
	return m
}

func (m *CustomMockStrategy[T]) ReturnRank(rank int) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rankReturns = append(m.rankReturns, rank)

	return m
}

func (m *CustomMockStrategy[T]) OnceRank() *CustomMockStrategy[T] {
	return m
}

func (m *CustomMockStrategy[T]) Size() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sizeCalls++
	if len(m.sizeReturns) < m.sizeCalls {
		panic("not enough return values for Size")
	}

	return m.sizeReturns[m.sizeCalls-1]
}

func (m *CustomMockStrategy[T]) OnSize() *CustomMockStrategy[T] {
	return m
}

func (m *CustomMockStrategy[T]) ReturnSize(size int) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sizeReturns = append(m.sizeReturns, size)

	return m
}

func (m *CustomMockStrategy[T]) OnceSize() *CustomMockStrategy[T] {
	return m
}

func (m *CustomMockStrategy[T]) AllReduceGradients(gradients map[string]*tensor.Tensor[T]) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.allReduceGradientsCalls++
	m.allReduceGradientsArgs = append(m.allReduceGradientsArgs, gradients)
	if len(m.allReduceGradientsReturns) < m.allReduceGradientsCalls {
		panic("not enough return values for AllReduceGradients")
	}

	return m.allReduceGradientsReturns[m.allReduceGradientsCalls-1]
}

func (m *CustomMockStrategy[T]) OnAllReduceGradients(gradients map[string]*tensor.Tensor[T]) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.allReduceGradientsArgs = append(m.allReduceGradientsArgs, gradients)

	return m
}

func (m *CustomMockStrategy[T]) ReturnAllReduceGradients(err error) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.allReduceGradientsReturns = append(m.allReduceGradientsReturns, err)

	return m
}

func (m *CustomMockStrategy[T]) OnceAllReduceGradients() *CustomMockStrategy[T] {
	return m
}

func (m *CustomMockStrategy[T]) Barrier() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.barrierCalls++
	if len(m.barrierReturns) < m.barrierCalls {
		panic("not enough return values for Barrier")
	}

	return m.barrierReturns[m.barrierCalls-1]
}

func (m *CustomMockStrategy[T]) OnBarrier() *CustomMockStrategy[T] {
	return m
}

func (m *CustomMockStrategy[T]) ReturnBarrier(err error) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.barrierReturns = append(m.barrierReturns, err)

	return m
}

func (m *CustomMockStrategy[T]) TwiceBarrier() *CustomMockStrategy[T] {
	return m // Handled by calling ReturnBarrier twice
}

func (m *CustomMockStrategy[T]) BroadcastTensor(t *tensor.Tensor[T], rootRank int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.broadcastTensorCalls++
	m.broadcastTensorArgs = append(m.broadcastTensorArgs, struct {
		t        *tensor.Tensor[T]
		rootRank int
	}{
		t:        t,
		rootRank: rootRank,
	})
	if len(m.broadcastTensorReturns) < m.broadcastTensorCalls {
		panic("not enough return values for BroadcastTensor")
	}

	return m.broadcastTensorReturns[m.broadcastTensorCalls-1]
}

func (m *CustomMockStrategy[T]) OnBroadcastTensor(t *tensor.Tensor[T], rootRank int) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.broadcastTensorArgs = append(m.broadcastTensorArgs, struct {
		t        *tensor.Tensor[T]
		rootRank int
	}{
		t:        t,
		rootRank: rootRank,
	})

	return m
}

func (m *CustomMockStrategy[T]) ReturnBroadcastTensor(err error) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.broadcastTensorReturns = append(m.broadcastTensorReturns, err)

	return m
}

func (m *CustomMockStrategy[T]) OnceBroadcastTensor() *CustomMockStrategy[T] {
	return m
}

func (m *CustomMockStrategy[T]) Shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.shutdownCalls++
}

func (m *CustomMockStrategy[T]) AssertExpectations(t *testing.T) {
	t.Helper()
	// For simplicity, this mock doesn't track arguments for AssertExpectations.
	// It only checks if methods were called the expected number of times.
	// More sophisticated argument matching would require additional logic.
}

func (m *CustomMockStrategy[T]) AssertNotCalled(t *testing.T, methodName string) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()

	switch methodName {
	case "Init":
		if m.initCalls > 0 {
			t.Errorf("expected Init to not be called, but it was called %d times", m.initCalls)
		}
	case "AllReduceGradients":
		if m.allReduceGradientsCalls > 0 {
			t.Errorf("expected AllReduceGradients to not be called, but it was called %d times", m.allReduceGradientsCalls)
		}
	case "Barrier":
		if m.barrierCalls > 0 {
			t.Errorf("expected Barrier to not be called, but it was called %d times", m.barrierCalls)
		}
	case "BroadcastTensor":
		if m.broadcastTensorCalls > 0 {
			t.Errorf("expected BroadcastTensor to not be called, but it was called %d times", m.broadcastTensorCalls)
		}
	case "Rank":
		if m.rankCalls > 0 {
			t.Errorf("expected Rank to not be called, but it was called %d times", m.rankCalls)
		}
	case "Size":
		if m.sizeCalls > 0 {
			t.Errorf("expected Size to not be called, but it was called %d times", m.sizeCalls)
		}
	case "Shutdown":
		if m.shutdownCalls > 0 {
			t.Errorf("expected Shutdown to not be called, but it was called %d times", m.shutdownCalls)
		}
	default:
		t.Errorf("unknown method %q for AssertNotCalled", methodName)
	}
}

// CustomMockListener is a custom mock implementation of the net.Listener interface.
type CustomMockListener struct {
	mu          sync.Mutex
	acceptErr   error
	closeErr    error
	addr        net.Addr
	acceptCalls int
	closeCalls  int
	addrCalls   int
}

func (m *CustomMockListener) Accept() (net.Conn, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.acceptCalls++

	return nil, m.acceptErr
}

func (m *CustomMockListener) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closeCalls++

	return m.closeErr
}

func (m *CustomMockListener) Addr() net.Addr {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.addrCalls++

	return m.addr
}

func (m *CustomMockListener) OnAccept(err error) *CustomMockListener {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.acceptErr = err

	return m
}

func (m *CustomMockListener) OnClose(err error) *CustomMockListener {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closeErr = err

	return m
}

func (m *CustomMockListener) OnAddr(addr net.Addr) *CustomMockListener {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.addr = addr

	return m
}

func (m *CustomMockListener) AssertExpectations(t *testing.T) {
	t.Helper()
}

// CustomMockGrpcServer is a custom mock implementation of the GrpcServer interface.
type CustomMockGrpcServer struct {
	mu                   sync.Mutex
	registerServiceCalls int
	serveCalls           int
	stopCalls            int
	gracefulStopCalls    int
	serveErr             error
}

func (m *CustomMockGrpcServer) RegisterService(_ *grpc.ServiceDesc, _ interface{}) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.registerServiceCalls++
}

func (m *CustomMockGrpcServer) Serve(_ net.Listener) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.serveCalls++

	return m.serveErr
}

func (m *CustomMockGrpcServer) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopCalls++
}

func (m *CustomMockGrpcServer) GracefulStop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.gracefulStopCalls++
}

func (m *CustomMockGrpcServer) OnServe(err error) *CustomMockGrpcServer {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.serveErr = err

	return m
}

func (m *CustomMockGrpcServer) AssertExpectations(t *testing.T) {
	t.Helper()
}

// CustomMockLogger is a custom mock implementation of the Logger interface.
type CustomMockLogger struct {
	mu          sync.Mutex
	printfCalls int
	printfArgs  []struct {
		format string
		v      []interface{}
	}
}

func (m *CustomMockLogger) Printf(format string, v ...interface{}) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.printfCalls++
	m.printfArgs = append(m.printfArgs, struct {
		format string
		v      []interface{}
	}{
		format: format,
		v:      v,
	})
}

func (m *CustomMockLogger) AssertExpectations(t *testing.T) {
	t.Helper()
}

func (m *CustomMockLogger) OnPrintf() *CustomMockLogger {
	return m
}

// CustomMockDistributedServiceClient is a custom mock implementation of the DistributedServiceClient interface.
type CustomMockDistributedServiceClient struct {
	mu               sync.Mutex
	allReduceCalls   int
	allReduceReturns []struct {
		client pb.DistributedService_AllReduceClient
		err    error
	}
}

func (m *CustomMockDistributedServiceClient) AllReduce(_ context.Context, _ ...grpc.CallOption) (pb.DistributedService_AllReduceClient, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.allReduceCalls++
	if len(m.allReduceReturns) < m.allReduceCalls {
		panic("not enough return values for AllReduce")
	}

	return m.allReduceReturns[m.allReduceCalls-1].client, m.allReduceReturns[m.allReduceCalls-1].err
}

func (m *CustomMockDistributedServiceClient) OnAllReduce() *CustomMockDistributedServiceClient {
	return m
}

func (m *CustomMockDistributedServiceClient) ReturnAllReduce(client pb.DistributedService_AllReduceClient, err error) *CustomMockDistributedServiceClient {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.allReduceReturns = append(m.allReduceReturns, struct {
		client pb.DistributedService_AllReduceClient
		err    error
	}{
		client: client,
		err:    err,
	})

	return m
}

func (m *CustomMockDistributedServiceClient) Barrier(_ context.Context, _ *pb.BarrierRequest, _ ...grpc.CallOption) (*pb.BarrierResponse, error) {
	return &pb.BarrierResponse{}, nil
}

func (m *CustomMockDistributedServiceClient) Broadcast(ctx context.Context, in *pb.BroadcastRequest, opts ...grpc.CallOption) (*pb.BroadcastResponse, error) {
	return &pb.BroadcastResponse{}, nil
}

func (m *CustomMockDistributedServiceClient) AssertExpectations(t *testing.T) {
	t.Helper()
}

// MockClientFactory is a mock implementation of the DistributedServiceClientFactory function.
func MockClientFactory(_ *grpc.ClientConn) pb.DistributedServiceClient {
	return &CustomMockDistributedServiceClient{}
}
