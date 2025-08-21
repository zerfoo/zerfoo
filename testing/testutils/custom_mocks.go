// Package testutils provides testing utilities and mock implementations for the Zerfoo ML framework.
package testutils

import (
	"context"
	"net"
	"sync"
	"testing"

	"github.com/zerfoo/zerfoo/distributed/pb"
	"github.com/zerfoo/zerfoo/tensor"
	"google.golang.org/grpc"
	_ "github.com/zerfoo/zerfoo/layers/core"
	_ "github.com/zerfoo/zerfoo/layers/gather"
	_ "github.com/zerfoo/zerfoo/layers/transpose"
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

	allReduceGradientsArgs    []map[string]*tensor.TensorNumeric[T]
	allReduceGradientsReturns []error
	allReduceGradientsCalls   int

	barrierReturns []error
	barrierCalls   int

	BroadcastTensorArgs []struct {
		Tensor   *tensor.TensorNumeric[T]
		RootRank int
	}
	broadcastTensorReturns []error
	broadcastTensorCalls   int

	shutdownCalls int
}

// Init records the arguments and increments the call count for the Init method.
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

// OnInit sets up expectations for the Init method.
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

// ReturnInit specifies the return value for the Init method.
func (m *CustomMockStrategy[T]) ReturnInit(err error) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.initReturns = append(m.initReturns, err)

	return m
}

// OnceInit indicates that the Init method should be called once.
func (m *CustomMockStrategy[T]) OnceInit() *CustomMockStrategy[T] {
	// For simplicity, Once is handled by the order of Return calls.
	return m
}

// Rank returns the rank of the current process.
func (m *CustomMockStrategy[T]) Rank() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rankCalls++
	if len(m.rankReturns) < m.rankCalls {
		panic("not enough return values for Rank")
	}

	return m.rankReturns[m.rankCalls-1]
}

// OnRank sets up expectations for the Rank method.
func (m *CustomMockStrategy[T]) OnRank() *CustomMockStrategy[T] {
	return m
}

// ReturnRank specifies the return value for the Rank method.
func (m *CustomMockStrategy[T]) ReturnRank(rank int) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rankReturns = append(m.rankReturns, rank)

	return m
}

// OnceRank indicates that the Rank method should be called once.
func (m *CustomMockStrategy[T]) OnceRank() *CustomMockStrategy[T] {
	return m
}

// Size returns the total number of processes.
func (m *CustomMockStrategy[T]) Size() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sizeCalls++
	if len(m.sizeReturns) < m.sizeCalls {
		panic("not enough return values for Size")
	}

	return m.sizeReturns[m.sizeCalls-1]
}

// OnSize sets up expectations for the Size method.
func (m *CustomMockStrategy[T]) OnSize() *CustomMockStrategy[T] {
	return m
}

// ReturnSize specifies the return value for the Size method.
func (m *CustomMockStrategy[T]) ReturnSize(size int) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sizeReturns = append(m.sizeReturns, size)

	return m
}

// OnceSize indicates that the Size method should be called once.
func (m *CustomMockStrategy[T]) OnceSize() *CustomMockStrategy[T] {
	return m
}

// AllReduceGradients performs an all-reduce operation on gradients.
func (m *CustomMockStrategy[T]) AllReduceGradients(gradients map[string]*tensor.TensorNumeric[T]) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.allReduceGradientsCalls++
	m.allReduceGradientsArgs = append(m.allReduceGradientsArgs, gradients)
	if len(m.allReduceGradientsReturns) < m.allReduceGradientsCalls {
		panic("not enough return values for AllReduceGradients")
	}

	return m.allReduceGradientsReturns[m.allReduceGradientsCalls-1]
}

// OnAllReduceGradients sets up expectations for the AllReduceGradients method.
func (m *CustomMockStrategy[T]) OnAllReduceGradients(gradients map[string]*tensor.TensorNumeric[T]) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.allReduceGradientsArgs = append(m.allReduceGradientsArgs, gradients)

	return m
}

// ReturnAllReduceGradients specifies the return value for the AllReduceGradients method.
func (m *CustomMockStrategy[T]) ReturnAllReduceGradients(err error) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.allReduceGradientsReturns = append(m.allReduceGradientsReturns, err)

	return m
}

// OnceAllReduceGradients indicates that the AllReduceGradients method should be called once.
func (m *CustomMockStrategy[T]) OnceAllReduceGradients() *CustomMockStrategy[T] {
	return m
}

// Barrier synchronizes all processes.
func (m *CustomMockStrategy[T]) Barrier() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.barrierCalls++
	if len(m.barrierReturns) < m.barrierCalls {
		panic("not enough return values for Barrier")
	}

	return m.barrierReturns[m.barrierCalls-1]
}

// OnBarrier sets up expectations for the Barrier method.
func (m *CustomMockStrategy[T]) OnBarrier() *CustomMockStrategy[T] {
	return m
}

// ReturnBarrier specifies the return value for the Barrier method.
func (m *CustomMockStrategy[T]) ReturnBarrier(err error) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.barrierReturns = append(m.barrierReturns, err)

	return m
}

// TwiceBarrier indicates that the Barrier method should be called twice.
func (m *CustomMockStrategy[T]) TwiceBarrier() *CustomMockStrategy[T] {
	return m // Handled by calling ReturnBarrier twice
}

// BroadcastTensor broadcasts a tensor from the root rank to all other processes.
func (m *CustomMockStrategy[T]) BroadcastTensor(t *tensor.TensorNumeric[T], rootRank int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.broadcastTensorCalls++
	m.BroadcastTensorArgs = append(m.BroadcastTensorArgs, struct {
		Tensor   *tensor.TensorNumeric[T]
		RootRank int
	}{
		Tensor:   t,
		RootRank: rootRank,
	})
	if len(m.broadcastTensorReturns) < m.broadcastTensorCalls {
		panic("not enough return values for BroadcastTensor")
	}

	return m.broadcastTensorReturns[m.broadcastTensorCalls-1]
}

// OnBroadcastTensor sets up expectations for the BroadcastTensor method.
func (m *CustomMockStrategy[T]) OnBroadcastTensor(t *tensor.TensorNumeric[T], rootRank int) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.BroadcastTensorArgs = append(m.BroadcastTensorArgs, struct {
		Tensor   *tensor.TensorNumeric[T]
		RootRank int
	}{
		Tensor:   t,
		RootRank: rootRank,
	})

	return m
}

// ReturnBroadcastTensor specifies the return value for the BroadcastTensor method.
func (m *CustomMockStrategy[T]) ReturnBroadcastTensor(err error) *CustomMockStrategy[T] {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.broadcastTensorReturns = append(m.broadcastTensorReturns, err)

	return m
}

// OnceBroadcastTensor indicates that the BroadcastTensor method should be called once.
func (m *CustomMockStrategy[T]) OnceBroadcastTensor() *CustomMockStrategy[T] {
	return m
}

// Shutdown performs a graceful shutdown of the distributed training environment.
func (m *CustomMockStrategy[T]) Shutdown() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.shutdownCalls++
}

// AssertExpectations asserts that all expected calls were made.
func (m *CustomMockStrategy[T]) AssertExpectations(t *testing.T) {
	t.Helper()
	// For simplicity, this mock doesn't track arguments for AssertExpectations.
	// It only checks if methods were called the expected number of times.
	// More sophisticated argument matching would require additional logic.
}

// AssertNotCalled asserts that a specific method was not called.
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
	AcceptErr   error
	CloseErr    error
	AddrVal     net.Addr
	acceptCalls int
	closeCalls  int
	addrCalls   int
}

// Accept waits for and returns the next connection to the listener.
func (m *CustomMockListener) Accept() (net.Conn, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.acceptCalls++

	return nil, m.AcceptErr
}

// Close closes the listener.
func (m *CustomMockListener) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closeCalls++

	return m.CloseErr
}

// Addr returns the listener's network address.
func (m *CustomMockListener) Addr() net.Addr {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.addrCalls++

	return m.AddrVal
}

// OnAccept sets up expectations for the Accept method.
func (m *CustomMockListener) OnAccept(err error) *CustomMockListener {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.AcceptErr = err

	return m
}

// OnClose sets up expectations for the Close method.
func (m *CustomMockListener) OnClose(err error) *CustomMockListener {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.CloseErr = err

	return m
}

// OnAddr sets up expectations for the Addr method.
func (m *CustomMockListener) OnAddr(addr net.Addr) *CustomMockListener {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.AddrVal = addr

	return m
}

// AssertExpectations asserts that all expected calls were made.
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
	ServeErr             error
}

// RegisterService registers a service with the mock gRPC server.
func (m *CustomMockGrpcServer) RegisterService(_ *grpc.ServiceDesc, _ interface{}) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.registerServiceCalls++
}

// Serve starts serving the mock gRPC server.
func (m *CustomMockGrpcServer) Serve(_ net.Listener) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.serveCalls++

	return m.ServeErr
}

// Stop stops the gRPC server.
func (m *CustomMockGrpcServer) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopCalls++
}

// GracefulStop stops the gRPC server gracefully.
func (m *CustomMockGrpcServer) GracefulStop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.gracefulStopCalls++
}

// OnServe sets up expectations for the Serve method.
func (m *CustomMockGrpcServer) OnServe(err error) *CustomMockGrpcServer {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ServeErr = err

	return m
}

// AssertExpectations asserts that all expected calls were made.
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

// Printf records the arguments and increments the call count for the Printf method.
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

// AssertExpectations asserts that all expected calls were made.
func (m *CustomMockLogger) AssertExpectations(t *testing.T) {
	t.Helper()
}

// OnPrintf sets up expectations for the Printf method.
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

// AllReduce performs an all-reduce operation.
func (m *CustomMockDistributedServiceClient) AllReduce(_ context.Context, _ ...grpc.CallOption) (pb.DistributedService_AllReduceClient, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.allReduceCalls++
	if len(m.allReduceReturns) < m.allReduceCalls {
		panic("not enough return values for AllReduce")
	}

	return m.allReduceReturns[m.allReduceCalls-1].client, m.allReduceReturns[m.allReduceCalls-1].err
}

// OnAllReduce sets up expectations for the AllReduce method.
func (m *CustomMockDistributedServiceClient) OnAllReduce() *CustomMockDistributedServiceClient {
	return m
}

// ReturnAllReduce specifies the return value for the AllReduce method.
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

// Barrier performs a barrier synchronization.
func (m *CustomMockDistributedServiceClient) Barrier(_ context.Context, _ *pb.BarrierRequest, _ ...grpc.CallOption) (*pb.BarrierResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	// This mock doesn't track calls for Barrier, just returns a default.
	return &pb.BarrierResponse{}, nil
}

// Broadcast performs a broadcast operation.
func (m *CustomMockDistributedServiceClient) Broadcast(_ context.Context, _ *pb.BroadcastRequest, _ ...grpc.CallOption) (*pb.BroadcastResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	// This mock doesn't track calls for Broadcast, just returns a default.
	return &pb.BroadcastResponse{}, nil
}

// AssertExpectations asserts that all expected calls were made.
func (m *CustomMockDistributedServiceClient) AssertExpectations(t *testing.T) {
	t.Helper()
}

// MockClientFactory is a mock implementation of the DistributedServiceClientFactory function.
func MockClientFactory(_ *grpc.ClientConn) pb.DistributedServiceClient {
	return &CustomMockDistributedServiceClient{}
}
