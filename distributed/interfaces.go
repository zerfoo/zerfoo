package distributed

import (
	"context"
	"fmt"
	"net"
	"time"

	"github.com/zerfoo/zerfoo/distributed/pb"
	"github.com/zerfoo/zerfoo/tensor"
	"google.golang.org/grpc"
)

// InternalStrategy defines the interface for a distributed training strategy.
type InternalStrategy[T tensor.Numeric] interface {
	// Init initializes the strategy.
	Init(rank int, size int, coordinatorAddress string) error
	// AllReduceGradients performs an all-reduce operation on the gradients.
	AllReduceGradients(gradients map[string]*tensor.Tensor[T]) error
	// Barrier blocks until all workers have reached the barrier.
	Barrier() error
	// BroadcastTensor broadcasts a tensor from the root to all other workers.
	BroadcastTensor(t *tensor.Tensor[T], rootRank int) error
	// Rank returns the rank of the current worker.
	Rank() int
	// Size returns the total number of workers.
	Size() int
	// Shutdown cleans up the resources used by the strategy.
	Shutdown()
}

// Dialer is a function that creates a gRPC client connection.
type Dialer func(ctx context.Context, target string) (*grpc.ClientConn, error)

// DistributedServiceClientFactory is a function that creates a new DistributedServiceClient.
type DistributedServiceClientFactory func(cc *grpc.ClientConn) pb.DistributedServiceClient

// NetworkManager is an interface for managing network connections between workers.
type NetworkManager interface {
	// ConnectToPeers establishes connections to all other workers in the cluster.
	ConnectToPeers(peers []string, selfRank int, timeout time.Duration) ([]pb.DistributedServiceClient, []*grpc.ClientConn, error)
	// CloseConnections closes all the given connections.
	CloseConnections(conns []*grpc.ClientConn)
}

// GrpcServer is an interface for a gRPC server.
type GrpcServer interface {
	RegisterService(desc *grpc.ServiceDesc, impl interface{})
	Serve(lis net.Listener) error
	Stop()
	GracefulStop()
}

// ListenerFactory is a function that creates a new net.Listener.
type ListenerFactory func(network, address string) (net.Listener, error)

// ServerManager is an interface for managing the gRPC server of a worker.
type ServerManager interface {
	Start(workerAddress string, service interface{}, serviceDesc *grpc.ServiceDesc) error
	Stop()
	GracefulStop()
	SetLogger(logger Logger)
}

// CoordinatorClient is an interface for a client of the coordinator service.
type CoordinatorClient interface {
	RegisterWorker(ctx context.Context, in *pb.RegisterWorkerRequest, opts ...grpc.CallOption) (*pb.RegisterWorkerResponse, error)
	UnregisterWorker(ctx context.Context, in *pb.UnregisterWorkerRequest, opts ...grpc.CallOption) (*pb.UnregisterWorkerResponse, error)
	Heartbeat(ctx context.Context, in *pb.HeartbeatRequest, opts ...grpc.CallOption) (*pb.HeartbeatResponse, error)
}

// Logger is an interface for logging.
type Logger interface {
	Printf(format string, v ...interface{})
}

type defaultLogger struct{}

func (l *defaultLogger) Printf(format string, v ...interface{}) {
	fmt.Printf(format, v...)
}
