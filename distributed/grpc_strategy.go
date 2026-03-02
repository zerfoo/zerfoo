package distributed

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/zerfoo/zerfoo/distributed/pb"
	"github.com/zerfoo/zerfoo/log"
	metrics "github.com/zerfoo/zerfoo/metrics/runtime"
	"github.com/zerfoo/zerfoo/tensor"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// GrpcStrategy implements InternalStrategy[T] using gRPC transport.
// It connects to the coordinator for registration, starts a local
// gRPC server (workerService) for incoming RPCs, and connects to
// peers for outgoing RPCs.
type GrpcStrategy[T tensor.Numeric] struct {
	rank int
	size int

	workerAddr    string
	service       *workerService
	serverManager ServerManager
	networkMgr    NetworkManager

	coordClient CoordinatorClient
	coordConn   *grpc.ClientConn

	peerClients []pb.DistributedServiceClient
	peerConns   []*grpc.ClientConn

	logger    log.Logger
	collector metrics.Collector

	shutdownOnce sync.Once
}

// GrpcStrategyConfig holds configuration for creating a GrpcStrategy.
type GrpcStrategyConfig struct {
	WorkerAddress  string
	WorkerID       string
	ServerManager  ServerManager
	NetworkManager NetworkManager
	Dialer         Dialer
	Logger         log.Logger
	Collector      metrics.Collector
}

// NewGrpcStrategy creates a new GrpcStrategy with the given configuration.
func NewGrpcStrategy[T tensor.Numeric](cfg GrpcStrategyConfig) *GrpcStrategy[T] {
	if cfg.Logger == nil {
		cfg.Logger = log.Nop()
	}
	if cfg.Collector == nil {
		cfg.Collector = metrics.Nop()
	}
	return &GrpcStrategy[T]{
		workerAddr:    cfg.WorkerAddress,
		serverManager: cfg.ServerManager,
		networkMgr:    cfg.NetworkManager,
		logger:        cfg.Logger,
		collector:     cfg.Collector,
	}
}

// Init registers with the coordinator, starts the local gRPC server, and
// connects to all peers.
func (s *GrpcStrategy[T]) Init(rank, size int, coordinatorAddress string) error {
	_ = rank // rank is assigned by the coordinator
	_ = size // size is determined by the coordinator response

	// Connect to the coordinator.
	conn, err := grpc.NewClient(coordinatorAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("failed to connect to coordinator: %w", err)
	}
	s.coordConn = conn
	s.coordClient = pb.NewCoordinatorClient(conn)

	// Register with the coordinator.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	resp, err := s.coordClient.RegisterWorker(ctx, &pb.RegisterWorkerRequest{
		WorkerId: s.workerAddr,
		Address:  s.workerAddr,
	})
	if err != nil {
		return fmt.Errorf("failed to register with coordinator: %w", err)
	}

	s.rank = int(resp.Rank)
	s.size = len(resp.Peers)

	s.logger.Info("registered with coordinator",
		"rank", fmt.Sprintf("%d", s.rank),
		"size", fmt.Sprintf("%d", s.size),
	)

	// Create worker service.
	s.service = NewWorkerService(int32(s.rank), int32(s.size), s.logger)
	s.service.SetCollector(s.collector)

	// Start gRPC server with worker service.
	if s.serverManager != nil {
		if startErr := s.serverManager.Start(
			s.workerAddr,
			s.service,
			&pb.DistributedService_ServiceDesc,
		); startErr != nil {
			return fmt.Errorf("failed to start gRPC server: %w", startErr)
		}
	}

	// Connect to peers.
	if s.networkMgr != nil && s.size > 1 {
		clients, conns, connErr := s.networkMgr.ConnectToPeers(
			resp.Peers, s.rank, 10*time.Second,
		)
		if connErr != nil {
			return fmt.Errorf("failed to connect to peers: %w", connErr)
		}
		s.peerClients = clients
		s.peerConns = conns
	}

	return nil
}

// AllReduceGradients performs a star-topology all-reduce. Root (rank 0)
// collects gradients from all peers, averages them, and sends the result back.
// Non-root workers send gradients to root and receive the averaged result.
func (s *GrpcStrategy[T]) AllReduceGradients(gradients map[string]*tensor.TensorNumeric[T]) error {
	start := time.Now()
	defer func() {
		s.collector.Counter("allreduce_client_count").Inc()
		s.collector.Histogram("allreduce_client_duration_seconds",
			[]float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}).
			Observe(time.Since(start).Seconds())
	}()

	// Convert gradients to proto.
	protoTensors := make(map[string]*pb.Tensor, len(gradients))
	for name, t := range gradients {
		protoTensors[name] = tensorToProto(t)
	}

	if s.rank == 0 {
		return s.allReduceAsRoot(gradients, protoTensors)
	}
	return s.allReduceAsWorker(gradients, protoTensors)
}

// allReduceAsRoot handles the root worker's all-reduce logic.
func (s *GrpcStrategy[T]) allReduceAsRoot(
	gradients map[string]*tensor.TensorNumeric[T],
	protoTensors map[string]*pb.Tensor,
) error {
	// Create a new session and submit root's own tensors.
	s.service.NewSession()
	s.service.SetLocalTensors(protoTensors)

	// Wait for all peers to submit (they call AllReduce RPC on this server).
	session := s.service.getSession()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	result := session.WaitForResult(ctx)
	if result == nil {
		return errors.New("allreduce timed out waiting for peers")
	}

	// Update gradients in place with the averaged result.
	for name, t := range result {
		if grad, ok := gradients[name]; ok {
			updateTensorFromProto(grad, t)
		}
	}
	return nil
}

// allReduceAsWorker handles a non-root worker's all-reduce logic.
func (s *GrpcStrategy[T]) allReduceAsWorker(
	gradients map[string]*tensor.TensorNumeric[T],
	protoTensors map[string]*pb.Tensor,
) error {
	// Open AllReduce stream to root (rank 0).
	if len(s.peerClients) == 0 || s.peerClients[0] == nil {
		return errors.New("no connection to root worker")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stream, err := s.peerClients[0].AllReduce(ctx)
	if err != nil {
		return fmt.Errorf("failed to open AllReduce stream: %w", err)
	}

	// Send all gradients.
	for name, t := range protoTensors {
		if sendErr := stream.Send(&pb.AllReduceRequest{Name: name, Tensor: t}); sendErr != nil {
			return fmt.Errorf("failed to send gradient %s: %w", name, sendErr)
		}
	}
	if err := stream.CloseSend(); err != nil {
		return fmt.Errorf("failed to close send: %w", err)
	}

	// Receive averaged result.
	results := make(map[string]*pb.Tensor)
	for {
		resp, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			return fmt.Errorf("failed to recv result: %w", recvErr)
		}
		results[resp.Name] = resp.Tensor
	}

	// Update gradients in place.
	for name, t := range results {
		if grad, ok := gradients[name]; ok {
			updateTensorFromProto(grad, t)
		}
	}
	return nil
}

// Barrier synchronizes all workers via the root's barrier service.
func (s *GrpcStrategy[T]) Barrier() error {
	start := time.Now()
	defer func() {
		s.collector.Counter("barrier_client_count").Inc()
		s.collector.Histogram("barrier_client_duration_seconds",
			[]float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}).
			Observe(time.Since(start).Seconds())
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if s.rank == 0 {
		// Root participates by calling its own barrier.
		return s.service.barrier.arrive(ctx)
	}

	// Non-root calls Barrier RPC on root.
	if len(s.peerClients) == 0 || s.peerClients[0] == nil {
		return errors.New("no connection to root worker")
	}
	_, err := s.peerClients[0].Barrier(ctx, &pb.BarrierRequest{Rank: int32(s.rank)})
	return err
}

// BroadcastTensor broadcasts a tensor from rootRank to all other workers.
func (s *GrpcStrategy[T]) BroadcastTensor(t *tensor.TensorNumeric[T], rootRank int) error {
	start := time.Now()
	defer func() {
		s.collector.Counter("broadcast_client_count").Inc()
		s.collector.Histogram("broadcast_client_duration_seconds",
			[]float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}).
			Observe(time.Since(start).Seconds())
	}()

	name := "broadcast"

	if s.rank == rootRank {
		// Root sets the tensor on the service for peers to retrieve.
		s.service.SetBroadcastTensor(name, tensorToProto(t))
		return nil
	}

	// Non-root retrieves the tensor from root.
	if len(s.peerClients) == 0 || s.peerClients[rootRank] == nil {
		return fmt.Errorf("no connection to root worker (rank %d)", rootRank)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := s.peerClients[rootRank].Broadcast(ctx, &pb.BroadcastRequest{Name: name})
	if err != nil {
		return fmt.Errorf("broadcast recv failed: %w", err)
	}

	updateTensorFromProto(t, resp.Tensor)
	return nil
}

// Rank returns the worker's rank.
func (s *GrpcStrategy[T]) Rank() int { return s.rank }

// Size returns the total number of workers.
func (s *GrpcStrategy[T]) Size() int { return s.size }

// Shutdown gracefully shuts down the strategy.
func (s *GrpcStrategy[T]) Shutdown() {
	s.shutdownOnce.Do(func() {
		s.logger.Info("shutting down GrpcStrategy", "rank", fmt.Sprintf("%d", s.rank))

		// Unregister from coordinator.
		if s.coordClient != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, err := s.coordClient.UnregisterWorker(ctx, &pb.UnregisterWorkerRequest{
				WorkerId: s.workerAddr,
			})
			if err != nil {
				s.logger.Warn("failed to unregister from coordinator", "error", err.Error())
			}
		}

		// Close peer connections.
		if s.networkMgr != nil {
			s.networkMgr.CloseConnections(s.peerConns)
		}

		// Stop gRPC server.
		if s.serverManager != nil {
			s.serverManager.GracefulStop()
		}

		// Close coordinator connection.
		if s.coordConn != nil {
			if err := s.coordConn.Close(); err != nil {
				s.logger.Warn("failed to close coordinator connection", "error", err.Error())
			}
		}
	})
}

// Close satisfies the shutdown.Closer interface.
func (s *GrpcStrategy[T]) Close(_ context.Context) error {
	s.Shutdown()
	return nil
}

// --- Tensor conversion helpers ---

// tensorToProto converts a tensor.TensorNumeric[T] to a pb.Tensor.
// For T=float32 this is a direct copy. For T=float64 values are narrowed.
func tensorToProto[T tensor.Numeric](t *tensor.TensorNumeric[T]) *pb.Tensor {
	if t == nil {
		return nil
	}
	data := t.Data()
	shape := t.Shape()

	protoData := make([]float32, len(data))
	for i, v := range data {
		protoData[i] = float32(v)
	}

	protoShape := make([]int32, len(shape))
	for i, v := range shape {
		protoShape[i] = int32(v)
	}

	return &pb.Tensor{Shape: protoShape, Data: protoData}
}

// protoToTensor converts a pb.Tensor to a tensor.TensorNumeric[T].
func protoToTensor[T tensor.Numeric](p *pb.Tensor) (*tensor.TensorNumeric[T], error) {
	if p == nil {
		return nil, errors.New("proto tensor is nil")
	}
	shape := make([]int, len(p.Shape))
	for i, v := range p.Shape {
		shape[i] = int(v)
	}

	data := make([]T, len(p.Data))
	for i, v := range p.Data {
		data[i] = T(v)
	}

	return tensor.New[T](shape, data)
}

// updateTensorFromProto updates a tensor's data in place from a pb.Tensor.
func updateTensorFromProto[T tensor.Numeric](t *tensor.TensorNumeric[T], p *pb.Tensor) {
	if t == nil || p == nil {
		return
	}
	data := t.Data()
	for i := range data {
		if i < len(p.Data) {
			data[i] = T(p.Data[i])
		}
	}
}

// Static interface assertion.
var _ InternalStrategy[float32] = (*GrpcStrategy[float32])(nil)
