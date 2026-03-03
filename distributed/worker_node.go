package distributed

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/zerfoo/zerfoo/health"
	"github.com/zerfoo/zerfoo/log"
	metrics "github.com/zerfoo/zerfoo/metrics/runtime"
	"github.com/zerfoo/zerfoo/tensor"
	"google.golang.org/grpc"
)

// WorkerNodeConfig holds configuration for creating a WorkerNode.
type WorkerNodeConfig struct {
	WorkerAddress      string
	CoordinatorAddress string
	WorldSize          int
	Logger             log.Logger
	Collector          metrics.Collector
	HealthServer       *health.Server
}

// WorkerNode encapsulates a distributed training worker. It manages
// the gRPC strategy, server, and network connections, and provides
// orderly startup and shutdown semantics compatible with shutdown.Coordinator.
type WorkerNode struct {
	config   WorkerNodeConfig
	strategy *GrpcStrategy[float32]
	logger   log.Logger

	mu      sync.Mutex
	started bool
}

// NewWorkerNode creates a new WorkerNode with the given configuration.
func NewWorkerNode(cfg WorkerNodeConfig) *WorkerNode {
	if cfg.Logger == nil {
		cfg.Logger = log.Nop()
	}
	if cfg.Collector == nil {
		cfg.Collector = metrics.Nop()
	}
	return &WorkerNode{
		config: cfg,
		logger: cfg.Logger,
	}
}

// Start initializes the distributed worker: creates a gRPC server and
// strategy, registers with the coordinator, connects to peers, and
// optionally registers a health check. The context is used only for
// cancellation of the start sequence, not the lifetime of the worker.
func (wn *WorkerNode) Start(_ context.Context) error {
	wn.mu.Lock()
	defer wn.mu.Unlock()

	if wn.started {
		return errors.New("worker node already started")
	}

	srv := grpc.NewServer()
	sm := NewServerManager(srv, nil)
	nm := NewNetworkManager(nil, nil)

	strategy := NewGrpcStrategy[float32](GrpcStrategyConfig{
		WorkerAddress:  wn.config.WorkerAddress,
		ServerManager:  sm,
		NetworkManager: nm,
		Logger:         wn.config.Logger,
		Collector:      wn.config.Collector,
	})

	if err := strategy.Init(0, wn.config.WorldSize, wn.config.CoordinatorAddress); err != nil {
		return fmt.Errorf("strategy init failed: %w", err)
	}

	wn.strategy = strategy
	wn.started = true

	// Register health check if a health server is provided.
	if wn.config.HealthServer != nil {
		wn.config.HealthServer.AddReadinessCheck("distributed-worker", wn.healthCheck())
	}

	wn.logger.Info("worker node started",
		"address", wn.config.WorkerAddress,
		"rank", fmt.Sprintf("%d", strategy.Rank()),
		"size", fmt.Sprintf("%d", strategy.Size()),
	)

	return nil
}

// healthCheck returns a health.CheckFunc that reports the worker's status.
func (wn *WorkerNode) healthCheck() health.CheckFunc {
	return func() error {
		wn.mu.Lock()
		defer wn.mu.Unlock()
		if !wn.started {
			return errors.New("worker node not started")
		}
		return nil
	}
}

// Strategy returns the underlying InternalStrategy, or nil if not started.
func (wn *WorkerNode) Strategy() InternalStrategy[float32] {
	wn.mu.Lock()
	defer wn.mu.Unlock()
	if wn.strategy == nil {
		return nil
	}
	return wn.strategy
}

// Close shuts down the worker node. It satisfies the shutdown.Closer interface.
// Calling Close on an unstarted or already-closed node is safe.
func (wn *WorkerNode) Close(_ context.Context) error {
	wn.mu.Lock()
	defer wn.mu.Unlock()

	if !wn.started {
		return nil
	}

	wn.logger.Info("shutting down worker node")
	wn.strategy.Shutdown()
	wn.strategy = nil
	wn.started = false
	return nil
}

// Rank returns the worker's rank, or -1 if not started.
func (wn *WorkerNode) Rank() int {
	wn.mu.Lock()
	defer wn.mu.Unlock()
	if wn.strategy == nil {
		return -1
	}
	return wn.strategy.Rank()
}

// Size returns the total number of workers, or 0 if not started.
func (wn *WorkerNode) Size() int {
	wn.mu.Lock()
	defer wn.mu.Unlock()
	if wn.strategy == nil {
		return 0
	}
	return wn.strategy.Size()
}

// Static interface assertion for shutdown.Closer compatibility.
var _ interface{ Close(context.Context) error } = (*WorkerNode)(nil)

// Static assertion that GrpcStrategy satisfies InternalStrategy.
var _ InternalStrategy[float32] = (*GrpcStrategy[float32])(nil)

// Generic type alias for external use.
type NumericStrategy[T tensor.Numeric] = InternalStrategy[T]
