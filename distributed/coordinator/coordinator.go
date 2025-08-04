// Package coordinator provides a distributed training coordinator.
package coordinator

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
	"time"

	"github.com/zerfoo/zerfoo/distributed/pb"
	"google.golang.org/grpc"
)

// Coordinator implements the pb.CoordinatorServer interface.
// It manages the state of the distributed training cluster.
type Coordinator struct {
	pb.UnimplementedCoordinatorServer
	mu          sync.Mutex
	workers     map[string]*WorkerInfo
	ranks       map[int]string
	checkpoints map[string]*CheckpointInfo
	nextRank    int
	server      *grpc.Server
	out         io.Writer // for logging
	logger      *log.Logger
	lis         net.Listener
	timeout     time.Duration
}

// WorkerInfo holds information about a worker in the cluster.
type WorkerInfo struct {
	ID            string
	Address       string
	Rank          int
	LastHeartbeat time.Time
}

// CheckpointInfo holds information about a checkpoint.

// CheckpointInfo holds information about a checkpoint.
type CheckpointInfo struct {
	ID        string
	Epoch     int32
	Path      string
	Workers   map[string]bool
	Completed bool
}

// NewCoordinator creates a new Coordinator.
func NewCoordinator(out io.Writer, timeout time.Duration) *Coordinator {
	logger := log.New(out, "coordinator: ", log.LstdFlags)
	c := &Coordinator{
		workers:     make(map[string]*WorkerInfo),
		ranks:       make(map[int]string),
		checkpoints: make(map[string]*CheckpointInfo),
		out:         out,
		logger:      logger,
		timeout:     timeout,
	}
	go c.reaper()
	return c
}

// Start starts the coordinator service on the given address.
func (c *Coordinator) Start(address string) error {
	lis, err := net.Listen("tcp", address)
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}
	c.start(lis)
	return nil
}

// start starts the coordinator service on the given listener.
func (c *Coordinator) start(lis net.Listener) {
	c.lis = lis
	c.server = grpc.NewServer()
	pb.RegisterCoordinatorServer(c.server, c)
	c.logger.Printf("starting gRPC server on %s", lis.Addr().String())
	go func() {
		if err := c.server.Serve(lis); err != nil && err != grpc.ErrServerStopped {
			c.logger.Printf("gRPC server failed: %v", err)
		}
	}()
}

// Addr returns the address the coordinator is listening on.
func (c *Coordinator) Addr() net.Addr {
	if c.lis == nil {
		return nil
	}
	return c.lis.Addr()
}

// Stop gracefully stops the coordinator service.
func (c *Coordinator) Stop() {
	if c.server != nil {
		c.logger.Println("stopping gRPC server")
		c.server.GracefulStop()
	}
}

// GracefulStop gracefully stops the coordinator service.
func (c *Coordinator) GracefulStop() {
	if c.server != nil {
		c.server.GracefulStop()
	}
}

func (c *Coordinator) reaper() {
	ticker := time.NewTicker(c.timeout / 2)
	defer ticker.Stop()
	for range ticker.C {
		c.mu.Lock()
		for id, worker := range c.workers {
			if time.Since(worker.LastHeartbeat) > c.timeout {
				c.logger.Printf("worker %s timed out", id)
				delete(c.workers, id)
				delete(c.ranks, worker.Rank)
			}
		}
		c.mu.Unlock()
	}
}

// RegisterWorker registers a new worker with the coordinator.
func (c *Coordinator) RegisterWorker(_ context.Context, req *pb.RegisterWorkerRequest) (*pb.RegisterWorkerResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if req.WorkerId == "" {
		return nil, errors.New("worker id cannot be empty")
	}

	if _, ok := c.workers[req.WorkerId]; ok {
		c.logger.Printf("worker %s already registered", req.WorkerId)
		return nil, fmt.Errorf("worker %s already registered", req.WorkerId)
	}

	rank := c.nextRank
	c.nextRank++

	w := &WorkerInfo{
		ID:            req.WorkerId,
		Address:       req.Address,
		Rank:          rank,
		LastHeartbeat: time.Now(),
	}
	c.workers[req.WorkerId] = w
	c.ranks[rank] = req.WorkerId
	c.logger.Printf("registered worker %s at address %s with rank %d", req.WorkerId, req.Address, rank)

	peers := make([]string, 0, len(c.workers))
	for r := 0; r < c.nextRank; r++ {
		workerID, ok := c.ranks[r]
		if !ok {
			// This should not happen, but if it does, we should log it.
			c.logger.Printf("rank %d not found in ranks map", r)
			continue
		}
		worker, ok := c.workers[workerID]
		if !ok {
			// This should not happen, but if it does, we should log it.
			c.logger.Printf("worker %s not found in workers map", workerID)
			continue
		}
		peers = append(peers, worker.Address)
	}

	// Safe conversion check for rank
	if rank > int(^uint32(0)>>1) {
		return nil, fmt.Errorf("rank %d exceeds int32 maximum value", rank)
	}
	return &pb.RegisterWorkerResponse{
		Rank:  int32(rank), // #nosec G115 - Range checked above
		Peers: peers,
	}, nil
}

// UnregisterWorker removes a worker from the coordinator.
func (c *Coordinator) UnregisterWorker(_ context.Context, req *pb.UnregisterWorkerRequest) (*pb.UnregisterWorkerResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if req.WorkerId == "" {
		return nil, errors.New("worker id cannot be empty")
	}

	w, ok := c.workers[req.WorkerId]
	if !ok {
		c.logger.Printf("worker %s not found for unregistration", req.WorkerId)
		return nil, fmt.Errorf("worker %s not found", req.WorkerId)
	}

	delete(c.workers, req.WorkerId)
	delete(c.ranks, w.Rank)
	c.logger.Printf("unregistered worker %s", req.WorkerId)

	return &pb.UnregisterWorkerResponse{}, nil
}

// Heartbeat is called by workers to signal that they are still alive.
func (c *Coordinator) Heartbeat(_ context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if req.WorkerId == "" {
		return nil, errors.New("worker id cannot be empty")
	}

	w, ok := c.workers[req.WorkerId]
	if !ok {
		c.logger.Printf("worker %s not found for heartbeat", req.WorkerId)
		return nil, fmt.Errorf("worker %s not found", req.WorkerId)
	}

	w.LastHeartbeat = time.Now()
	c.logger.Printf("received heartbeat from worker %s", req.WorkerId)

	return &pb.HeartbeatResponse{Status: "OK"}, nil
}

// StartCheckpoint initiates a new checkpoint process.
func (c *Coordinator) StartCheckpoint(_ context.Context, req *pb.StartCheckpointRequest) (*pb.StartCheckpointResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	checkpointID := fmt.Sprintf("ckpt-%d", req.Epoch)
	c.logger.Printf("starting checkpoint %s for epoch %d, path: %s", checkpointID, req.Epoch, req.Path)

	workers := make(map[string]bool)
	for id := range c.workers {
		workers[id] = false
	}

	// Safe conversion check for epoch
	if req.Epoch > int64(^uint32(0)>>1) {
		return nil, fmt.Errorf("epoch %d exceeds int32 maximum value", req.Epoch)
	}
	c.checkpoints[checkpointID] = &CheckpointInfo{
		ID:      checkpointID,
		Epoch:   int32(req.Epoch), // #nosec G115 - Range checked above
		Path:    req.Path,
		Workers: workers,
	}

	return &pb.StartCheckpointResponse{CheckpointId: checkpointID}, nil
}

// EndCheckpoint is called by workers to report the completion of a checkpoint.
func (c *Coordinator) EndCheckpoint(_ context.Context, req *pb.EndCheckpointRequest) (*pb.EndCheckpointResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if req.WorkerId == "" {
		return nil, errors.New("worker id cannot be empty")
	}

	checkpoint, ok := c.checkpoints[req.CheckpointId]
	if !ok {
		return nil, fmt.Errorf("checkpoint %s not found", req.CheckpointId)
	}

	checkpoint.Workers[req.WorkerId] = true
	c.logger.Printf("worker %s finished checkpoint %s for epoch %d", req.WorkerId, req.CheckpointId, req.Epoch)

	completed := true
	for _, status := range checkpoint.Workers {
		if !status {
			completed = false
			break
		}
	}

	if completed {
		checkpoint.Completed = true
		c.logger.Printf("checkpoint %s for epoch %d completed", req.CheckpointId, req.Epoch)
	}

	return &pb.EndCheckpointResponse{}, nil
}
