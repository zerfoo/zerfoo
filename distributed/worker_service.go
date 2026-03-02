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
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// workerService implements pb.DistributedServiceServer.
// It handles AllReduce, Barrier, and Broadcast RPCs from peers.
type workerService struct {
	pb.UnimplementedDistributedServiceServer

	rank      int32
	worldSize int32
	logger    log.Logger
	collector metrics.Collector

	// session holds the active reduce session for the current training step.
	session   *reduceSession
	sessionMu sync.Mutex

	// barrier coordinates Barrier RPCs across workers.
	barrier *barrierState

	// broadcasts stores tensors for Broadcast RPCs.
	broadcasts   map[string]*broadcastEntry
	broadcastsMu sync.Mutex
}

// broadcastEntry stores a broadcast tensor and a channel to signal availability.
type broadcastEntry struct {
	tensor *pb.Tensor
	ready  chan struct{}
}

// NewWorkerService creates a new workerService.
func NewWorkerService(rank, worldSize int32, logger log.Logger) *workerService {
	if logger == nil {
		logger = log.Nop()
	}
	return &workerService{
		rank:       rank,
		worldSize:  worldSize,
		logger:     logger,
		collector:  metrics.Nop(),
		barrier:    newBarrierState(worldSize),
		broadcasts: make(map[string]*broadcastEntry),
	}
}

// SetCollector sets the metrics collector for the worker service.
func (ws *workerService) SetCollector(c metrics.Collector) {
	if c == nil {
		c = metrics.Nop()
	}
	ws.collector = c
}

// NewSession creates a new reduce session for the current training step.
// Must be called before AllReduce streams begin for each step.
func (ws *workerService) NewSession() {
	ws.sessionMu.Lock()
	defer ws.sessionMu.Unlock()
	ws.session = newReduceSession(ws.worldSize)
}

// SetLocalTensors submits the root worker's own tensors to the active reduce session.
func (ws *workerService) SetLocalTensors(tensors map[string]*pb.Tensor) {
	ws.sessionMu.Lock()
	s := ws.session
	ws.sessionMu.Unlock()
	if s != nil {
		s.Submit(ws.rank, tensors)
	}
}

// getSession returns the current reduce session.
func (ws *workerService) getSession() *reduceSession {
	ws.sessionMu.Lock()
	defer ws.sessionMu.Unlock()
	return ws.session
}

// SetBroadcastTensor stores a tensor for broadcast retrieval by non-root workers.
func (ws *workerService) SetBroadcastTensor(name string, t *pb.Tensor) {
	ws.broadcastsMu.Lock()
	defer ws.broadcastsMu.Unlock()
	entry, ok := ws.broadcasts[name]
	if !ok {
		entry = &broadcastEntry{ready: make(chan struct{})}
		ws.broadcasts[name] = entry
	}
	entry.tensor = t
	select {
	case <-entry.ready:
		// already closed
	default:
		close(entry.ready)
	}
}

// getBroadcastEntry returns (or creates) a broadcast entry for the given name.
func (ws *workerService) getBroadcastEntry(name string) *broadcastEntry {
	ws.broadcastsMu.Lock()
	defer ws.broadcastsMu.Unlock()
	entry, ok := ws.broadcasts[name]
	if !ok {
		entry = &broadcastEntry{ready: make(chan struct{})}
		ws.broadcasts[name] = entry
	}
	return entry
}

// ClearBroadcasts removes all stored broadcast tensors.
func (ws *workerService) ClearBroadcasts() {
	ws.broadcastsMu.Lock()
	defer ws.broadcastsMu.Unlock()
	ws.broadcasts = make(map[string]*broadcastEntry)
}

// --- reduceSession ---

// reduceSession coordinates all-reduce across concurrent bidi streams.
// It collects tensors by name from each peer, waits for all peers to submit,
// computes the element-wise average, and distributes the result.
type reduceSession struct {
	worldSize int32

	mu        sync.Mutex
	cond      *sync.Cond
	submitted int32
	tensors   map[string][][]float32 // name -> slice of data from each peer
	shapes    map[string][]int32     // name -> shape (all peers must match)
	result    map[string]*pb.Tensor  // computed after all peers submit
	done      bool
}

// newReduceSession creates a new reduce session for the given world size.
func newReduceSession(worldSize int32) *reduceSession {
	rs := &reduceSession{
		worldSize: worldSize,
		tensors:   make(map[string][][]float32),
		shapes:    make(map[string][]int32),
	}
	rs.cond = sync.NewCond(&rs.mu)
	return rs
}

// Submit adds a peer's tensors to the session. Each peer should call this once.
func (rs *reduceSession) Submit(_ int32, tensors map[string]*pb.Tensor) {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	for name, t := range tensors {
		if t == nil {
			continue
		}
		rs.tensors[name] = append(rs.tensors[name], t.Data)
		if _, ok := rs.shapes[name]; !ok {
			rs.shapes[name] = t.Shape
		}
	}
	rs.submitted++

	if rs.submitted >= rs.worldSize {
		rs.computeResult()
		rs.done = true
		rs.cond.Broadcast()
	}
}

// WaitForResult blocks until all peers have submitted and the result is ready.
// Returns nil if the context is canceled before the result is available.
func (rs *reduceSession) WaitForResult(ctx context.Context) map[string]*pb.Tensor {
	// Use a goroutine to cancel the wait if context expires.
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			rs.cond.Broadcast() // wake up waiters so they can check context
		case <-done:
		}
	}()
	defer close(done)

	rs.mu.Lock()
	defer rs.mu.Unlock()
	for !rs.done {
		if ctx.Err() != nil {
			return nil
		}
		rs.cond.Wait()
	}
	return rs.result
}

// computeResult computes the element-wise average of all submitted tensors.
// Must be called with rs.mu held.
func (rs *reduceSession) computeResult() {
	rs.result = make(map[string]*pb.Tensor, len(rs.tensors))
	n := float32(rs.worldSize)

	for name, allData := range rs.tensors {
		if len(allData) == 0 {
			continue
		}
		size := len(allData[0])
		avg := make([]float32, size)
		for _, data := range allData {
			for i := range avg {
				if i < len(data) {
					avg[i] += data[i]
				}
			}
		}
		for i := range avg {
			avg[i] /= n
		}
		rs.result[name] = &pb.Tensor{
			Shape: rs.shapes[name],
			Data:  avg,
		}
	}
}

// --- barrierState ---

// barrierState tracks barrier arrivals across workers.
type barrierState struct {
	mu        sync.Mutex
	cond      *sync.Cond
	worldSize int32
	arrived   int32
	epoch     int64
}

// newBarrierState creates a new barrierState for the given world size.
func newBarrierState(worldSize int32) *barrierState {
	bs := &barrierState{worldSize: worldSize}
	bs.cond = sync.NewCond(&bs.mu)
	return bs
}

// arrive increments the arrival count. When all workers have arrived,
// it resets the state and advances the epoch. Blocks the caller until
// all workers arrive or the context expires.
func (bs *barrierState) arrive(ctx context.Context) error {
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			bs.cond.Broadcast()
		case <-done:
		}
	}()
	defer close(done)

	bs.mu.Lock()
	defer bs.mu.Unlock()

	currentEpoch := bs.epoch
	bs.arrived++

	if bs.arrived >= bs.worldSize {
		bs.arrived = 0
		bs.epoch++
		bs.cond.Broadcast()
		return nil
	}

	for bs.epoch == currentEpoch {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		bs.cond.Wait()
	}
	return ctx.Err()
}

// --- RPC Handlers ---

// Default histogram buckets for distributed service operations.
var svcOpDurationBuckets = []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}

// recordOp records an operation metric.
func (ws *workerService) recordOp(name string, start time.Time) {
	ws.collector.Counter(name + "_count").Inc()
	ws.collector.Histogram(name+"_duration_seconds", svcOpDurationBuckets).Observe(time.Since(start).Seconds())
}

// AllReduce handles a bidi streaming all-reduce from a peer.
// Each peer sends its tensors as AllReduceRequest messages, then the server
// waits for all peers to submit, computes the average, and sends back
// AllReduceResponse messages with the result.
func (ws *workerService) AllReduce(stream pb.DistributedService_AllReduceServer) error {
	defer ws.recordOp("allreduce_server", time.Now())

	session := ws.getSession()
	if session == nil {
		return status.Error(codes.FailedPrecondition, "no active reduce session")
	}

	// Receive all tensors from this peer until EOF.
	tensors := make(map[string]*pb.Tensor)
	for {
		req, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			ws.logger.Error("allreduce recv error", "error", err.Error())
			return status.Errorf(codes.Internal, "recv error: %v", err)
		}
		tensors[req.Name] = req.Tensor
	}

	// Submit this peer's tensors and wait for the global result.
	session.Submit(-1, tensors) // rank -1 for incoming peers; rank is not used in Submit
	result := session.WaitForResult(stream.Context())
	if result == nil {
		return status.Error(codes.DeadlineExceeded, "allreduce timed out waiting for all peers")
	}

	// Send reduced tensors back to this peer.
	for name, t := range result {
		if err := stream.Send(&pb.AllReduceResponse{Name: name, Tensor: t}); err != nil {
			ws.logger.Error("allreduce send error", "error", err.Error())
			return status.Errorf(codes.Internal, "send error: %v", err)
		}
	}
	return nil
}

// Barrier handles a barrier synchronization request from a peer.
// Blocks until all workers have called Barrier or the context expires.
func (ws *workerService) Barrier(ctx context.Context, req *pb.BarrierRequest) (*pb.BarrierResponse, error) {
	defer ws.recordOp("barrier_server", time.Now())

	if req.Rank < 0 || req.Rank >= ws.worldSize {
		return nil, status.Errorf(codes.InvalidArgument, "rank %d out of range [0, %d)", req.Rank, ws.worldSize)
	}

	if err := ws.barrier.arrive(ctx); err != nil {
		return nil, status.Errorf(codes.DeadlineExceeded, "barrier timed out: %v", err)
	}
	return &pb.BarrierResponse{}, nil
}

// Broadcast handles a broadcast request.
// Root sets the tensor via SetBroadcastTensor before non-root workers call this.
// Non-root workers block until the tensor is available or the context expires.
func (ws *workerService) Broadcast(ctx context.Context, req *pb.BroadcastRequest) (*pb.BroadcastResponse, error) {
	defer ws.recordOp("broadcast_server", time.Now())

	if req.Name == "" {
		return nil, status.Error(codes.InvalidArgument, "broadcast name cannot be empty")
	}

	entry := ws.getBroadcastEntry(req.Name)

	// Wait for the tensor to be available.
	select {
	case <-entry.ready:
		return &pb.BroadcastResponse{Tensor: entry.tensor}, nil
	case <-ctx.Done():
		return nil, status.Errorf(codes.DeadlineExceeded, "broadcast timed out waiting for tensor %q: %v", req.Name, ctx.Err())
	}
}

// validateTensor checks that a pb.Tensor is valid.
func validateTensor(t *pb.Tensor, fieldName string) error {
	if t == nil {
		return fmt.Errorf("%s: tensor cannot be nil", fieldName)
	}
	if len(t.Shape) == 0 {
		return fmt.Errorf("%s: tensor shape cannot be empty", fieldName)
	}
	product := 1
	for _, dim := range t.Shape {
		if dim <= 0 {
			return fmt.Errorf("%s: tensor shape dimension must be positive, got %d", fieldName, dim)
		}
		product *= int(dim)
	}
	if product != len(t.Data) {
		return fmt.Errorf("%s: tensor shape product %d does not match data length %d", fieldName, product, len(t.Data))
	}
	return nil
}

// Static interface assertion.
var _ pb.DistributedServiceServer = (*workerService)(nil)
