package distributed

import (
	"context"
	"sync"

	"github.com/zerfoo/zerfoo/distributed/pb"
	"github.com/zerfoo/zerfoo/log"
	metrics "github.com/zerfoo/zerfoo/metrics/runtime"
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

// Static interface assertion.
var _ pb.DistributedServiceServer = (*workerService)(nil)
