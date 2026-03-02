// Package distributed provides distributed training strategies and coordination mechanisms
// for multi-node machine learning workloads in the Zerfoo framework.
package distributed

import (
	"fmt"
	"time"

	metrics "github.com/zerfoo/zerfoo/metrics/runtime"
	"github.com/zerfoo/zerfoo/tensor"
)

// Default histogram buckets for distributed operation duration (seconds).
var distOpDurationBuckets = []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}

// AllReduceStrategy implements a more advanced AllReduce algorithm.
type AllReduceStrategy[T tensor.Numeric] struct {
	localStrategy     InternalStrategy[T]
	crossNodeStrategy InternalStrategy[T]
	localRank         int
	isNodeLeader      bool
	collector         metrics.Collector
}

// NewAllReduceStrategy creates a new AllReduceStrategy.
func NewAllReduceStrategy[T tensor.Numeric](
	localStrategy, crossNodeStrategy InternalStrategy[T],
) *AllReduceStrategy[T] {
	return &AllReduceStrategy[T]{
		localStrategy:     localStrategy,
		crossNodeStrategy: crossNodeStrategy,
		collector:         metrics.Nop(),
	}
}

// SetCollector replaces the strategy's metrics collector.
func (s *AllReduceStrategy[T]) SetCollector(c metrics.Collector) {
	if c == nil {
		c = metrics.Nop()
	}
	s.collector = c
}

// recordOp increments the operation counter and records the duration.
func (s *AllReduceStrategy[T]) recordOp(name string, start time.Time) {
	c := s.collector
	if c == nil {
		return
	}
	c.Counter(name + "_count").Inc()
	c.Histogram(name+"_duration_seconds", distOpDurationBuckets).Observe(time.Since(start).Seconds())
}

// Init initializes the hierarchical strategy.
func (s *AllReduceStrategy[T]) Init(rank, size int, coordinatorAddress string) error {
	if err := s.localStrategy.Init(rank, size, coordinatorAddress); err != nil {
		return fmt.Errorf("failed to initialize local strategy: %w", err)
	}

	s.localRank = s.localStrategy.Rank() % s.localStrategy.Size()
	s.isNodeLeader = s.localRank == 0

	if s.isNodeLeader {
		if err := s.crossNodeStrategy.Init(rank, size, coordinatorAddress); err != nil {
			return fmt.Errorf("failed to initialize cross-node strategy: %w", err)
		}
	}

	return nil
}

// AllReduceGradients performs hierarchical all-reduce on gradients.
func (s *AllReduceStrategy[T]) AllReduceGradients(gradients map[string]*tensor.TensorNumeric[T]) error {
	defer s.recordOp("allreduce", time.Now())
	// Step 1: Local AllReduce within the node.
	if err := s.localStrategy.AllReduceGradients(gradients); err != nil {
		return fmt.Errorf("local AllReduce failed: %w", err)
	}

	// Step 2: Cross-node AllReduce among node leaders.
	if s.isNodeLeader {
		if err := s.crossNodeStrategy.AllReduceGradients(gradients); err != nil {
			return fmt.Errorf("cross-node AllReduce failed: %w", err)
		}
	}

	// Step 3: Broadcast from node leaders to their local groups.
	for name, grad := range gradients {
		if err := s.localStrategy.BroadcastTensor(grad, 0); err != nil {
			return fmt.Errorf("broadcast of %s failed: %w", name, err)
		}
	}

	return nil
}

// Rank returns the rank from the local strategy.
func (s *AllReduceStrategy[T]) Rank() int {
	return s.localStrategy.Rank()
}

// Size returns the size from the local strategy.
func (s *AllReduceStrategy[T]) Size() int {
	return s.localStrategy.Size()
}

// Barrier synchronizes all workers across all nodes.
func (s *AllReduceStrategy[T]) Barrier() error {
	defer s.recordOp("barrier", time.Now())
	// First, synchronize within the local node.
	if err := s.localStrategy.Barrier(); err != nil {
		return fmt.Errorf("local barrier failed: %w", err)
	}
	// Then, synchronize across nodes (only leaders participate).
	if s.isNodeLeader {
		if err := s.crossNodeStrategy.Barrier(); err != nil {
			return fmt.Errorf("cross-node barrier failed: %w", err)
		}
	}
	// Finally, another local barrier to ensure all workers wait for the cross-node barrier to complete.
	if err := s.localStrategy.Barrier(); err != nil {
		return fmt.Errorf("post-cross-node local barrier failed: %w", err)
	}

	return nil
}

// BroadcastTensor broadcasts a tensor from the root rank to all other ranks in the distributed system.
// The tensor is first broadcast within the root's local node, then across node leaders, and finally
// within each local node to ensure all ranks receive the broadcasted tensor.
func (s *AllReduceStrategy[T]) BroadcastTensor(t *tensor.TensorNumeric[T], rootRank int) error {
	defer s.recordOp("broadcast", time.Now())
	// Determine the node leader of the root rank.
	rootNodeLeaderRank := rootRank - (rootRank % s.localStrategy.Size())

	// If the current worker is a node leader, it participates in the cross-node broadcast.
	if s.isNodeLeader {
		if err := s.crossNodeStrategy.BroadcastTensor(t, rootNodeLeaderRank); err != nil {
			return fmt.Errorf("cross-node broadcast failed: %w", err)
		}
	}

	// All workers in a node participate in the local broadcast.
	if err := s.localStrategy.BroadcastTensor(t, 0); err != nil {
		return fmt.Errorf("local broadcast failed: %w", err)
	}

	return nil
}

// Shutdown gracefully closes all connections.
func (s *AllReduceStrategy[T]) Shutdown() {
	s.localStrategy.Shutdown()

	if s.isNodeLeader {
		s.crossNodeStrategy.Shutdown()
	}
}

// Statically assert that the type implements the interface.
var _ InternalStrategy[float32] = (*AllReduceStrategy[float32])(nil)
