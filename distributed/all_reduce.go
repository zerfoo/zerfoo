// Package distributed provides distributed training strategies and coordination mechanisms
// for multi-node machine learning workloads in the Zerfoo framework.
package distributed

import (
	"fmt"

	"github.com/zerfoo/zerfoo/tensor"
)

// AllReduceStrategy implements a more advanced AllReduce algorithm.
type AllReduceStrategy[T tensor.Numeric] struct {
	localStrategy     InternalStrategy[T]
	crossNodeStrategy InternalStrategy[T]
	localRank         int
	isNodeLeader      bool
}

// NewAllReduceStrategy creates a new AllReduceStrategy.
func NewAllReduceStrategy[T tensor.Numeric](
	localStrategy, crossNodeStrategy InternalStrategy[T],
) *AllReduceStrategy[T] {
	return &AllReduceStrategy[T]{
		localStrategy:     localStrategy,
		crossNodeStrategy: crossNodeStrategy,
	}
}

// Init initializes the hierarchical strategy.
func (s *AllReduceStrategy[T]) Init(rank int, size int, coordinatorAddress string) error {
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
func (s *AllReduceStrategy[T]) AllReduceGradients(gradients map[string]*tensor.Tensor[T]) error {
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
func (s *AllReduceStrategy[T]) BroadcastTensor(t *tensor.Tensor[T], rootRank int) error {
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
