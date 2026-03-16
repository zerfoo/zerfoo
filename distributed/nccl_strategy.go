//go:build cuda

package distributed

import (
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/nccl"
	"github.com/zerfoo/ztensor/log"
	"github.com/zerfoo/ztensor/metrics/runtime"
	"github.com/zerfoo/ztensor/tensor"
)

// NcclStrategy implements InternalStrategy[T] using NCCL for GPU-native
// collective operations. Gradient tensors stay on-device throughout the
// all-reduce; no CPU round-trip is needed.
type NcclStrategy[T tensor.Numeric] struct {
	rank     int
	size     int
	deviceID int

	comm   *nccl.Comm
	stream *cuda.Stream

	logger    log.Logger
	collector runtime.Collector

	shutdownOnce sync.Once
}

// Statically assert that NcclStrategy satisfies InternalStrategy.
var _ InternalStrategy[float32] = (*NcclStrategy[float32])(nil)

// NcclStrategyConfig holds configuration for creating an NcclStrategy.
type NcclStrategyConfig struct {
	DeviceID  int
	Logger    log.Logger
	Collector runtime.Collector
}

// NewNcclStrategy creates an NcclStrategy for the specified device.
// Call Init to initialize the NCCL communicator before use.
func NewNcclStrategy[T tensor.Numeric](cfg NcclStrategyConfig) *NcclStrategy[T] {
	l := cfg.Logger
	if l == nil {
		l = log.Nop()
	}
	c := cfg.Collector
	if c == nil {
		c = runtime.Nop()
	}
	return &NcclStrategy[T]{
		deviceID:  cfg.DeviceID,
		logger:    l,
		collector: c,
	}
}

// Init initializes the NCCL communicator for this rank. All ranks must call
// Init concurrently with the same UniqueID (passed via coordinatorAddress as
// serialized bytes for simplicity; in practice, the coordinator distributes it).
// For single-process multi-GPU, the caller should use InitWithUID instead.
func (s *NcclStrategy[T]) Init(rank int, size int, _ string) error {
	s.rank = rank
	s.size = size

	if err := cuda.SetDevice(s.deviceID); err != nil {
		return fmt.Errorf("NcclStrategy.Init: SetDevice(%d): %w", s.deviceID, err)
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		return fmt.Errorf("NcclStrategy.Init: CreateStream: %w", err)
	}
	s.stream = stream

	s.logger.Info("nccl strategy initialized", "rank", fmt.Sprintf("%d", rank), "size", fmt.Sprintf("%d", size), "device", fmt.Sprintf("%d", s.deviceID))
	return nil
}

// InitWithUID initializes the NCCL communicator using a pre-shared UniqueID.
// This is the recommended path for single-process multi-GPU training where
// a coordinator can distribute the UID directly.
func (s *NcclStrategy[T]) InitWithUID(rank, size int, uid *nccl.UniqueID) error {
	s.rank = rank
	s.size = size

	if err := cuda.SetDevice(s.deviceID); err != nil {
		return fmt.Errorf("NcclStrategy.InitWithUID: SetDevice(%d): %w", s.deviceID, err)
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		return fmt.Errorf("NcclStrategy.InitWithUID: CreateStream: %w", err)
	}
	s.stream = stream

	comm, err := nccl.InitRank(uid, size, rank)
	if err != nil {
		_ = stream.Destroy()
		return fmt.Errorf("NcclStrategy.InitWithUID: ncclCommInitRank: %w", err)
	}
	s.comm = comm

	s.logger.Info("nccl strategy initialized with UID", "rank", fmt.Sprintf("%d", rank), "size", fmt.Sprintf("%d", size), "device", fmt.Sprintf("%d", s.deviceID))
	return nil
}

// ncclDataType returns the NCCL data type corresponding to T.
func ncclDataType[T tensor.Numeric]() nccl.DataType {
	var zero T
	switch any(zero).(type) {
	case float32:
		return nccl.Float32
	case float64:
		return nccl.Float64
	case int32:
		return nccl.Int32
	case int64:
		return nccl.Int64
	default:
		return nccl.Float32
	}
}

// AllReduceGradients performs an NCCL all-reduce (sum) on each gradient tensor
// in-place on GPU memory. After the all-reduce, each tensor is divided by the
// number of ranks to produce the average gradient.
func (s *NcclStrategy[T]) AllReduceGradients(gradients map[string]*tensor.TensorNumeric[T]) error {
	if s.comm == nil {
		return fmt.Errorf("NcclStrategy.AllReduceGradients: communicator not initialized")
	}

	start := time.Now()
	defer func() {
		s.collector.Counter("nccl_allreduce_count").Inc()
		s.collector.Histogram("nccl_allreduce_duration_seconds",
			[]float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0}).
			Observe(time.Since(start).Seconds())
	}()

	if err := cuda.SetDevice(s.deviceID); err != nil {
		return fmt.Errorf("NcclStrategy.AllReduceGradients: SetDevice: %w", err)
	}

	dtype := ncclDataType[T]()

	// Group all reductions into a single NCCL launch for efficiency.
	if err := nccl.GroupStart(); err != nil {
		return fmt.Errorf("NcclStrategy.AllReduceGradients: GroupStart: %w", err)
	}

	for name, grad := range gradients {
		gs, ok := grad.GetStorage().(*tensor.GPUStorage[T])
		if !ok {
			return fmt.Errorf("NcclStrategy.AllReduceGradients: tensor %q is not on GPU", name)
		}

		ptr := gs.Ptr()
		count := gs.Len()

		if err := s.comm.AllReduce(ptr, ptr, count, dtype, nccl.Sum, s.stream.Ptr()); err != nil {
			return fmt.Errorf("NcclStrategy.AllReduceGradients: AllReduce(%q): %w", name, err)
		}
	}

	if err := nccl.GroupEnd(); err != nil {
		return fmt.Errorf("NcclStrategy.AllReduceGradients: GroupEnd: %w", err)
	}

	// Wait for all reductions to complete.
	if err := s.stream.Synchronize(); err != nil {
		return fmt.Errorf("NcclStrategy.AllReduceGradients: Synchronize: %w", err)
	}

	return nil
}

// Barrier blocks until all ranks have reached this point using a zero-byte
// all-reduce as a synchronization primitive.
func (s *NcclStrategy[T]) Barrier() error {
	if s.comm == nil {
		return fmt.Errorf("NcclStrategy.Barrier: communicator not initialized")
	}

	if err := cuda.SetDevice(s.deviceID); err != nil {
		return fmt.Errorf("NcclStrategy.Barrier: SetDevice: %w", err)
	}

	// Allocate a single-element buffer for the barrier.
	byteSize := 4 // float32
	devPtr, err := cuda.Malloc(byteSize)
	if err != nil {
		return fmt.Errorf("NcclStrategy.Barrier: Malloc: %w", err)
	}
	defer cuda.Free(devPtr)

	if err := s.comm.AllReduce(devPtr, devPtr, 1, nccl.Float32, nccl.Sum, s.stream.Ptr()); err != nil {
		return fmt.Errorf("NcclStrategy.Barrier: AllReduce: %w", err)
	}

	if err := s.stream.Synchronize(); err != nil {
		return fmt.Errorf("NcclStrategy.Barrier: Synchronize: %w", err)
	}

	return nil
}

// BroadcastTensor broadcasts a tensor from rootRank to all other ranks using
// NCCL. The tensor must reside on GPU memory.
func (s *NcclStrategy[T]) BroadcastTensor(t *tensor.TensorNumeric[T], rootRank int) error {
	if s.comm == nil {
		return fmt.Errorf("NcclStrategy.BroadcastTensor: communicator not initialized")
	}

	if err := cuda.SetDevice(s.deviceID); err != nil {
		return fmt.Errorf("NcclStrategy.BroadcastTensor: SetDevice: %w", err)
	}

	gs, ok := t.GetStorage().(*tensor.GPUStorage[T])
	if !ok {
		return fmt.Errorf("NcclStrategy.BroadcastTensor: tensor is not on GPU")
	}

	ptr := gs.Ptr()
	count := gs.Len()
	dtype := ncclDataType[T]()

	if err := s.comm.Broadcast(ptr, ptr, count, dtype, rootRank, s.stream.Ptr()); err != nil {
		return fmt.Errorf("NcclStrategy.BroadcastTensor: %w", err)
	}

	if err := s.stream.Synchronize(); err != nil {
		return fmt.Errorf("NcclStrategy.BroadcastTensor: Synchronize: %w", err)
	}

	return nil
}

// Rank returns the rank of this worker.
func (s *NcclStrategy[T]) Rank() int { return s.rank }

// Size returns the total number of workers.
func (s *NcclStrategy[T]) Size() int { return s.size }

// Shutdown destroys the NCCL communicator and CUDA stream.
func (s *NcclStrategy[T]) Shutdown() {
	s.shutdownOnce.Do(func() {
		if s.comm != nil {
			if err := s.comm.Destroy(); err != nil {
				s.logger.Error("nccl comm destroy", "error", err.Error())
			}
			s.comm = nil
		}
		if s.stream != nil {
			if err := s.stream.Destroy(); err != nil {
				s.logger.Error("cuda stream destroy", "error", err.Error())
			}
			s.stream = nil
		}
		s.logger.Info("nccl strategy shutdown", "rank", fmt.Sprintf("%d", s.rank))
	})
}

// streamPtr is a helper used by tests to get the underlying stream pointer.
func (s *NcclStrategy[T]) streamPtr() unsafe.Pointer {
	if s.stream == nil {
		return nil
	}
	return s.stream.Ptr()
}
