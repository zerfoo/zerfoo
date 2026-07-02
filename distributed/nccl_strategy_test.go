//go:build cuda

package distributed

import (
	"sync"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/nccl"
	"github.com/zerfoo/ztensor/tensor"
)

func TestNcclStrategy_NewAndFields(t *testing.T) {
	s := NewNcclStrategy[float32](NcclStrategyConfig{
		DeviceID: 0,
	})
	if s == nil {
		t.Fatal("expected non-nil strategy")
	}
	if s.deviceID != 0 {
		t.Errorf("deviceID = %d, want 0", s.deviceID)
	}
}

func TestNcclStrategy_RankAndSize(t *testing.T) {
	s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: 0})
	if s.Rank() != 0 {
		t.Errorf("Rank() = %d, want 0 (before init)", s.Rank())
	}
	if s.Size() != 0 {
		t.Errorf("Size() = %d, want 0 (before init)", s.Size())
	}
}

func TestNcclStrategy_AllReduceWithoutInit(t *testing.T) {
	s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: 0})
	err := s.AllReduceGradients(nil)
	if err == nil {
		t.Error("expected error when comm is nil")
	}
}

func TestNcclStrategy_BarrierWithoutInit(t *testing.T) {
	s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: 0})
	err := s.Barrier()
	if err == nil {
		t.Error("expected error when comm is nil")
	}
}

func TestNcclStrategy_BroadcastWithoutInit(t *testing.T) {
	s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: 0})
	err := s.BroadcastTensor(nil, 0)
	if err == nil {
		t.Error("expected error when comm is nil")
	}
}

func TestNcclStrategy_ShutdownIdempotent(t *testing.T) {
	s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: 0})
	// Shutdown with no init should not panic.
	s.Shutdown()
	s.Shutdown()
}

func TestNcclStrategy_SingleRankAllReduce(t *testing.T) {
	count, err := cuda.GetDeviceCount()
	if err != nil || count < 1 {
		t.Skip("requires at least 1 CUDA device")
	}

	uid, err := nccl.GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}

	s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: 0})
	if err := s.InitWithUID(0, 1, uid); err != nil {
		t.Fatalf("InitWithUID: %v", err)
	}
	defer s.Shutdown()

	// Create a GPU tensor with [1, 2, 3, 4].
	data := []float32{1, 2, 3, 4}
	gs, err := tensor.NewGPUStorageFromSlice(data, 0)
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice: %v", err)
	}
	grad, err := tensor.NewWithStorage([]int{4}, gs)
	if err != nil {
		t.Fatalf("NewWithStorage: %v", err)
	}

	gradients := map[string]*tensor.TensorNumeric[float32]{
		"layer1.weight": grad,
	}
	if err := s.AllReduceGradients(gradients); err != nil {
		t.Fatalf("AllReduceGradients: %v", err)
	}

	// With 1 rank, sum == original values.
	result := grad.Data()
	for i, want := range data {
		if result[i] != want {
			t.Errorf("result[%d] = %f, want %f", i, result[i], want)
		}
	}
}

func TestNcclStrategy_SingleRankBarrier(t *testing.T) {
	count, err := cuda.GetDeviceCount()
	if err != nil || count < 1 {
		t.Skip("requires at least 1 CUDA device")
	}

	uid, err := nccl.GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}

	s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: 0})
	if err := s.InitWithUID(0, 1, uid); err != nil {
		t.Fatalf("InitWithUID: %v", err)
	}
	defer s.Shutdown()

	if err := s.Barrier(); err != nil {
		t.Fatalf("Barrier: %v", err)
	}
}

func TestNcclStrategy_SingleRankBroadcast(t *testing.T) {
	count, err := cuda.GetDeviceCount()
	if err != nil || count < 1 {
		t.Skip("requires at least 1 CUDA device")
	}

	uid, err := nccl.GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}

	s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: 0})
	if err := s.InitWithUID(0, 1, uid); err != nil {
		t.Fatalf("InitWithUID: %v", err)
	}
	defer s.Shutdown()

	data := []float32{10, 20, 30}
	gs, err := tensor.NewGPUStorageFromSlice(data, 0)
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice: %v", err)
	}
	tt, err := tensor.NewWithStorage([]int{3}, gs)
	if err != nil {
		t.Fatalf("NewWithStorage: %v", err)
	}

	if err := s.BroadcastTensor(tt, 0); err != nil {
		t.Fatalf("BroadcastTensor: %v", err)
	}

	result := tt.Data()
	for i, want := range data {
		if result[i] != want {
			t.Errorf("result[%d] = %f, want %f", i, result[i], want)
		}
	}
}

func TestNcclStrategy_TwoGPUAllReduce(t *testing.T) {
	count, err := cuda.GetDeviceCount()
	if err != nil || count < 2 {
		t.Skip("requires at least 2 CUDA devices")
	}

	uid, err := nccl.GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}

	n := 4
	nRanks := 2
	var wg sync.WaitGroup
	errs := make([]error, nRanks)
	results := make([][]float32, nRanks)

	for rank := range nRanks {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()

			s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: rank})
			if err := s.InitWithUID(rank, nRanks, uid); err != nil {
				errs[rank] = err
				return
			}
			defer s.Shutdown()

			// Rank 0: [1,2,3,4], Rank 1: [10,20,30,40].
			data := make([]float32, n)
			for i := range n {
				data[i] = float32((rank*9 + 1) * (i + 1))
			}
			gs, err := tensor.NewGPUStorageFromSlice(data, rank)
			if err != nil {
				errs[rank] = err
				return
			}
			grad, err := tensor.NewWithStorage([]int{n}, gs)
			if err != nil {
				errs[rank] = err
				return
			}

			gradients := map[string]*tensor.TensorNumeric[float32]{
				"w": grad,
			}
			if err := s.AllReduceGradients(gradients); err != nil {
				errs[rank] = err
				return
			}

			results[rank] = grad.Data()
		}(rank)
	}
	wg.Wait()

	for rank, err := range errs {
		if err != nil {
			t.Fatalf("rank %d error: %v", rank, err)
		}
	}

	// Rank 0 data: [1,2,3,4], Rank 1 data: [10,20,30,40]. Sum: [11,22,33,44].
	for rank := range nRanks {
		for i := range n {
			want := float32(11 * (i + 1))
			if results[rank][i] != want {
				t.Errorf("rank %d result[%d] = %f, want %f", rank, i, results[rank][i], want)
			}
		}
	}
}

func TestNcclStrategy_AllReduceCPUTensorError(t *testing.T) {
	count, err := cuda.GetDeviceCount()
	if err != nil || count < 1 {
		t.Skip("requires at least 1 CUDA device")
	}

	uid, err := nccl.GetUniqueID()
	if err != nil {
		t.Fatalf("GetUniqueID: %v", err)
	}

	s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: 0})
	if err := s.InitWithUID(0, 1, uid); err != nil {
		t.Fatalf("InitWithUID: %v", err)
	}
	defer s.Shutdown()

	// Create a CPU tensor (not GPU).
	cpuTensor, err := tensor.New([]int{4}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	gradients := map[string]*tensor.TensorNumeric[float32]{
		"cpu_grad": cpuTensor,
	}
	err = s.AllReduceGradients(gradients)
	if err == nil {
		t.Error("expected error for CPU tensor in AllReduceGradients")
	}
}

// ncclDataType_test verifies the type mapping helper.
func TestNcclDataType(t *testing.T) {
	if ncclDataType[float32]() != nccl.Float32 {
		t.Error("float32 should map to nccl.Float32")
	}
	if ncclDataType[float64]() != nccl.Float64 {
		t.Error("float64 should map to nccl.Float64")
	}
	if ncclDataType[int32]() != nccl.Int32 {
		t.Error("int32 should map to nccl.Int32")
	}
	if ncclDataType[int64]() != nccl.Int64 {
		t.Error("int64 should map to nccl.Int64")
	}
}

// streamPtrHelper ensures the unexported helper works.
func TestNcclStrategy_StreamPtr(t *testing.T) {
	s := NewNcclStrategy[float32](NcclStrategyConfig{DeviceID: 0})
	if s.streamPtr() != nil {
		t.Error("expected nil stream ptr before init")
	}
}

// Ensure unused import of unsafe is consumed.
var _ = unsafe.Pointer(nil)
