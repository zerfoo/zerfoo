package distributed

import (
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestAllReduceStrategy_Init(t *testing.T) {
	customMockLocal := new(testutils.CustomMockStrategy[float32])
	customMockCross := new(testutils.CustomMockStrategy[float32])
	t.Run("successful init for leader", func(t *testing.T) {
		customMockLocal.OnInit(0, 4, "coord").ReturnInit(nil).OnceInit()
		customMockLocal.OnRank().ReturnRank(0).OnceRank()
		customMockLocal.OnSize().ReturnSize(4).OnceSize()
		customMockCross.OnInit(0, 4, "coord").ReturnInit(nil).OnceInit()

		strategy := NewAllReduceStrategy[float32](customMockLocal, customMockCross)
		err := strategy.Init(0, 4, "coord")
		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}
		customMockLocal.AssertExpectations(t)
		customMockCross.AssertExpectations(t)
	})

	t.Run("successful init for non-leader", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])

		customMockLocal.OnInit(1, 4, "coord").ReturnInit(nil).OnceInit()
		customMockLocal.OnRank().ReturnRank(1).OnceRank()
		customMockLocal.OnSize().ReturnSize(4).OnceSize()

		strategy := NewAllReduceStrategy[float32](customMockLocal, customMockCross)
		err := strategy.Init(1, 4, "coord")
		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}
		customMockLocal.AssertExpectations(t)
		customMockCross.AssertNotCalled(t, "Init")
	})

	t.Run("local init fails", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])

		customMockLocal.OnInit(0, 4, "coord").ReturnInit(errors.New("local error")).OnceInit()

		strategy := NewAllReduceStrategy[float32](customMockLocal, customMockCross)
		err := strategy.Init(0, 4, "coord")

		if err == nil {
			t.Errorf("expected an error, got nil")
		}
		customMockLocal.AssertExpectations(t)
	})

	t.Run("cross-node init fails", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])

		customMockLocal.OnInit(0, 4, "coord").ReturnInit(nil).OnceInit()
		customMockLocal.OnRank().ReturnRank(0).OnceRank()
		customMockLocal.OnSize().ReturnSize(4).OnceSize()
		customMockCross.OnInit(0, 4, "coord").ReturnInit(errors.New("cross-node error")).OnceInit()

		strategy := NewAllReduceStrategy[float32](customMockLocal, customMockCross)
		err := strategy.Init(0, 4, "coord")

		if err == nil {
			t.Errorf("expected an error, got nil")
		}
		customMockLocal.AssertExpectations(t)
		customMockCross.AssertExpectations(t)
	})
}

func TestAllReduceStrategy_AllReduceGradients(t *testing.T) {
	grad, _ := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	gradients := map[string]*tensor.Tensor[float32]{"param1": grad}

	t.Run("leader performs all steps", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     customMockLocal,
			crossNodeStrategy: customMockCross,
			isNodeLeader:      true,
		}

		customMockLocal.OnAllReduceGradients(gradients).ReturnAllReduceGradients(nil).OnceAllReduceGradients()
		customMockCross.OnAllReduceGradients(gradients).ReturnAllReduceGradients(nil).OnceAllReduceGradients()
		customMockLocal.OnBroadcastTensor(nil, 0).ReturnBroadcastTensor(nil).OnceBroadcastTensor()

		err := strategy.AllReduceGradients(gradients)
		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}
		customMockLocal.AssertExpectations(t)
		customMockCross.AssertExpectations(t)
	})

	t.Run("non-leader performs only local all-reduce and broadcast", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     customMockLocal,
			crossNodeStrategy: customMockCross,
			isNodeLeader:      false,
		}

		customMockLocal.OnAllReduceGradients(gradients).ReturnAllReduceGradients(nil).OnceAllReduceGradients()
		customMockLocal.OnBroadcastTensor(nil, 0).ReturnBroadcastTensor(nil).OnceBroadcastTensor()

		err := strategy.AllReduceGradients(gradients)
		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}
		customMockLocal.AssertExpectations(t)
		customMockCross.AssertNotCalled(t, "AllReduceGradients")
	})
}

func TestAllReduceStrategy_Barrier(t *testing.T) {
	t.Run("leader performs all barriers", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     customMockLocal,
			crossNodeStrategy: customMockCross,
			isNodeLeader:      true,
		}

		customMockLocal.OnBarrier().ReturnBarrier(nil)
		customMockLocal.OnBarrier().ReturnBarrier(nil)
		customMockCross.OnBarrier().ReturnBarrier(nil)

		err := strategy.Barrier()
		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}
		customMockLocal.AssertExpectations(t)
		customMockCross.AssertExpectations(t)
	})

	t.Run("non-leader performs only local barrier", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     customMockLocal,
			crossNodeStrategy: customMockCross,
			isNodeLeader:      false,
		}

		customMockLocal.OnBarrier().ReturnBarrier(nil)
		customMockLocal.OnBarrier().ReturnBarrier(nil)

		err := strategy.Barrier()
		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}
		customMockLocal.AssertExpectations(t)
		customMockCross.AssertNotCalled(t, "Barrier")
	})
}

func TestAllReduceStrategy_AllReduceGradients_Error(t *testing.T) {
	grad, _ := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	gradients := map[string]*tensor.Tensor[float32]{"param1": grad}

	t.Run("local all-reduce fails", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     customMockLocal,
			crossNodeStrategy: customMockCross,
		}
		customMockLocal.OnAllReduceGradients(gradients).ReturnAllReduceGradients(errors.New("local error")).OnceAllReduceGradients()

		err := strategy.AllReduceGradients(gradients)
		if err == nil {
			t.Errorf("expected an error, got nil")
		}
		customMockLocal.AssertExpectations(t)
	})

	t.Run("cross-node all-reduce fails", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     customMockLocal,
			crossNodeStrategy: customMockCross,
			isNodeLeader:      true,
		}
		customMockLocal.OnAllReduceGradients(gradients).ReturnAllReduceGradients(nil).OnceAllReduceGradients()
		customMockCross.OnAllReduceGradients(gradients).ReturnAllReduceGradients(errors.New("cross-node error")).OnceAllReduceGradients()

		err := strategy.AllReduceGradients(gradients)
		if err == nil {
			t.Errorf("expected an error, got nil")
		}
		customMockLocal.AssertExpectations(t)
		customMockCross.AssertExpectations(t)
	})

	t.Run("broadcast fails", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     customMockLocal,
			crossNodeStrategy: customMockCross,
		}
		customMockLocal.OnAllReduceGradients(gradients).ReturnAllReduceGradients(nil).OnceAllReduceGradients()
		customMockLocal.OnBroadcastTensor(nil, 0).ReturnBroadcastTensor(errors.New("broadcast error")).OnceBroadcastTensor()

		err := strategy.AllReduceGradients(gradients)
		if err == nil {
			t.Errorf("expected an error, got nil")
		}
		customMockLocal.AssertExpectations(t)
	})
}

func TestAllReduceStrategy_Barrier_Error(t *testing.T) {
	t.Run("local barrier fails", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     customMockLocal,
			crossNodeStrategy: customMockCross,
		}
		customMockLocal.OnBarrier().ReturnBarrier(errors.New("local barrier error"))

		err := strategy.Barrier()
		if err == nil {
			t.Errorf("expected an error, got nil")
		}
		customMockLocal.AssertExpectations(t)
	})

	t.Run("cross-node barrier fails", func(t *testing.T) {
		customMockLocal := new(testutils.CustomMockStrategy[float32])
		customMockCross := new(testutils.CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     customMockLocal,
			crossNodeStrategy: customMockCross,
			isNodeLeader:      true,
		}
		customMockLocal.OnBarrier().ReturnBarrier(nil)
		customMockCross.OnBarrier().ReturnBarrier(errors.New("cross-node barrier error"))

		err := strategy.Barrier()
		if err == nil {
			t.Errorf("expected an error, got nil")
		}
		customMockLocal.AssertExpectations(t)
		customMockCross.AssertExpectations(t)
	})
}

func TestAllReduceStrategy_Rank(t *testing.T) {
	customMockLocal := new(testutils.CustomMockStrategy[float32])
	strategy := &AllReduceStrategy[float32]{
		localStrategy: customMockLocal,
	}
	customMockLocal.OnRank().ReturnRank(5)
	testutils.AssertEqual(t, 5, strategy.Rank(), "expected rank to be 5")
}

func TestAllReduceStrategy_Size(t *testing.T) {
	customMockLocal := new(testutils.CustomMockStrategy[float32])
	strategy := &AllReduceStrategy[float32]{
		localStrategy: customMockLocal,
	}
	customMockLocal.OnSize().ReturnSize(10)
	testutils.AssertEqual(t, 10, strategy.Size(), "expected size to be 10")
}
