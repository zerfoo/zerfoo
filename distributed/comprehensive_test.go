package distributed

import (
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

// ---------- BroadcastTensor tests ----------

func TestAllReduceStrategy_BroadcastTensor(t *testing.T) {
	tt, _ := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})

	t.Run("leader_broadcasts_cross_node_and_local", func(t *testing.T) {
		local := new(CustomMockStrategy[float32])
		cross := new(CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     local,
			crossNodeStrategy: cross,
			isNodeLeader:      true,
		}

		local.OnSize().ReturnSize(4)
		cross.OnBroadcastTensor(tt, 0).ReturnBroadcastTensor(nil).OnceBroadcastTensor()
		local.OnBroadcastTensor(tt, 0).ReturnBroadcastTensor(nil).OnceBroadcastTensor()

		err := strategy.BroadcastTensor(tt, 0)
		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}
	})

	t.Run("non_leader_broadcasts_local_only", func(t *testing.T) {
		local := new(CustomMockStrategy[float32])
		cross := new(CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     local,
			crossNodeStrategy: cross,
			isNodeLeader:      false,
		}

		local.OnSize().ReturnSize(4)
		local.OnBroadcastTensor(tt, 0).ReturnBroadcastTensor(nil).OnceBroadcastTensor()

		err := strategy.BroadcastTensor(tt, 0)
		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}
		cross.AssertNotCalled(t, "BroadcastTensor")
	})

	t.Run("cross_node_broadcast_error", func(t *testing.T) {
		local := new(CustomMockStrategy[float32])
		cross := new(CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     local,
			crossNodeStrategy: cross,
			isNodeLeader:      true,
		}

		local.OnSize().ReturnSize(4)
		cross.OnBroadcastTensor(tt, 0).ReturnBroadcastTensor(errors.New("cross error")).OnceBroadcastTensor()

		err := strategy.BroadcastTensor(tt, 0)
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("local_broadcast_error", func(t *testing.T) {
		local := new(CustomMockStrategy[float32])
		cross := new(CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     local,
			crossNodeStrategy: cross,
			isNodeLeader:      false,
		}

		local.OnSize().ReturnSize(4)
		local.OnBroadcastTensor(tt, 0).ReturnBroadcastTensor(errors.New("local error")).OnceBroadcastTensor()

		err := strategy.BroadcastTensor(tt, 0)
		if err == nil {
			t.Error("expected error")
		}
	})
}

// ---------- Shutdown tests ----------

func TestAllReduceStrategy_Shutdown(t *testing.T) {
	t.Run("leader_shuts_down_both", func(t *testing.T) {
		local := new(CustomMockStrategy[float32])
		cross := new(CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     local,
			crossNodeStrategy: cross,
			isNodeLeader:      true,
		}

		strategy.Shutdown()

		if local.shutdownCalls != 1 {
			t.Errorf("expected local shutdown to be called once, got %d", local.shutdownCalls)
		}
		if cross.shutdownCalls != 1 {
			t.Errorf("expected cross shutdown to be called once, got %d", cross.shutdownCalls)
		}
	})

	t.Run("non_leader_shuts_down_local_only", func(t *testing.T) {
		local := new(CustomMockStrategy[float32])
		cross := new(CustomMockStrategy[float32])
		strategy := &AllReduceStrategy[float32]{
			localStrategy:     local,
			crossNodeStrategy: cross,
			isNodeLeader:      false,
		}

		strategy.Shutdown()

		if local.shutdownCalls != 1 {
			t.Errorf("expected local shutdown to be called once, got %d", local.shutdownCalls)
		}
		cross.AssertNotCalled(t, "Shutdown")
	})
}

// ---------- Barrier: post-cross-node local barrier failure ----------

func TestAllReduceStrategy_Barrier_PostCrossNodeLocalFail(t *testing.T) {
	local := new(CustomMockStrategy[float32])
	cross := new(CustomMockStrategy[float32])
	strategy := &AllReduceStrategy[float32]{
		localStrategy:     local,
		crossNodeStrategy: cross,
		isNodeLeader:      true,
	}

	// First local barrier succeeds, cross barrier succeeds, second local barrier fails
	local.OnBarrier().ReturnBarrier(nil)
	cross.OnBarrier().ReturnBarrier(nil)
	local.OnBarrier().ReturnBarrier(errors.New("post-cross-node error"))

	err := strategy.Barrier()
	if err == nil {
		t.Error("expected error from post-cross-node local barrier")
	}
}

// ---------- ServerManager SetLogger, Stop, GracefulStop ----------

func TestServerManager_SetLogger(t *testing.T) {
	srv := &mockGrpcServer{}
	sm := NewServerManager(srv, nil)
	sm.SetLogger(&CustomMockLogger{})
	// Verify no panic and the logger was set.

	// nil logger should default to Nop.
	sm.SetLogger(nil)
}

func TestServerManager_Stop(t *testing.T) {
	t.Run("with_server", func(t *testing.T) {
		srv := &mockGrpcServer{}
		sm := NewServerManager(srv, nil)
		sm.Stop()
		if !srv.stopCalled {
			t.Error("expected Stop to be called on server")
		}
	})

	t.Run("nil_server", func(t *testing.T) {
		sm := &serverManager{server: nil}
		sm.Stop() // should not panic
	})
}

func TestServerManager_GracefulStop(t *testing.T) {
	t.Run("with_server", func(t *testing.T) {
		srv := &mockGrpcServer{}
		sm := NewServerManager(srv, nil)
		sm.GracefulStop()
		if !srv.gracefulStop {
			t.Error("expected GracefulStop to be called on server")
		}
	})

	t.Run("nil_server", func(t *testing.T) {
		sm := &serverManager{server: nil}
		sm.GracefulStop() // should not panic
	})
}


// ---------- NewNetworkManager default dialer/factory coverage ----------

func TestNewNetworkManager_DefaultsInvoked(t *testing.T) {
	// Create with nil dialer and nil clientFactory to exercise the default lambda bodies
	nm := NewNetworkManager(nil, nil)
	// ConnectToPeers with a single peer and selfRank=0 skips index 0,
	// so no dial calls happen. Use 2 peers so at least one dial occurs.
	// grpc.NewClient is lazy (doesn't actually connect), so this will succeed.
	clients, conns, err := nm.ConnectToPeers([]string{"localhost:0", "localhost:0"}, 0, 1)
	if err != nil {
		t.Fatalf("ConnectToPeers with defaults failed: %v", err)
	}
	// clients[1] should be non-nil (created by default factory)
	if clients[1] == nil {
		t.Error("expected non-nil client for peer 1")
	}
	nm.CloseConnections(conns)
}
