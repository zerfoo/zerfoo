package distributed

import (
	"context"
	"testing"
)

func TestNewWorkerNode_Defaults(t *testing.T) {
	wn := NewWorkerNode(WorkerNodeConfig{
		WorkerAddress:      "localhost:0",
		CoordinatorAddress: "localhost:0",
	})
	if wn == nil {
		t.Fatal("expected non-nil WorkerNode")
	}
	if wn.logger == nil {
		t.Error("expected non-nil logger")
	}
	if wn.config.Collector == nil {
		t.Error("expected non-nil collector")
	}
}

func TestWorkerNode_RankSizeBeforeStart(t *testing.T) {
	wn := NewWorkerNode(WorkerNodeConfig{
		WorkerAddress: "localhost:0",
	})
	if wn.Rank() != -1 {
		t.Errorf("Rank() = %d, want -1 before start", wn.Rank())
	}
	if wn.Size() != 0 {
		t.Errorf("Size() = %d, want 0 before start", wn.Size())
	}
}

func TestWorkerNode_StrategyBeforeStart(t *testing.T) {
	wn := NewWorkerNode(WorkerNodeConfig{
		WorkerAddress: "localhost:0",
	})
	if wn.Strategy() != nil {
		t.Error("expected nil strategy before start")
	}
}

func TestWorkerNode_CloseBeforeStart(t *testing.T) {
	wn := NewWorkerNode(WorkerNodeConfig{
		WorkerAddress: "localhost:0",
	})
	// Close on an unstarted node should be safe.
	if err := wn.Close(context.Background()); err != nil {
		t.Errorf("Close() error = %v, want nil", err)
	}
}

func TestWorkerNode_DoubleClose(t *testing.T) {
	wn := NewWorkerNode(WorkerNodeConfig{
		WorkerAddress: "localhost:0",
	})
	// Double close should not panic.
	_ = wn.Close(context.Background())
	_ = wn.Close(context.Background())
}

func TestWorkerNode_HealthCheck_NotStarted(t *testing.T) {
	wn := NewWorkerNode(WorkerNodeConfig{
		WorkerAddress: "localhost:0",
	})
	check := wn.healthCheck()
	if err := check(); err == nil {
		t.Error("expected error from health check before start")
	}
}
