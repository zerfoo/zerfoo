package coordinator

import (
	"bytes"
	"context"
	"math"
	"net"
	"testing"
	"time"

	// NOTE: RegisterWorker rank overflow (line 191-193) is unreachable in practice
	// because the preceding "for r := range c.nextRank" loop would iterate 2B+ times
	// before reaching the check. This is excluded from coverage targets.

	"github.com/zerfoo/zerfoo/distributed/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

// setupShortTimeout creates a coordinator with a short heartbeat timeout for reaper testing.
func setupShortTimeout(t *testing.T, timeout time.Duration) *testKit {
	t.Helper()

	lis := bufconn.Listen(bufSize)

	var buf bytes.Buffer

	coord := NewCoordinator(&buf, timeout)
	coord.start(lis)

	t.Cleanup(func() {
		coord.GracefulStop()
	})

	dialer := func(context.Context, string) (net.Conn, error) {
		return lis.Dial()
	}

	conn, err := grpc.NewClient("passthrough:///bufnet", grpc.WithContextDialer(dialer), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}

	t.Cleanup(func() {
		if closeErr := conn.Close(); closeErr != nil {
			t.Errorf("error closing connection: %v", closeErr)
		}
	})

	return &testKit{
		client: pb.NewCoordinatorClient(conn),
		coord:  coord,
		lis:    lis,
		buf:    &buf,
	}
}

// ---------- Reaper tests ----------

func TestReaper_TimesOutWorker(t *testing.T) {
	timeout := 100 * time.Millisecond
	kit := setupShortTimeout(t, timeout)
	ctx := context.Background()

	// Register a worker
	_, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "w1", Address: "a1"})
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}

	// Verify worker exists
	kit.coord.mu.Lock()
	_, exists := kit.coord.workers["w1"]
	kit.coord.mu.Unlock()

	if !exists {
		t.Fatal("worker not found after registration")
	}

	// Wait for reaper to fire (timeout + reaper interval + buffer)
	//nolint:mnd // test-specific sleep
	time.Sleep(timeout*2 + 50*time.Millisecond)

	// Worker should be reaped
	kit.coord.mu.Lock()
	_, exists = kit.coord.workers["w1"]
	kit.coord.mu.Unlock()

	if exists {
		t.Error("expected worker to be reaped after timeout")
	}
}

// ---------- StartCheckpoint with registered workers ----------

func TestStartCheckpoint_WithWorkers(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()

	// Register workers first so the workers loop body executes
	_, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "w1", Address: "a1"})
	if err != nil {
		t.Fatal(err)
	}

	_, err = kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "w2", Address: "a2"})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := kit.client.StartCheckpoint(ctx, &pb.StartCheckpointRequest{Epoch: 1, Path: "/ckpt"})
	if err != nil {
		t.Fatal(err)
	}

	kit.coord.mu.Lock()
	ckpt := kit.coord.checkpoints[resp.CheckpointId]
	numWorkers := len(ckpt.Workers)
	kit.coord.mu.Unlock()

	if numWorkers != 2 {
		t.Errorf("expected 2 workers in checkpoint, got %d", numWorkers)
	}
}

// ---------- StartCheckpoint epoch overflow ----------

func TestStartCheckpoint_EpochOverflow(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()

	epoch := int64(math.MaxInt32) + 1
	_, err := kit.client.StartCheckpoint(ctx, &pb.StartCheckpointRequest{Epoch: epoch, Path: "/ckpt"})
	if err == nil {
		t.Error("expected epoch overflow error")
	}
}

// ---------- EndCheckpoint: checkpoint not found ----------

func TestEndCheckpoint_NotFound(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()

	_, err := kit.client.EndCheckpoint(ctx, &pb.EndCheckpointRequest{
		WorkerId:     "w1",
		Epoch:        1,
		CheckpointId: "nonexistent",
	})
	if err == nil {
		t.Error("expected checkpoint not found error")
	}
}

// ---------- EndCheckpoint: partial completion ----------

func TestEndCheckpoint_PartialCompletion(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()

	// Register 2 workers
	_, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "w1", Address: "a1"})
	if err != nil {
		t.Fatal(err)
	}

	_, err = kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "w2", Address: "a2"})
	if err != nil {
		t.Fatal(err)
	}

	// Start checkpoint (workers loop body executes, covering lines 256-258)
	resp, err := kit.client.StartCheckpoint(ctx, &pb.StartCheckpointRequest{Epoch: 1, Path: "/ckpt"})
	if err != nil {
		t.Fatal(err)
	}

	// Only w1 reports completion → partial (covers lines 295-298)
	_, err = kit.client.EndCheckpoint(ctx, &pb.EndCheckpointRequest{
		WorkerId:     "w1",
		Epoch:        1,
		CheckpointId: resp.CheckpointId,
	})
	if err != nil {
		t.Fatal(err)
	}

	kit.coord.mu.Lock()
	partialCompleted := kit.coord.checkpoints[resp.CheckpointId].Completed
	kit.coord.mu.Unlock()

	if partialCompleted {
		t.Error("checkpoint should not be completed with only 1 of 2 workers done")
	}

	// w2 also reports → now completed
	_, err = kit.client.EndCheckpoint(ctx, &pb.EndCheckpointRequest{
		WorkerId:     "w2",
		Epoch:        1,
		CheckpointId: resp.CheckpointId,
	})
	if err != nil {
		t.Fatal(err)
	}

	kit.coord.mu.Lock()
	fullyCompleted := kit.coord.checkpoints[resp.CheckpointId].Completed
	kit.coord.mu.Unlock()

	if !fullyCompleted {
		t.Error("checkpoint should be completed after all workers report")
	}
}
