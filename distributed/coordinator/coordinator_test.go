package coordinator

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/distributed/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"github.com/zerfoo/zerfoo/testing/testutils"
)

const bufSize = 1024 * 1024

type testKit struct {
	client pb.CoordinatorClient
	coord  *Coordinator
	lis    *bufconn.Listener
	buf    *bytes.Buffer
}

func setup(t *testing.T) *testKit {
	lis := bufconn.Listen(bufSize)
	var buf bytes.Buffer
	coord := NewCoordinator(&buf, 10*time.Second)

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
		if err := conn.Close(); err != nil {
			t.Errorf("error closing connection: %v", err)
		}
	})

	return &testKit{
		client: pb.NewCoordinatorClient(conn),
		coord:  coord,
		lis:    lis,
		buf:    &buf,
	}
}
func TestCoordinator_Start(t *testing.T) {
	t.Run("successful start", func(t *testing.T) {
		var buf bytes.Buffer
		coord := NewCoordinator(&buf, 10*time.Second)
		err := coord.Start("localhost:0")
		if err != nil {
			t.Fatalf("failed to start coordinator: %v", err)
		}
		defer coord.Stop()
		if coord.Addr() == nil {
			t.Errorf("expected coordinator address to not be nil, got nil")
		}
	})

	t.Run("listen error", func(t *testing.T) {
		var buf bytes.Buffer
		coord := NewCoordinator(&buf, 10*time.Second)
		// Let's create a server on a port to make the next call fail.
		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("failed to listen: %v", err)
		}
		defer func() { _ = lis.Close() }()

		err = coord.Start(lis.Addr().String())
		if err == nil {
			t.Errorf("expected an error, got nil")
		}
	})

	t.Run("serve error", func(t *testing.T) {
		var buf bytes.Buffer
		coord := NewCoordinator(&buf, 10*time.Second)
		ml := &testutils.CustomMockListener{
			AcceptErr: errors.New("mock error"),
			AddrVal:   &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 12345},
		}
		coord.start(ml)
		time.Sleep(10 * time.Millisecond) // give time for the go routine to run
		testutils.AssertContains(t, buf.String(), "gRPC server failed", "expected log to contain %q, got %q")
	})
}
func TestCoordinator_RegisterWorker(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()

	// Test first worker
	resp1, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-1", Address: "addr-1"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	if resp1.Rank != int32(0) {
		t.Errorf("expected rank 0, got %d", resp1.Rank)
	}
	if len(resp1.Peers) != 1 || resp1.Peers[0] != "addr-1" {
		t.Errorf("expected peers [\"addr-1\"], got %v", resp1.Peers)
	}

	// Test second worker
	resp2, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-2", Address: "addr-2"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	if resp2.Rank != int32(1) {
		t.Errorf("expected rank 1, got %d", resp2.Rank)
	}
	// Custom ElementsMatch for string slices
	if !testutils.ElementsMatch(resp2.Peers, []string{"addr-1", "addr-2"}) {
		t.Errorf("expected peers %v, got %v", []string{"addr-1", "addr-2"}, resp2.Peers)
	}

	// Test duplicate registration
	_, err = kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-1", Address: "addr-1"})
	if err == nil {
		t.Errorf("expected an error for duplicate registration, got nil")
	}

	// Test rank not found - this is a synthetic test
	kit.coord.mu.Lock()
	// create a gap in ranks
	delete(kit.coord.ranks, 0)
	kit.coord.mu.Unlock()
	resp3, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-3", Address: "addr-3"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	if resp3.Rank != int32(2) {
		t.Errorf("expected rank 2, got %d", resp3.Rank)
	}
	// worker-2 is rank 1, worker-3 is rank 2. worker-1 was deleted.
	if !testutils.ElementsMatch(resp3.Peers, []string{"addr-2", "addr-3"}) {
		t.Errorf("expected peers %v, got %v", []string{"addr-2", "addr-3"}, resp3.Peers)
	}
	// Test worker not found in workers map - this is a synthetic test
	kit.coord.mu.Lock()
	// create an inconsistent state
	kit.coord.ranks[1] = "worker-dne"
	kit.coord.mu.Unlock()
	resp4, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-4", Address: "addr-4"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	if resp4.Rank != int32(3) {
		t.Errorf("expected rank 3, got %d", resp4.Rank)
	}
	if !strings.Contains(kit.buf.String(), "worker worker-dne not found in workers map") {
		t.Errorf("expected log to contain \"worker worker-dne not found in workers map\", got %s", kit.buf.String())
	}
}

func TestCoordinator_UnregisterWorker(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()

	// Register a worker first
	_, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-1", Address: "addr-1"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}

	// Unregister the worker
	_, err = kit.client.UnregisterWorker(ctx, &pb.UnregisterWorkerRequest{WorkerId: "worker-1"})
	if err != nil {
		t.Fatalf("failed to unregister worker: %v", err)
	}

	// Verify it's gone
	kit.coord.mu.Lock()
	_, ok := kit.coord.workers["worker-1"]
	kit.coord.mu.Unlock()
	if ok {
		t.Errorf("expected worker to be unregistered, but it still exists")
	}

	// Test unregistering a non-existent worker
	_, err = kit.client.UnregisterWorker(ctx, &pb.UnregisterWorkerRequest{WorkerId: "worker-2"})
	if err == nil {
		t.Errorf("expected an error for non-existent worker, got nil")
	}
}

func TestCoordinator_Heartbeat(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()

	// Register a worker first
	_, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-1", Address: "addr-1"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}

	// Get initial heartbeat time
	kit.coord.mu.Lock()
	initialHeartbeat := kit.coord.workers["worker-1"].LastHeartbeat
	kit.coord.mu.Unlock()

	// Wait a bit to ensure time progresses
	time.Sleep(10 * time.Millisecond)

	// Send heartbeat
	resp, err := kit.client.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: "worker-1"})
	if err != nil {
		t.Fatalf("failed to send heartbeat: %v", err)
	}
	testutils.AssertEqual(t, "OK", resp.Status, "expected status %q, got %q")

	// Verify heartbeat time was updated
	kit.coord.mu.Lock()
	newHeartbeat := kit.coord.workers["worker-1"].LastHeartbeat
	kit.coord.mu.Unlock()
	testutils.AssertTrue(t, newHeartbeat.After(initialHeartbeat), "expected new heartbeat to be after initial heartbeat")

	// Test heartbeat for non-existent worker
	_, err = kit.client.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: "worker-2"})
	testutils.AssertError(t, err, "expected an error for non-existent worker, got nil")
}

func TestCoordinator_Checkpoints(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()

	// Start Checkpoint
	startResp, err := kit.client.StartCheckpoint(ctx, &pb.StartCheckpointRequest{Epoch: 1, Path: "/checkpoints"})
	if err != nil {
		t.Fatalf("failed to start checkpoint: %v", err)
	}
	testutils.AssertEqual(t, "ckpt-1", startResp.CheckpointId, "expected checkpoint ID %q, got %q")

	// End Checkpoint
	_, err = kit.client.EndCheckpoint(ctx, &pb.EndCheckpointRequest{WorkerId: "worker-1", Epoch: 1, CheckpointId: "ckpt-1"})
	if err != nil {
		t.Fatalf("failed to end checkpoint: %v", err)
	}
}

func TestCoordinator_ConcurrentOperations(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()
	var wg sync.WaitGroup
	numWorkers := 50

	// Register initial workers
	for i := 0; i < numWorkers; i++ {
		workerID := fmt.Sprintf("worker-%d", i)
		addr := fmt.Sprintf("addr-%d", i)
		_, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: workerID, Address: addr})
		if err != nil {
			t.Fatalf("failed to register worker: %v", err)
		}
	}

	// Concurrent register, unregister, and heartbeat
	for i := 0; i < numWorkers*2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			op := i % 3
			switch op {
			case 0: // Register new worker
				workerID := fmt.Sprintf("new-worker-%d", i)
				addr := fmt.Sprintf("new-addr-%d", i)
				_, _ = kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: workerID, Address: addr})
			case 1: // Unregister existing worker
				workerID := fmt.Sprintf("worker-%d", i/3)
				_, _ = kit.client.UnregisterWorker(ctx, &pb.UnregisterWorkerRequest{WorkerId: workerID})
			case 2: // Heartbeat existing worker
				workerID := fmt.Sprintf("worker-%d", i/3)
				_, _ = kit.client.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: workerID})
			}
		}(i)
	}

	wg.Wait()
}

func TestCoordinator_StartAndStop(t *testing.T) {
	var buf bytes.Buffer
	coord := NewCoordinator(&buf, 10*time.Second)

	err := coord.Start("localhost:0")
	testutils.AssertNoError(t, err, "failed to start coordinator: %v")
	testutils.AssertNotNil(t, coord.Addr(), "expected coordinator address to not be nil, got nil")

	// Stop the server
	coord.Stop()

	// Try to dial the closed server (should fail)
	_, err = net.DialTimeout("tcp", coord.Addr().String(), 100*time.Millisecond)
	testutils.AssertError(t, err, "expected an error when dialing closed server, got nil")
}

func TestCoordinator_Addr(t *testing.T) {
	t.Run("returns nil when not started", func(t *testing.T) {
		var buf bytes.Buffer
		coord := NewCoordinator(&buf, 10*time.Second)
		testutils.AssertNil(t, coord.Addr(), "expected coordinator address to be nil")
	})

	t.Run("returns address when started", func(t *testing.T) {
		var buf bytes.Buffer
		coord := NewCoordinator(&buf, 10*time.Second)
		err := coord.Start("localhost:0")
		if err != nil {
			t.Fatalf("failed to start coordinator: %v", err)
		}
		defer coord.Stop()
		testutils.AssertNotNil(t, coord.Addr(), "expected coordinator address to not be nil, got nil")
	})
}

func TestCoordinator_GracefulStop(t *testing.T) {
	var buf bytes.Buffer
	coord := NewCoordinator(&buf, 10*time.Second)

	err := coord.Start("localhost:0")
	testutils.AssertNoError(t, err, "failed to start coordinator: %v")
	testutils.AssertNotNil(t, coord.Addr(), "expected coordinator address to not be nil, got nil")

	// Stop the server
	coord.GracefulStop()
	time.Sleep(100 * time.Millisecond)

	// Try to dial the closed server (should fail)
	_, err = net.DialTimeout("tcp", coord.Addr().String(), 100*time.Millisecond)
	testutils.AssertError(t, err, "expected an error when dialing closed server, got nil")
}

func TestCoordinator_RegisterWorker_RankAssignment(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()

	// Register a few workers
	resp1, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-a", Address: "addr-a"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	testutils.AssertEqual(t, int32(0), resp1.Rank, "expected rank %d, got %d")

	resp2, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-b", Address: "addr-b"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	testutils.AssertEqual(t, int32(1), resp2.Rank, "expected rank %d, got %d")

	// Unregister the first worker
	_, err = kit.client.UnregisterWorker(ctx, &pb.UnregisterWorkerRequest{WorkerId: "worker-a"})
	if err != nil {
		t.Fatalf("failed to unregister worker: %v", err)
	}

	// Register a new worker, it should get a new rank, not reuse the old one.
	resp3, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-c", Address: "addr-c"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	testutils.AssertEqual(t, int32(2), resp3.Rank, "expected rank %d, got %d")
}

func TestCoordinator_RegisterWorker_Peers(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()

	// Worker 1
	resp1, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-1", Address: "addr-1"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	testutils.AssertTrue(t, testutils.ElementsMatch(resp1.Peers, []string{"addr-1"}), "expected peers to match")

	// Worker 2
	resp2, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-2", Address: "addr-2"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	testutils.AssertTrue(t, testutils.ElementsMatch(resp2.Peers, []string{"addr-1", "addr-2"}), "expected peers to match")

	// Worker 3
	resp3, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-3", Address: "addr-3"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	testutils.AssertTrue(t, testutils.ElementsMatch(resp3.Peers, []string{"addr-1", "addr-2", "addr-3"}), "expected peers to match")

	// Unregister worker 2
	_, err = kit.client.UnregisterWorker(ctx, &pb.UnregisterWorkerRequest{WorkerId: "worker-2"})
	if err != nil {
		t.Fatalf("failed to unregister worker: %v", err)
	}

	// Register worker 4, check peers again
	resp4, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "worker-4", Address: "addr-4"})
	if err != nil {
		t.Fatalf("failed to register worker: %v", err)
	}
	// The peers should now be worker 1, 3, and 4.
	// Note: The order of peers is not guaranteed, so we use ElementsMatch.
	// The ranks map has a gap, but the peers list should be dense.
	kit.coord.mu.Lock()
	expectedPeers := []string{}
	for r := 0; r < kit.coord.nextRank; r++ {
		if workerID, ok := kit.coord.ranks[r]; ok {
			expectedPeers = append(expectedPeers, kit.coord.workers[workerID].Address)
		}
	}
	kit.coord.mu.Unlock()

	testutils.AssertTrue(t, testutils.ElementsMatch(expectedPeers, resp4.Peers), "expected peers to match")
}

// mockListener is a mock implementation of net.Listener for testing purposes.
type mockListener struct {
	acceptErr error
	closeErr  error
	addr      net.Addr
}

func (m *mockListener) Accept() (net.Conn, error) {
	if m.acceptErr != nil {
		return nil, m.acceptErr
	}
	// This will block forever, which is fine for the test where we don't expect Accept to be called.
	select {}
}

func (m *mockListener) Close() error {
	return m.closeErr
}

func (m *mockListener) Addr() net.Addr {
	return m.addr
}

func TestCoordinator_Stop(t *testing.T) {
	t.Run("stops a running server", func(t *testing.T) {
		var buf bytes.Buffer
		coord := NewCoordinator(&buf, 10*time.Second)
		err := coord.Start("localhost:0")
		testutils.AssertNoError(t, err, "failed to start coordinator: %v")
		testutils.AssertNotNil(t, coord.Addr(), "expected coordinator address to not be nil, got nil")
		coord.Stop()
		_, err = net.DialTimeout("tcp", coord.Addr().String(), 100*time.Millisecond)
		testutils.AssertError(t, err, "expected an error when dialing closed server, got nil")
	})

	t.Run("does nothing if server is not started", func(t *testing.T) {
		var buf bytes.Buffer
		coord := NewCoordinator(&buf, 10*time.Second)
		// Note: We are not calling coord.Start()
		coord.Stop() // Should not panic
	})
}

func TestCoordinator_UnregisterWorker_Table(t *testing.T) {
	tests := []struct {
		name          string
		workerID      string
		setupFunc     func(kit *testKit)
		expectErr     bool
		expectWorkers map[string]*WorkerInfo
	}{
		{
			name:     "successful unregister",
			workerID: "worker-1",
			setupFunc: func(kit *testKit) {
				_, _ = kit.client.RegisterWorker(context.Background(), &pb.RegisterWorkerRequest{WorkerId: "worker-1", Address: "addr-1"})
			},
			expectErr:     false,
			expectWorkers: map[string]*WorkerInfo{},
		},
		{
			name:      "unregister non-existent worker",
			workerID:  "worker-2",
			setupFunc: nil,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kit := setup(t)
			ctx := context.Background()

			if tt.setupFunc != nil {
				tt.setupFunc(kit)
			}

			_, err := kit.client.UnregisterWorker(ctx, &pb.UnregisterWorkerRequest{WorkerId: tt.workerID})

			if tt.expectErr {
				testutils.AssertError(t, err, "expected an error, got nil")
			} else {
				testutils.AssertNoError(t, err, "expected no error, got %v")
				kit.coord.mu.Lock()
				defer kit.coord.mu.Unlock()
				if tt.expectWorkers != nil {
					testutils.AssertEqual(t, len(tt.expectWorkers), len(kit.coord.workers), "expected %d workers, got %d")
				}
			}
		})
	}
}

func TestCoordinator_Heartbeat_Table(t *testing.T) {
	tests := []struct {
		name         string
		workerID     string
		setupFunc    func(kit *testKit)
		expectErr    bool
		expectStatus string
	}{
		{
			name:     "successful heartbeat",
			workerID: "worker-1",
			setupFunc: func(kit *testKit) {
				_, _ = kit.client.RegisterWorker(context.Background(), &pb.RegisterWorkerRequest{WorkerId: "worker-1", Address: "addr-1"})
			},
			expectErr:    false,
			expectStatus: "OK",
		},
		{
			name:      "heartbeat for non-existent worker",
			workerID:  "worker-2",
			setupFunc: nil,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kit := setup(t)
			ctx := context.Background()

			if tt.setupFunc != nil {
				tt.setupFunc(kit)
			}

			resp, err := kit.client.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: tt.workerID})

			if tt.expectErr {
				testutils.AssertError(t, err, "expected an error, got nil")
			} else {
				testutils.AssertNoError(t, err, "expected no error, got %v")
				testutils.AssertEqual(t, tt.expectStatus, resp.Status, "expected status %q, got %q")
			}
		})
	}
}

func TestNewCoordinator(t *testing.T) {
	var buf bytes.Buffer
	coord := NewCoordinator(&buf, 10*time.Second)
	testutils.AssertNotNil(t, coord, "expected coordinator to not be nil")
	testutils.AssertNotNil(t, coord.workers, "expected workers map to not be nil")
	testutils.AssertNotNil(t, coord.ranks, "expected ranks map to not be nil")
	testutils.AssertNotNil(t, coord.logger, "expected logger to not be nil")
	testutils.AssertEqual(t, &buf, coord.out.(*bytes.Buffer), "expected output buffer to be %v, got %v")
}

func TestWorkerInfo(t *testing.T) {
	wi := &WorkerInfo{
		ID:            "worker-1",
		Address:       "localhost:1234",
		Rank:          0,
		LastHeartbeat: time.Now(),
	}
	testutils.AssertEqual(t, "worker-1", wi.ID, "expected ID %q, got %q")
	testutils.AssertEqual(t, "localhost:1234", wi.Address, "expected address %q, got %q")
	testutils.AssertEqual(t, 0, wi.Rank, "expected rank %d, got %d")
	testutils.AssertFalse(t, wi.LastHeartbeat.IsZero(), "expected LastHeartbeat to not be zero")
}

func TestCoordinator_RegisterWorker_EmptyID(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()
	_, err := kit.client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "", Address: "addr-1"})
	testutils.AssertError(t, err, "expected an error for empty worker ID, got nil")
}

func TestCoordinator_UnregisterWorker_EmptyID(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()
	_, err := kit.client.UnregisterWorker(ctx, &pb.UnregisterWorkerRequest{WorkerId: ""})
	testutils.AssertError(t, err, "expected an error for empty worker ID, got nil")
}

func TestCoordinator_Heartbeat_EmptyID(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()
	_, err := kit.client.Heartbeat(ctx, &pb.HeartbeatRequest{WorkerId: ""})
	testutils.AssertError(t, err, "expected an error for empty worker ID, got nil")
}

func TestCoordinator_EndCheckpoint_EmptyWorkerID(t *testing.T) {
	kit := setup(t)
	ctx := context.Background()
	_, err := kit.client.EndCheckpoint(ctx, &pb.EndCheckpointRequest{WorkerId: "", Epoch: 1, CheckpointId: "ckpt-1"})
	testutils.AssertError(t, err, "expected an error for empty worker ID, got nil")
}
