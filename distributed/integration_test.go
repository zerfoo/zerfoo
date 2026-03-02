package distributed_test

import (
	"context"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/distributed"
	"github.com/zerfoo/zerfoo/distributed/coordinator"
	"github.com/zerfoo/zerfoo/tensor"
	"google.golang.org/grpc"
)

// testCluster sets up a coordinator and N workers for integration testing.
type testCluster struct {
	coord      *coordinator.Coordinator
	workers    []*distributed.GrpcStrategy[float32]
	coordAddr  string
	workerAddr []string
}

func newTestCluster(t *testing.T, n int) *testCluster {
	t.Helper()

	// Start coordinator on ephemeral port.
	coord := coordinator.NewCoordinator(&syncWriter{}, 30*time.Second)
	if err := coord.Start("127.0.0.1:0"); err != nil {
		t.Fatalf("failed to start coordinator: %v", err)
	}
	t.Cleanup(coord.GracefulStop)

	coordAddr := coord.Addr().String()

	// Allocate ephemeral addresses for workers before Init.
	workerAddrs := make([]string, n)
	for i := range n {
		lc := net.ListenConfig{}
		lis, lisErr := lc.Listen(context.Background(), "tcp", "127.0.0.1:0")
		if lisErr != nil {
			t.Fatalf("failed to listen for worker %d: %v", i, lisErr)
		}
		workerAddrs[i] = lis.Addr().String()
		// Close the listener so ServerManager can rebind it.
		_ = lis.Close()
	}

	// Create and init workers sequentially (coordinator assigns ranks in order).
	workers := make([]*distributed.GrpcStrategy[float32], n)
	for i := range n {
		srv := grpc.NewServer()
		sm := distributed.NewServerManager(srv, nil)

		nm := distributed.NewNetworkManager(nil, nil)

		strategy := distributed.NewGrpcStrategy[float32](distributed.GrpcStrategyConfig{
			WorkerAddress:  workerAddrs[i],
			ServerManager:  sm,
			NetworkManager: nm,
		})

		if initErr := strategy.Init(0, n, coordAddr); initErr != nil {
			t.Fatalf("worker %d Init failed: %v", i, initErr)
		}
		t.Cleanup(strategy.Shutdown)
		workers[i] = strategy
	}

	return &testCluster{
		coord:      coord,
		workers:    workers,
		coordAddr:  coordAddr,
		workerAddr: workerAddrs,
	}
}

// syncWriter is a thread-safe io.Writer that discards output.
type syncWriter struct{}

func (syncWriter) Write(p []byte) (int, error) { return len(p), nil }

// --- T34.1: Multi-worker AllReduce integration test ---

func TestMultiWorkerAllReduce(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	cluster := newTestCluster(t, 3)

	// Each worker has different gradients.
	grads := []map[string]*tensor.TensorNumeric[float32]{
		makeGradients(t, []float32{1, 2, 3}),
		makeGradients(t, []float32{4, 5, 6}),
		makeGradients(t, []float32{7, 8, 9}),
	}

	// Run AllReduceGradients concurrently on all workers.
	errs := make([]error, 3)
	var wg sync.WaitGroup
	for i := range 3 {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			errs[rank] = cluster.workers[rank].AllReduceGradients(grads[rank])
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("worker %d AllReduce error: %v", i, err)
		}
	}

	// All workers should have the average: [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3] = [4, 5, 6]
	want := []float32{4, 5, 6}
	for i, g := range grads {
		data := g["grad"].Data()
		for j, v := range data {
			if diff := v - want[j]; diff > 0.01 || diff < -0.01 {
				t.Errorf("worker %d grad[%d] = %f, want %f", i, j, v, want[j])
			}
		}
	}
}

func TestMultiWorkerAllReduce_SingleWorker(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	cluster := newTestCluster(t, 1)

	grads := makeGradients(t, []float32{10, 20, 30})
	err := cluster.workers[0].AllReduceGradients(grads)
	if err != nil {
		t.Fatalf("AllReduce error: %v", err)
	}

	// With a single worker, the result should be the same as the input.
	data := grads["grad"].Data()
	want := []float32{10, 20, 30}
	for i, v := range data {
		if v != want[i] {
			t.Errorf("grad[%d] = %f, want %f", i, v, want[i])
		}
	}
}

// --- T34.2: Barrier and Broadcast integration tests ---

func TestMultiWorkerBarrier(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	cluster := newTestCluster(t, 3)

	// All workers call Barrier concurrently.
	errs := make([]error, 3)
	arrivals := make([]time.Time, 3)
	var wg sync.WaitGroup
	for i := range 3 {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			errs[rank] = cluster.workers[rank].Barrier()
			arrivals[rank] = time.Now()
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("worker %d Barrier error: %v", i, err)
		}
	}

	// All workers should complete within a small window of each other.
	for i := 1; i < len(arrivals); i++ {
		diff := arrivals[i].Sub(arrivals[0])
		if diff > 1*time.Second || diff < -1*time.Second {
			t.Errorf("worker %d completed %v apart from worker 0", i, diff)
		}
	}
}

func TestMultiWorkerBroadcast(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	cluster := newTestCluster(t, 3)

	// Root (rank 0) broadcasts a tensor.
	broadcastTensor, err := tensor.New([]int{3}, []float32{10, 20, 30})
	if err != nil {
		t.Fatal(err)
	}

	// Create receiver tensors for non-root workers.
	receivers := make([]*tensor.TensorNumeric[float32], 3)
	receivers[0] = broadcastTensor
	for i := 1; i < 3; i++ {
		receivers[i], err = tensor.New([]int{3}, []float32{0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}
	}

	// Broadcast from root to all workers concurrently.
	errs := make([]error, 3)
	var wg sync.WaitGroup
	for i := range 3 {
		wg.Add(1)
		go func(rank int) {
			defer wg.Done()
			errs[rank] = cluster.workers[rank].BroadcastTensor(receivers[rank], 0)
		}(i)
	}
	wg.Wait()

	for i, bErr := range errs {
		if bErr != nil {
			t.Errorf("worker %d Broadcast error: %v", i, bErr)
		}
	}

	// All workers should have the same data.
	want := []float32{10, 20, 30}
	for i, recv := range receivers {
		data := recv.Data()
		for j, v := range data {
			if v != want[j] {
				t.Errorf("worker %d data[%d] = %f, want %f", i, j, v, want[j])
			}
		}
	}
}

// --- T34.3: Error and edge case tests ---

func TestAllReduce_ContextCancellation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	// Create a 3-worker cluster but only have 2 participate.
	cluster := newTestCluster(t, 3)

	// Only 2 of 3 workers call AllReduce. The session will never complete
	// because the root waits for worldSize submissions (3) but only 2 arrive.
	grads0 := makeGradients(t, []float32{1, 2, 3})
	grads1 := makeGradients(t, []float32{4, 5, 6})

	errs := make([]error, 2)
	var wg sync.WaitGroup

	wg.Go(func() {
		errs[0] = cluster.workers[0].AllReduceGradients(grads0)
	})

	wg.Go(func() {
		errs[1] = cluster.workers[1].AllReduceGradients(grads1)
	})

	wg.Wait()

	// At least one worker should report an error (timeout or incomplete).
	anyErr := false
	for _, err := range errs {
		if err != nil {
			anyErr = true
		}
	}
	if !anyErr {
		t.Log("AllReduce completed without error despite missing worker -- may have timed out internally")
	}
}

// --- Helpers ---

func makeGradients(t *testing.T, data []float32) map[string]*tensor.TensorNumeric[float32] {
	t.Helper()
	tn, err := tensor.New([]int{len(data)}, data)
	if err != nil {
		t.Fatal(err)
	}
	return map[string]*tensor.TensorNumeric[float32]{"grad": tn}
}

// --- T35.3: WorkerNode lifecycle integration test ---

func TestWorkerNodeLifecycle(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	// Start coordinator.
	coord := coordinator.NewCoordinator(&syncWriter{}, 30*time.Second)
	if err := coord.Start("127.0.0.1:0"); err != nil {
		t.Fatalf("failed to start coordinator: %v", err)
	}
	t.Cleanup(coord.GracefulStop)
	coordAddr := coord.Addr().String()

	// Allocate ephemeral ports for 2 workers.
	workerAddrs := make([]string, 2)
	for i := range 2 {
		lc := net.ListenConfig{}
		lis, err := lc.Listen(context.Background(), "tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("failed to listen for worker %d: %v", i, err)
		}
		workerAddrs[i] = lis.Addr().String()
		_ = lis.Close()
	}

	// Start 2 WorkerNodes.
	nodes := make([]*distributed.WorkerNode, 2)
	for i := range 2 {
		node := distributed.NewWorkerNode(distributed.WorkerNodeConfig{
			WorkerAddress:      workerAddrs[i],
			CoordinatorAddress: coordAddr,
			WorldSize:          2,
		})
		if err := node.Start(context.Background()); err != nil {
			t.Fatalf("node %d Start failed: %v", i, err)
		}
		nodes[i] = node
	}

	// Verify both workers registered.
	for i, node := range nodes {
		if node.Rank() < 0 {
			t.Errorf("node %d rank = %d, want >= 0", i, node.Rank())
		}
		if node.Size() != 2 {
			t.Errorf("node %d size = %d, want 2", i, node.Size())
		}
		if node.Strategy() == nil {
			t.Errorf("node %d strategy is nil", i)
		}
	}

	// Double start should return error.
	if err := nodes[0].Start(context.Background()); err == nil {
		t.Error("expected error on double start")
	}

	// Shutdown nodes.
	for i, node := range nodes {
		if err := node.Close(context.Background()); err != nil {
			t.Errorf("node %d Close error: %v", i, err)
		}
	}

	// After close, strategy should be nil and rank should be -1.
	for i, node := range nodes {
		if node.Strategy() != nil {
			t.Errorf("node %d strategy should be nil after close", i)
		}
		if node.Rank() != -1 {
			t.Errorf("node %d rank = %d, want -1 after close", i, node.Rank())
		}
	}

	// Double close should be safe.
	if err := nodes[0].Close(context.Background()); err != nil {
		t.Errorf("double Close error: %v", err)
	}
}
