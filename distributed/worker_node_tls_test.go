package distributed_test

import (
	"context"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/distributed"
	"github.com/zerfoo/zerfoo/distributed/coordinator"
	"github.com/zerfoo/zerfoo/distributed/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// reserveLoopbackAddr grabs an ephemeral loopback port, returns its address
// string, and releases the listener so the caller (typically a WorkerNode)
// can rebind it.
func reserveLoopbackAddr(t *testing.T) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to reserve loopback address: %v", err)
	}
	addr := lis.Addr().String()
	if err := lis.Close(); err != nil {
		t.Fatalf("failed to release reserved listener: %v", err)
	}
	return addr
}

// TestWorkerNode_Start_RefusesNonLoopbackWithoutTLS is DIST-1 regression
// coverage: a worker bound to a routable (non-loopback) address with no TLS
// configuration must refuse to start rather than run an unauthenticated
// gRPC server. The refusal must happen before any coordinator dial is
// attempted -- there is no reachable coordinator in this test, so a slow
// failure would indicate the check fired too late (e.g. after a 10s
// registration timeout) rather than up front.
func TestWorkerNode_Start_RefusesNonLoopbackWithoutTLS(t *testing.T) {
	wn := distributed.NewWorkerNode(distributed.WorkerNodeConfig{
		WorkerAddress:      "0.0.0.0:0",
		CoordinatorAddress: "127.0.0.1:1", // deliberately unreachable; must never be dialed
		WorldSize:          1,
	})

	start := time.Now()
	err := wn.Start(context.Background())
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected Start to refuse a non-loopback bind without TLS, got nil error")
	}
	if !strings.Contains(err.Error(), "refusing non-loopback bind without TLS") {
		t.Errorf("Start() error = %q, want it to mention refusing a non-loopback bind without TLS", err.Error())
	}
	if elapsed > 2*time.Second {
		t.Errorf("Start() took %v to refuse; expected an immediate rejection before any coordinator dial", elapsed)
	}
	if wn.Rank() != -1 {
		t.Error("expected worker to remain unstarted after a refused Start")
	}
}

// TestWorkerNode_Start_LoopbackWithoutTLS_Starts is DIST-1 regression
// coverage: single-host / dev usage (loopback bind, no TLS configured) must
// keep working exactly as before this fix.
func TestWorkerNode_Start_LoopbackWithoutTLS_Starts(t *testing.T) {
	coord := coordinator.NewCoordinator(&syncWriter{}, 30*time.Second)
	if err := coord.Start("127.0.0.1:0"); err != nil {
		t.Fatalf("failed to start coordinator: %v", err)
	}
	t.Cleanup(coord.GracefulStop)

	workerAddr := reserveLoopbackAddr(t)

	wn := distributed.NewWorkerNode(distributed.WorkerNodeConfig{
		WorkerAddress:      workerAddr,
		CoordinatorAddress: coord.Addr().String(),
		WorldSize:          1,
	})

	if err := wn.Start(context.Background()); err != nil {
		t.Fatalf("Start() error = %v, want nil for a loopback bind without TLS", err)
	}
	t.Cleanup(func() { _ = wn.Close(context.Background()) })

	if wn.Rank() != 0 {
		t.Errorf("Rank() = %d, want 0 for the sole registered worker", wn.Rank())
	}
}

// TestWorkerNode_Start_TLS_AcceptsValidClientCert is DIST-1 regression
// coverage for the mTLS path: a worker configured with TLS starts its gRPC
// server with server credentials wired in, and a client presenting a cert
// signed by the configured CA is accepted. A client with no credentials at
// all must be rejected, proving the credentials are actually enforced
// rather than silently optional.
func TestWorkerNode_Start_TLS_AcceptsValidClientCert(t *testing.T) {
	dir := t.TempDir()
	caCert, cert, key := distributed.GenerateTestCerts(t, dir)

	tlsCfg := &distributed.TLSConfig{
		CACertPath: caCert,
		CertPath:   cert,
		KeyPath:    key,
	}

	coordCreds, err := tlsCfg.ServerCredentials()
	if err != nil {
		t.Fatalf("coordinator server credentials: %v", err)
	}
	coord := coordinator.NewCoordinator(&syncWriter{}, 30*time.Second)
	coord.SetServerOptions(grpc.Creds(coordCreds))
	if err := coord.Start("127.0.0.1:0"); err != nil {
		t.Fatalf("failed to start coordinator: %v", err)
	}
	t.Cleanup(coord.GracefulStop)

	workerAddr := reserveLoopbackAddr(t)

	wn := distributed.NewWorkerNode(distributed.WorkerNodeConfig{
		WorkerAddress:      workerAddr,
		CoordinatorAddress: coord.Addr().String(),
		WorldSize:          1,
		TLS:                tlsCfg,
	})

	if err := wn.Start(context.Background()); err != nil {
		t.Fatalf("Start() error = %v, want nil for a TLS-configured worker", err)
	}
	t.Cleanup(func() { _ = wn.Close(context.Background()) })

	clientCreds, err := tlsCfg.ClientCredentials()
	if err != nil {
		t.Fatalf("client credentials: %v", err)
	}

	t.Run("valid client cert is accepted", func(t *testing.T) {
		conn, err := grpc.NewClient(workerAddr, grpc.WithTransportCredentials(clientCreds))
		if err != nil {
			t.Fatalf("grpc.NewClient: %v", err)
		}
		defer func() { _ = conn.Close() }()

		client := pb.NewDistributedServiceClient(conn)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if _, err := client.Barrier(ctx, &pb.BarrierRequest{Rank: 0}); err != nil {
			t.Errorf("Barrier() with a valid client cert: %v, want nil", err)
		}
	})

	t.Run("unauthenticated client is rejected", func(t *testing.T) {
		conn, err := grpc.NewClient(workerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("grpc.NewClient: %v", err)
		}
		defer func() { _ = conn.Close() }()

		client := pb.NewDistributedServiceClient(conn)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if _, err := client.Barrier(ctx, &pb.BarrierRequest{Rank: 0}); err == nil {
			t.Error("Barrier() with no client credentials succeeded; want the mTLS handshake to reject it")
		}
	})
}
