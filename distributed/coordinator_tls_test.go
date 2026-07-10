package distributed_test

import (
	"context"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/distributed"
	"github.com/zerfoo/zerfoo/distributed/coordinator"
	"github.com/zerfoo/zerfoo/distributed/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// TestCoordinator_RegisterWorker_RejectsUnauthenticatedCaller is DIST-2
// regression coverage: a coordinator configured with mutual TLS (SetTLS with
// CACertPath set) must reject a caller that presents no TLS credentials at
// all before RegisterWorker ever returns the peer list. This is the attack
// the review calls out -- an unauthenticated network caller enumerating
// every worker address in the cluster via a plaintext RegisterWorker call.
func TestCoordinator_RegisterWorker_RejectsUnauthenticatedCaller(t *testing.T) {
	dir := t.TempDir()
	caCert, cert, key := distributed.GenerateTestCerts(t, dir)

	tlsCfg := &distributed.TLSConfig{
		CACertPath: caCert,
		CertPath:   cert,
		KeyPath:    key,
	}

	coord := coordinator.NewCoordinator(&syncWriter{}, 30*time.Second)
	coord.SetTLS(tlsCfg)
	if err := coord.Start("127.0.0.1:0"); err != nil {
		t.Fatalf("failed to start coordinator: %v", err)
	}
	t.Cleanup(coord.GracefulStop)

	conn, err := grpc.NewClient(coord.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer func() { _ = conn.Close() }()

	client := pb.NewCoordinatorClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "attacker", Address: "10.0.0.1:9001"})
	if err == nil {
		t.Fatalf("RegisterWorker() with no TLS credentials succeeded and returned peers=%v; want the handshake to reject it", resp.GetPeers())
	}
	if resp != nil {
		t.Errorf("RegisterWorker() returned a non-nil response %v alongside error %v; want nil response on rejection", resp, err)
	}
	if got := status.Code(err); got != codes.Unavailable && got != codes.Unauthenticated {
		t.Errorf("RegisterWorker() error code = %v, want Unavailable (TLS handshake failure) or Unauthenticated, got err = %v", got, err)
	}
}

// TestCoordinator_RegisterWorker_TLSPairCompletesRegistration is DIST-2
// regression coverage for the positive path: two callers that present valid
// client certificates signed by the configured CA can register and receive
// the correct peer list, proving the mTLS gate authenticates rather than
// simply blocking all traffic.
func TestCoordinator_RegisterWorker_TLSPairCompletesRegistration(t *testing.T) {
	dir := t.TempDir()
	caCert, cert, key := distributed.GenerateTestCerts(t, dir)

	tlsCfg := &distributed.TLSConfig{
		CACertPath: caCert,
		CertPath:   cert,
		KeyPath:    key,
	}

	coord := coordinator.NewCoordinator(&syncWriter{}, 30*time.Second)
	coord.SetTLS(tlsCfg)
	if err := coord.Start("127.0.0.1:0"); err != nil {
		t.Fatalf("failed to start coordinator: %v", err)
	}
	t.Cleanup(coord.GracefulStop)

	clientCreds, err := tlsCfg.ClientCredentials()
	if err != nil {
		t.Fatalf("client credentials: %v", err)
	}

	conn, err := grpc.NewClient(coord.Addr().String(), grpc.WithTransportCredentials(clientCreds))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer func() { _ = conn.Close() }()

	client := pb.NewCoordinatorClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp1, err := client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "w0", Address: "127.0.0.1:9001"})
	if err != nil {
		t.Fatalf("RegisterWorker(w0) with a valid client cert: %v, want nil", err)
	}
	if resp1.GetRank() != 0 {
		t.Errorf("RegisterWorker(w0).Rank = %d, want 0", resp1.GetRank())
	}
	if got, want := resp1.GetPeers(), []string{"127.0.0.1:9001"}; len(got) != len(want) || got[0] != want[0] {
		t.Errorf("RegisterWorker(w0).Peers = %v, want %v", got, want)
	}

	resp2, err := client.RegisterWorker(ctx, &pb.RegisterWorkerRequest{WorkerId: "w1", Address: "127.0.0.1:9002"})
	if err != nil {
		t.Fatalf("RegisterWorker(w1) with a valid client cert: %v, want nil", err)
	}
	if resp2.GetRank() != 1 {
		t.Errorf("RegisterWorker(w1).Rank = %d, want 1", resp2.GetRank())
	}
	wantPeers := []string{"127.0.0.1:9001", "127.0.0.1:9002"}
	gotPeers := resp2.GetPeers()
	if len(gotPeers) != len(wantPeers) {
		t.Fatalf("RegisterWorker(w1).Peers = %v, want %v", gotPeers, wantPeers)
	}
	for i, addr := range wantPeers {
		if gotPeers[i] != addr {
			t.Errorf("RegisterWorker(w1).Peers[%d] = %q, want %q", i, gotPeers[i], addr)
		}
	}
}

// TestCoordinator_Start_RefusesNonLoopbackWithoutTLS is DIST-2 regression
// coverage mirroring TestWorkerNode_Start_RefusesNonLoopbackWithoutTLS: a
// coordinator bound to a routable address with no TLS configured must
// refuse to start.
func TestCoordinator_Start_RefusesNonLoopbackWithoutTLS(t *testing.T) {
	coord := coordinator.NewCoordinator(&syncWriter{}, 30*time.Second)

	err := coord.Start("0.0.0.0:0")
	if err == nil {
		coord.GracefulStop()
		t.Fatal("expected Start to refuse a non-loopback bind without TLS, got nil error")
	}
	if got := err.Error(); got == "" {
		t.Error("expected a descriptive error")
	}
}
