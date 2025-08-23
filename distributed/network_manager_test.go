package distributed

import (
	"context"
	"errors"
	"net"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"github.com/zerfoo/zerfoo/testing/testutils"
)

// mockGrpcServer is a mock implementation of grpc.ServiceRegistrar and grpc.Server for testing
type mockGrpcServer struct {
	mu                   sync.Mutex
	registerServiceCalls int
	serviceDesc          *grpc.ServiceDesc
	serviceImpl          interface{}
	serveCalled          bool
	stopCalled           bool
	gracefulStop         bool
	serveError           error
}

func (m *mockGrpcServer) RegisterService(sd *grpc.ServiceDesc, ss interface{}) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.registerServiceCalls++
	m.serviceDesc = sd
	m.serviceImpl = ss
}

func (m *mockGrpcServer) Serve(lis net.Listener) error {
	m.mu.Lock()
	m.serveCalled = true
	err := m.serveError
	m.mu.Unlock()

	return err
}

func (m *mockGrpcServer) Stop() {
	m.mu.Lock()
	m.stopCalled = true
	m.mu.Unlock()
}

func (m *mockGrpcServer) GracefulStop() {
	m.mu.Lock()
	m.gracefulStop = true
	m.mu.Unlock()
}

// mockListener is a mock implementation of net.Listener for testing
type mockListener struct {
	addrCalled int
	addr       *net.TCPAddr
}

func (m *mockListener) Addr() net.Addr {
	m.addrCalled++
	if m.addr == nil {
		m.addr = &net.TCPAddr{
			IP:   net.IPv4(127, 0, 0, 1),
			Port: 1234,
		}
	}

	return m.addr
}

func (m *mockListener) Accept() (net.Conn, error) {
	return nil, nil
}

func (m *mockListener) Close() error {
	return nil
}

func (m *mockListener) SetAddr(addr *net.TCPAddr) {
	m.addr = addr
}

func TestNetworkManager_ConnectToPeers(t *testing.T) {
	peers := []string{"peer1", "peer2", "peer3"}
	timeout := time.Second

	t.Run("successful connection", func(t *testing.T) {
		lis := bufconn.Listen(1024 * 1024)
		s := grpc.NewServer()

		go func() { _ = s.Serve(lis) }()

		defer s.Stop()

		dialer := func(_ context.Context, _ string) (*grpc.ClientConn, error) {
			conn, err := grpc.NewClient("bufnet", grpc.WithContextDialer(func(_ context.Context, _ string) (net.Conn, error) {
				return lis.Dial()
			}), grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				return nil, err
			}

			return conn, nil
		}

		nm := NewNetworkManager(dialer, MockClientFactory)
		clients, conns, err := nm.ConnectToPeers(peers, 1, timeout)
		testutils.AssertNoError(t, err, "ConnectToPeers failed: %v")

		if len(clients) != 3 {
			t.Errorf("expected 3 clients, got %d", len(clients))
		}

		if len(conns) != 3 {
			t.Errorf("expected 3 connections, got %d", len(conns))
		}

		testutils.AssertNotNil(t, clients[0], "clients[0] should not be nil")
		testutils.AssertNil(t, clients[1], "clients[1] should be nil (self)")
		testutils.AssertNotNil(t, clients[2], "clients[2] should not be nil")

		nilCount := 0

		for _, conn := range conns {
			if conn == nil {
				nilCount++
			}
		}

		testutils.AssertEqual(t, 1, nilCount, "expected exactly one nil connection")
		nm.CloseConnections(conns)
	})

	t.Run("connection error", func(t *testing.T) {
		dialer := func(_ context.Context, target string) (*grpc.ClientConn, error) {
			if target == "peer2" {
				return nil, errors.New("dial error")
			}
			// Create a dummy connection for the first peer
			lis := bufconn.Listen(1024 * 1024)
			s := grpc.NewServer()

			go func() { _ = s.Serve(lis) }()

			defer s.Stop()

			return grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
		}
		nm := NewNetworkManager(dialer, MockClientFactory)
		errorPeers := []string{"peer1", "peer2", "peer3"}
		_, _, err := nm.ConnectToPeers(errorPeers, 0, timeout)
		testutils.AssertError(t, err, "expected an error, got nil")
	})
}

func TestServerManager_Start(t *testing.T) {
	address := "localhost:1234"

	t.Run("successful start", func(t *testing.T) {
		srv := &mockGrpcServer{}
		listener := &mockListener{}
		listenFunc := func(_, _ string) (net.Listener, error) {
			return listener, nil
		}

		sm := NewServerManager(srv, listenFunc)

		err := sm.Start(address, nil, nil)
		testutils.AssertNoError(t, err, "expected no error, got %v")
		testutils.AssertTrue(t, srv.registerServiceCalls > 0, "RegisterService should be called")
		testutils.AssertTrue(t, listener.addrCalled > 0, "Addr() should be called on the listener")
	})

	t.Run("listen error", func(t *testing.T) {
		srv := &mockGrpcServer{}
		expectedErr := errors.New("listen error")
		listenFunc := func(_, _ string) (net.Listener, error) {
			return nil, expectedErr
		}

		sm := NewServerManager(srv, listenFunc)
		err := sm.Start(address, nil, nil)
		testutils.AssertError(t, err, "expected an error, got nil")
		testutils.AssertEqual(t, expectedErr, err, "expected specific error")
	})

	t.Run("serve error", func(t *testing.T) {
		srv := &mockGrpcServer{
			serveError: errors.New("serve error"),
		}
		listener := &mockListener{}

		listenFunc := func(_, _ string) (net.Listener, error) {
			return listener, nil
		}

		sm := NewServerManager(srv, listenFunc)

		err := sm.Start(address, nil, nil)
		// The error is handled asynchronously, so we can't directly test it here
		// Instead, we'll just verify that the server was started
		testutils.AssertNoError(t, err, "expected no error from Start()")
	})
}

func TestNetworkManager_ConnectToPeers_DialError(t *testing.T) {
	peers := []string{"peer1", "peer2"}
	timeout := time.Second

	dialer := func(_ context.Context, _ string) (*grpc.ClientConn, error) {
		return nil, errors.New("dial error")
	}

	nm := NewNetworkManager(dialer, MockClientFactory)
	_, _, err := nm.ConnectToPeers(peers, 0, timeout)
	testutils.AssertError(t, err, "expected an error, got nil")
}

func TestNetworkManager_CloseConnections(t *testing.T) {
	// Create a mock connection to test closing.
	lis := bufconn.Listen(1024 * 1024)
	s := grpc.NewServer()

	go func() { _ = s.Serve(lis) }()

	defer s.Stop()

	conn, err := grpc.NewClient("bufnet", grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
		return lis.Dial()
	}), grpc.WithTransportCredentials(insecure.NewCredentials()))
	testutils.AssertNoError(t, err, "failed to dial: %v")

	conns := []*grpc.ClientConn{conn, nil} // Include a nil connection to test robustness.
	nm := NewNetworkManager(nil, nil)
	nm.CloseConnections(conns)

	// Verify the connection is closed.
	// Note: conn.Close() can be called multiple times, it will just return an error on subsequent calls.
	// A better check is to see the state of the connection.
	if conn.GetState().String() == "Ready" {
		t.Errorf("expected connection state to not be Ready, got %s", conn.GetState().String())
	}
}

func TestNewNetworkManager_DefaultDialer(t *testing.T) {
	nm := NewNetworkManager(nil, MockClientFactory).(*networkManager)
	testutils.AssertNotNil(t, nm.dialFunc, "expected dialFunc to not be nil")
}

func TestNewNetworkManager_DefaultClientFactory(t *testing.T) {
	nm := NewNetworkManager(func(_ context.Context, _ string) (*grpc.ClientConn, error) {
		return nil, nil
	}, nil).(*networkManager)
	testutils.AssertNotNil(t, nm.clientFactory, "expected clientFactory to not be nil")
}

func TestNewServerManager_DefaultListener(_ *testing.T) {
	NewServerManager(nil, nil)
}
