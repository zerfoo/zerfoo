package distributed

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"github.com/zerfoo/zerfoo/testing/testutils"
)

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
	customMockServer := new(CustomMockGrpcServer)
	customMockListener := new(CustomMockListener)

	t.Run("successful start", func(t *testing.T) {
		listenFunc := func(_ string, addr string) (net.Listener, error) {
			return customMockListener, nil
		}
		sm := NewServerManager(customMockServer, listenFunc)

		// customMockServer.On("RegisterService", mock.Anything, mock.Anything).Once()
		// customMockServer.On("Serve", customMockListener).Return(nil).Once()
		customMockListener.OnAddr(&net.TCPAddr{})

		err := sm.Start(address, nil, nil)
		testutils.AssertNoError(t, err, "expected no error, got %v")
	})

	t.Run("listen error", func(t *testing.T) {
		listenFunc := func(_ string, addr string) (net.Listener, error) {
			return nil, errors.New("listen error")
		}
		sm := NewServerManager(customMockServer, listenFunc)
		err := sm.Start(address, nil, nil)
		testutils.AssertError(t, err, "expected an error, got nil")
	})

	t.Run("serve error", func(t *testing.T) {
		listenFunc := func(_ string, addr string) (net.Listener, error) {
			return customMockListener, nil
		}
		sm := NewServerManager(customMockServer, listenFunc)
		customMockLogger := new(CustomMockLogger)
		sm.SetLogger(customMockLogger)

		// customMockServer.On("RegisterService", mock.Anything, mock.Anything).Once()
		customMockServer.OnServe(errors.New("serve error"))
		customMockListener.OnAddr(&net.TCPAddr{})
		// customMockLogger.On("Printf", mock.Anything, mock.Anything).Once()

		err := sm.Start(address, nil, nil)
		testutils.AssertNoError(t, err, "expected no error, got %v")

		// Give the goroutine time to run
		time.Sleep(10 * time.Millisecond)

		// Give the goroutine time to run
		time.Sleep(10 * time.Millisecond)

		// customMockLogger.AssertExpectations(t)
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
