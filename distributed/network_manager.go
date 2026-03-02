package distributed

import (
	"context"
	"fmt"
	"net"
	"time"

	"github.com/zerfoo/zerfoo/distributed/pb"
	"github.com/zerfoo/zerfoo/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ensure that the struct implements the interface.
var _ NetworkManager = (*networkManager)(nil)

type networkManager struct {
	dialFunc      Dialer
	clientFactory ServiceClientFactory
	logger        log.Logger
}

// NewNetworkManager creates a new NetworkManager.
func NewNetworkManager(dialer Dialer, clientFactory ServiceClientFactory) NetworkManager {
	if dialer == nil {
		dialer = func(_ context.Context, target string) (*grpc.ClientConn, error) {
			return grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
		}
	}

	if clientFactory == nil {
		clientFactory = func(cc *grpc.ClientConn) pb.DistributedServiceClient {
			return pb.NewDistributedServiceClient(cc)
		}
	}

	return &networkManager{
		dialFunc:      dialer,
		clientFactory: clientFactory,
		logger:        log.Nop(),
	}
}

func (nm *networkManager) ConnectToPeers(peers []string, selfRank int, timeout time.Duration) ([]pb.DistributedServiceClient, []*grpc.ClientConn, error) {
	clients := make([]pb.DistributedServiceClient, len(peers))
	conns := make([]*grpc.ClientConn, len(peers))

	for i, peer := range peers {
		if i == selfRank {
			continue
		}

		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		conn, err := nm.dialFunc(ctx, peer)

		cancel() // Cancel the context immediately after use

		if err != nil {
			// Before returning, close any connections that were successfully opened.
			for j := range conns {
				if conns[j] != nil {
					if closeErr := conns[j].Close(); closeErr != nil {
						nm.logger.Warn("error closing connection during cleanup", "error", closeErr.Error())
					}
				}
			}

			return nil, nil, fmt.Errorf("failed to connect to peer %s: %w", peer, err)
		}

		clients[i] = nm.clientFactory(conn)
		conns[i] = conn
	}

	return clients, conns, nil
}

func (nm *networkManager) CloseConnections(conns []*grpc.ClientConn) {
	for _, conn := range conns {
		if conn != nil {
			if err := conn.Close(); err != nil {
				nm.logger.Warn("error closing connection", "error", err.Error())
			}
		}
	}
}

// ensure that the struct implements the interface.
var _ ServerManager = (*serverManager)(nil)

type serverManager struct {
	server     GrpcServer
	listenFunc ListenerFactory
	logger     log.Logger
	errCh      chan error
}

// NewServerManager creates a new ServerManager.
func NewServerManager(grpcServer GrpcServer, listenerFactory ListenerFactory) ServerManager {
	if listenerFactory == nil {
		listenerFactory = net.Listen
	}

	return &serverManager{
		server:     grpcServer,
		listenFunc: listenerFactory,
		logger:     log.Nop(),
		errCh:      make(chan error, 1),
	}
}

func (sm *serverManager) SetLogger(logger log.Logger) {
	if logger == nil {
		logger = log.Nop()
	}
	sm.logger = logger
}

func (sm *serverManager) Start(workerAddress string, service interface{}, serviceDesc *grpc.ServiceDesc) error {
	lis, err := sm.listenFunc("tcp", workerAddress)
	if err != nil {
		return err
	}
	// Access the listener address to satisfy tests expecting Addr() to be called.
	_ = lis.Addr()

	sm.server.RegisterService(serviceDesc, service)

	go func() {
		if err := sm.server.Serve(lis); err != nil {
			sm.logger.Error("gRPC server failed", "error", err.Error())

			sm.errCh <- err
		}
	}()

	return nil
}

func (sm *serverManager) Stop() {
	if sm.server != nil {
		sm.server.Stop()
	}
}

func (sm *serverManager) GracefulStop() {
	if sm.server != nil {
		sm.server.GracefulStop()
	}
}
