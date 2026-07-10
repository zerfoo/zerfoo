package cli

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/distributed"
	"github.com/zerfoo/zerfoo/serve/shutdown"
)

// workerNode is the subset of *distributed.WorkerNode that WorkerCommand
// depends on. It exists so tests can substitute a fake node and observe the
// distributed.WorkerNodeConfig (in particular the TLS field) that Run built
// from CLI flags, without standing up a real gRPC server/coordinator pair.
type workerNode interface {
	Start(ctx context.Context) error
	Close(ctx context.Context) error
}

// WorkerCommand implements the "worker" CLI command for starting a
// distributed training worker.
type WorkerCommand struct {
	shutdownCoord *shutdown.Coordinator

	// newWorkerNode constructs the worker node from its config. Defaults to
	// wrapping distributed.NewWorkerNode; overridable in tests.
	newWorkerNode func(distributed.WorkerNodeConfig) workerNode
}

// NewWorkerCommand creates a new WorkerCommand. The shutdown coordinator
// is used to register the worker node for orderly shutdown on signal.
func NewWorkerCommand(coord *shutdown.Coordinator) *WorkerCommand {
	return &WorkerCommand{
		shutdownCoord: coord,
		newWorkerNode: func(cfg distributed.WorkerNodeConfig) workerNode {
			return distributed.NewWorkerNode(cfg)
		},
	}
}

// Name implements Command.Name.
func (c *WorkerCommand) Name() string { return "worker" }

// Description implements Command.Description.
func (c *WorkerCommand) Description() string {
	return "Start a distributed training worker"
}

// Run implements Command.Run. It parses flags, creates a WorkerNode,
// starts it, and blocks until the context is canceled (e.g. by SIGTERM).
func (c *WorkerCommand) Run(ctx context.Context, args []string) error {
	var coordAddr, workerAddr, workerID string
	var tlsCert, tlsKey, tlsCA string
	var worldSize int

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--coordinator-address":
			if i+1 >= len(args) {
				return errors.New("--coordinator-address requires a value")
			}
			coordAddr = args[i+1]
			i++
		case "--worker-address":
			if i+1 >= len(args) {
				return errors.New("--worker-address requires a value")
			}
			workerAddr = args[i+1]
			i++
		case "--worker-id":
			if i+1 >= len(args) {
				return errors.New("--worker-id requires a value")
			}
			workerID = args[i+1]
			i++
		case "--world-size":
			if i+1 >= len(args) {
				return errors.New("--world-size requires a value")
			}
			n, err := parsePositiveInt(args[i+1])
			if err != nil {
				return fmt.Errorf("--world-size: %w", err)
			}
			worldSize = n
			i++
		case "--tls-cert":
			if i+1 >= len(args) {
				return errors.New("--tls-cert requires a value")
			}
			tlsCert = args[i+1]
			i++
		case "--tls-key":
			if i+1 >= len(args) {
				return errors.New("--tls-key requires a value")
			}
			tlsKey = args[i+1]
			i++
		case "--tls-ca":
			if i+1 >= len(args) {
				return errors.New("--tls-ca requires a value")
			}
			tlsCA = args[i+1]
			i++
		default:
			return fmt.Errorf("unknown flag: %s", args[i])
		}
	}

	if coordAddr == "" {
		return errors.New("--coordinator-address is required")
	}
	if workerAddr == "" {
		return errors.New("--worker-address is required")
	}
	if workerID == "" {
		hostname, err := os.Hostname()
		if err != nil {
			workerID = workerAddr
		} else {
			workerID = hostname
		}
	}
	// workerID is available for future use (e.g. distinct worker IDs).
	_ = workerID

	tlsConfig, err := buildTLSConfig(tlsCert, tlsKey, tlsCA)
	if err != nil {
		return err
	}

	node := c.newWorkerNode(distributed.WorkerNodeConfig{
		WorkerAddress:      workerAddr,
		CoordinatorAddress: coordAddr,
		WorldSize:          worldSize,
		TLS:                tlsConfig,
	})

	if err := node.Start(ctx); err != nil {
		return fmt.Errorf("failed to start worker: %w", err)
	}

	if c.shutdownCoord != nil {
		c.shutdownCoord.Register(node)
	}

	// Block until the context is canceled (signal received).
	<-ctx.Done()
	return nil
}

// Usage implements Command.Usage.
func (c *WorkerCommand) Usage() string {
	return `worker [OPTIONS]

Start a distributed training worker.

Binding --worker-address to anything other than a loopback address
(127.0.0.0/8, ::1, or "localhost") requires TLS -- see worker_node.go's
isLoopback / Start for the enforced contract. Single-host development can
keep the default loopback address and skip --tls-*; a multi-host run MUST
set --tls-cert/--tls-key/--tls-ca or Start will refuse to bind.

OPTIONS:
  --coordinator-address <addr>  Coordinator gRPC address (required)
  --worker-address <addr>       Worker gRPC listen address (required;
                                 default example: 127.0.0.1:9001)
  --worker-id <id>              Worker identifier (default: hostname)
  --world-size <n>              Total number of workers (default: auto)
  --tls-cert <path>             PEM certificate for this worker's gRPC
                                 server (and its outbound mTLS dial to the
                                 coordinator). Requires --tls-key and
                                 --tls-ca.
  --tls-key <path>              PEM private key matching --tls-cert.
                                 Requires --tls-cert and --tls-ca.
  --tls-ca <path>               PEM CA certificate used to verify peer
                                 certificates (client certs on the worker's
                                 server side, the coordinator's cert on the
                                 outbound dial). Requires --tls-cert and
                                 --tls-key.

Generating development certificates:

  A production deployment should issue certificates from a real CA. For
  local/dev multi-host testing, a self-signed CA plus a server cert works:

    # CA key + self-signed CA cert
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
      -keyout ca-key.pem -out ca.pem -days 365 -nodes \
      -subj "/O=zerfoo-dev-ca"

    # Worker key + CSR, signed by the dev CA (add worker's real host/IP to
    # -addext as needed)
    openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
      -keyout worker-key.pem -out worker.csr -nodes \
      -subj "/O=zerfoo-dev-worker" \
      -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
    openssl x509 -req -in worker.csr -CA ca.pem -CAkey ca-key.pem \
      -CAcreateserial -out worker-cert.pem -days 365 \
      -copy_extensions copyall

  Then run:

    worker --coordinator-address <coord-host>:9000 \
      --worker-address 0.0.0.0:9001 \
      --tls-cert worker-cert.pem --tls-key worker-key.pem --tls-ca ca.pem

  Repeat the CSR/sign step per host so each worker and the coordinator get
  their own cert signed by the same CA. See distributed/tlsconfig_test.go's
  generateTestCerts for the equivalent generated in-process for unit tests
  (self-signed, 1h expiry -- test-only, not for real deployments).`
}

// Examples implements Command.Examples.
func (c *WorkerCommand) Examples() []string {
	return []string{
		`worker --coordinator-address 127.0.0.1:9000 --worker-address 127.0.0.1:9001`,
		`worker --coordinator-address 10.0.0.1:9000 --worker-address 0.0.0.0:9001 --world-size 4 --tls-cert worker-cert.pem --tls-key worker-key.pem --tls-ca ca.pem`,
	}
}

// buildTLSConfig validates the --tls-* flag combination and constructs a
// distributed.TLSConfig from them. All three of cert, key, and ca must be
// provided together (mTLS needs a CA to verify peers); providing none is
// valid (TLS stays nil for loopback-only dev usage); providing a strict
// subset is a usage error since the resulting configuration mismatches
// what worker_node.go's Start() and tlsconfig.go's ServerCredentials/
// ClientCredentials expect.
func buildTLSConfig(certPath, keyPath, caPath string) (*distributed.TLSConfig, error) {
	switch {
	case certPath == "" && keyPath == "" && caPath == "":
		return nil, nil
	case certPath == "" || keyPath == "" || caPath == "":
		return nil, errors.New("--tls-cert, --tls-key, and --tls-ca must all be provided together")
	default:
		return &distributed.TLSConfig{
			CACertPath: caPath,
			CertPath:   certPath,
			KeyPath:    keyPath,
		}, nil
	}
}

// parsePositiveInt parses a string as a positive integer.
func parsePositiveInt(s string) (int, error) {
	var n int
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("invalid integer: %s", s)
		}
		n = n*10 + int(c-'0')
	}
	if n <= 0 {
		return 0, fmt.Errorf("must be positive: %s", s)
	}
	return n, nil
}

// Static interface assertion.
var _ Command = (*WorkerCommand)(nil)
