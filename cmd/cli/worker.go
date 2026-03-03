package cli

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/distributed"
	"github.com/zerfoo/zerfoo/shutdown"
)

// WorkerCommand implements the "worker" CLI command for starting a
// distributed training worker.
type WorkerCommand struct {
	shutdownCoord *shutdown.Coordinator
}

// NewWorkerCommand creates a new WorkerCommand. The shutdown coordinator
// is used to register the worker node for orderly shutdown on signal.
func NewWorkerCommand(coord *shutdown.Coordinator) *WorkerCommand {
	return &WorkerCommand{shutdownCoord: coord}
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

	node := distributed.NewWorkerNode(distributed.WorkerNodeConfig{
		WorkerAddress:      workerAddr,
		CoordinatorAddress: coordAddr,
		WorldSize:          worldSize,
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

OPTIONS:
  --coordinator-address <addr>  Coordinator gRPC address (required)
  --worker-address <addr>       Worker gRPC listen address (required)
  --worker-id <id>              Worker identifier (default: hostname)
  --world-size <n>              Total number of workers (default: auto)`
}

// Examples implements Command.Examples.
func (c *WorkerCommand) Examples() []string {
	return []string{
		`worker --coordinator-address localhost:9000 --worker-address localhost:9001`,
		`worker --coordinator-address 10.0.0.1:9000 --worker-address 0.0.0.0:9001 --world-size 4`,
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
