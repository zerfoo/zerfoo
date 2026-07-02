package cli

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/zerfoo/zerfoo/model/registry"
)

// PullCommand implements the "pull" CLI command for downloading
// and caching a model from a remote registry.
type PullCommand struct {
	reg registry.ModelRegistry
	out io.Writer
}

// NewPullCommand creates a new PullCommand. If reg is nil, a default
// LocalRegistry will be created when the command runs.
func NewPullCommand(reg registry.ModelRegistry, out io.Writer) *PullCommand {
	return &PullCommand{reg: reg, out: out}
}

// Name implements Command.Name.
func (c *PullCommand) Name() string { return "pull" }

// Description implements Command.Description.
func (c *PullCommand) Description() string {
	return "Download and cache a model"
}

// Run implements Command.Run.
func (c *PullCommand) Run(ctx context.Context, args []string) error {
	var cacheDir, modelID, quant string

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--cache-dir":
			if i+1 >= len(args) {
				return errors.New("--cache-dir requires a value")
			}
			cacheDir = args[i+1]
			i++
		case "--quant":
			if i+1 >= len(args) {
				return errors.New("--quant requires a value")
			}
			quant = args[i+1]
			i++
		default:
			if modelID != "" {
				return fmt.Errorf("unexpected argument: %s", args[i])
			}
			modelID = args[i]
		}
	}

	if modelID == "" {
		return errors.New("model ID is required")
	}

	// Default quantization.
	if quant == "" {
		quant = "Q4_K_M"
	}

	reg := c.reg
	if reg == nil {
		lr, err := registry.NewLocalRegistry(cacheDir)
		if err != nil {
			return fmt.Errorf("create registry: %w", err)
		}
		progress := newProgressDisplay(c.out, isTTY(c.out))
		lr.SetPullFunc(registry.NewHFPullFunc(registry.HFPullOptions{
			Quant:      quant,
			OnProgress: progress.callback,
		}))
		reg = lr
	}

	// Check if already cached.
	if info, ok := reg.Get(modelID); ok {
		_, _ = fmt.Fprintf(c.out, "Already up to date: %s\n", info.Path)
		return nil
	}

	_, _ = fmt.Fprintf(c.out, "Pulling %s (quant: %s)...\n", modelID, quant)
	info, err := reg.Pull(ctx, modelID)
	if err != nil {
		return fmt.Errorf("pull %q: %w", modelID, err)
	}

	_, _ = fmt.Fprintf(c.out, "Model saved to: %s\n", info.Path)
	_, _ = fmt.Fprintf(c.out, "Size: %d bytes\n", info.Size)
	return nil
}

// Usage implements Command.Usage.
func (c *PullCommand) Usage() string {
	return `pull [OPTIONS] <model-id>

Download and cache a model from a remote registry.

OPTIONS:
  --quant <type>     Quantization type (default: Q4_K_M)
  --cache-dir <dir>  Override default cache directory`
}

// Examples implements Command.Examples.
func (c *PullCommand) Examples() []string {
	return []string{
		"pull google/gemma-3-4b",
		"pull google/gemma-3-4b --quant Q8_0",
		"pull google/gemma-3-4b --cache-dir /data/models",
	}
}

// Static interface assertion.
var _ Command = (*PullCommand)(nil)
