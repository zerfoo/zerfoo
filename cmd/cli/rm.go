package cli

import (
	"context"
	"fmt"
	"io"

	"github.com/zerfoo/zerfoo/model/registry"
)

// RmCommand implements the "rm" CLI command for removing cached models.
type RmCommand struct {
	reg registry.ModelRegistry
	out io.Writer
}

// NewRmCommand creates a new RmCommand. If reg is nil, a default
// LocalRegistry will be created when the command runs.
func NewRmCommand(reg registry.ModelRegistry, out io.Writer) *RmCommand {
	return &RmCommand{reg: reg, out: out}
}

// Name implements Command.Name.
func (c *RmCommand) Name() string { return "rm" }

// Description implements Command.Description.
func (c *RmCommand) Description() string {
	return "Remove a cached model"
}

// Run implements Command.Run.
func (c *RmCommand) Run(_ context.Context, args []string) error {
	var cacheDir, modelID string

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--cache-dir":
			if i+1 >= len(args) {
				return fmt.Errorf("--cache-dir requires a value")
			}
			cacheDir = args[i+1]
			i++
		default:
			if modelID != "" {
				return fmt.Errorf("unexpected argument: %s", args[i])
			}
			modelID = args[i]
		}
	}

	if modelID == "" {
		return fmt.Errorf("model ID is required")
	}

	reg := c.reg
	if reg == nil {
		lr, err := registry.NewLocalRegistry(cacheDir)
		if err != nil {
			return fmt.Errorf("create registry: %w", err)
		}
		reg = lr
	}

	if err := reg.Delete(modelID); err != nil {
		return fmt.Errorf("remove %q: %w", modelID, err)
	}

	_, _ = fmt.Fprintf(c.out, "Removed %s\n", modelID)
	return nil
}

// Usage implements Command.Usage.
func (c *RmCommand) Usage() string {
	return `rm [OPTIONS] <model-id>

Remove a cached model and its files.

OPTIONS:
  --cache-dir <dir>  Override default cache directory`
}

// Examples implements Command.Examples.
func (c *RmCommand) Examples() []string {
	return []string{
		"rm google/gemma-3-4b",
	}
}

// Static interface assertion.
var _ Command = (*RmCommand)(nil)
