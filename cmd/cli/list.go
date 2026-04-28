package cli

import (
	"context"
	"fmt"
	"io"

	"github.com/zerfoo/zerfoo/model/registry"
)

// ListCommand implements the "list" CLI command for showing cached models.
type ListCommand struct {
	reg registry.ModelRegistry
	out io.Writer
}

// NewListCommand creates a new ListCommand. If reg is nil, a default
// LocalRegistry will be created when the command runs.
func NewListCommand(reg registry.ModelRegistry, out io.Writer) *ListCommand {
	return &ListCommand{reg: reg, out: out}
}

// Name implements Command.Name.
func (c *ListCommand) Name() string { return "list" }

// Description implements Command.Description.
func (c *ListCommand) Description() string {
	return "List cached models"
}

// Run implements Command.Run.
func (c *ListCommand) Run(_ context.Context, args []string) error {
	var cacheDir string

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--cache-dir":
			if i+1 >= len(args) {
				return fmt.Errorf("--cache-dir requires a value")
			}
			cacheDir = args[i+1]
			i++
		default:
			return fmt.Errorf("unexpected argument: %s", args[i])
		}
	}

	reg := c.reg
	if reg == nil {
		lr, err := registry.NewLocalRegistry(cacheDir)
		if err != nil {
			return fmt.Errorf("create registry: %w", err)
		}
		reg = lr
	}

	models := reg.List()
	if len(models) == 0 {
		_, _ = fmt.Fprintln(c.out, "No cached models.")
		return nil
	}

	_, _ = fmt.Fprintf(c.out, "%-30s %-30s %s\n", "REPO", "FILE", "SIZE")
	for _, m := range models {
		_, _ = fmt.Fprintf(c.out, "%-30s %-30s %s\n", m.ID, m.Path, formatSize(m.Size))
	}
	return nil
}

// formatSize returns a human-readable size string.
func formatSize(b int64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(b)/float64(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(b)/float64(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(b)/float64(1<<10))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

// Usage implements Command.Usage.
func (c *ListCommand) Usage() string {
	return `list [OPTIONS]

List all locally cached models with sizes.

OPTIONS:
  --cache-dir <dir>  Override default cache directory`
}

// Examples implements Command.Examples.
func (c *ListCommand) Examples() []string {
	return []string{
		"list",
		"list --cache-dir /data/models",
	}
}

// Static interface assertion.
var _ Command = (*ListCommand)(nil)
