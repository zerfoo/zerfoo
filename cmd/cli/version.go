package cli

import (
	"context"
	"fmt"
	"io"
	"os"
)

// VersionCommand implements the "version" CLI command.
type VersionCommand struct {
	version string
	out     io.Writer
}

// NewVersionCommand creates a new VersionCommand. The version string is
// typically set at build time via -ldflags "-X main.version=v1.2.3".
// If version is empty, "(devel)" is displayed.
func NewVersionCommand(version string, out io.Writer) *VersionCommand {
	if out == nil {
		out = os.Stdout
	}
	v := version
	if v == "" {
		v = "(devel)"
	}
	return &VersionCommand{version: v, out: out}
}

// Name implements Command.Name.
func (c *VersionCommand) Name() string { return "version" }

// Description implements Command.Description.
func (c *VersionCommand) Description() string {
	return "Print the Zerfoo version"
}

// Run implements Command.Run.
func (c *VersionCommand) Run(_ context.Context, _ []string) error {
	fmt.Fprintf(c.out, "zerfoo version %s\n", c.version)
	return nil
}

// Usage implements Command.Usage.
func (c *VersionCommand) Usage() string {
	return "version\n\nPrint the Zerfoo version."
}

// Examples implements Command.Examples.
func (c *VersionCommand) Examples() []string {
	return []string{"version"}
}

// Static interface assertion.
var _ Command = (*VersionCommand)(nil)
