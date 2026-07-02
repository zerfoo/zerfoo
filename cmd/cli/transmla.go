package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/inference/transmla"
)

// TransMLACommand implements the "transmla" CLI command for converting
// standard MHA weights to multi-head latent attention (MLA) form.
type TransMLACommand struct {
	out io.Writer
}

// NewTransMLACommand creates a new TransMLACommand that writes status
// output to out.
func NewTransMLACommand(out io.Writer) *TransMLACommand {
	if out == nil {
		out = os.Stdout
	}
	return &TransMLACommand{out: out}
}

// Name implements Command.Name.
func (c *TransMLACommand) Name() string { return "transmla" }

// Description implements Command.Description.
func (c *TransMLACommand) Description() string {
	return "Convert MHA GGUF weights to multi-head latent attention (MLA) via truncated SVD"
}

// Run implements Command.Run.
func (c *TransMLACommand) Run(_ context.Context, args []string) error {
	var inputPath, outputPath string
	rank := 512

	for i := 0; i < len(args); i++ {
		arg := args[i]
		var eqVal string
		var hasEq bool
		if flag, val, ok := splitFlag(arg); ok {
			arg = flag
			eqVal = val
			hasEq = true
		}
		nextVal := func(flagName string) (string, error) {
			if hasEq {
				return eqVal, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", flagName)
			}
			i++
			return args[i], nil
		}
		switch arg {
		case "--input":
			v, err := nextVal("--input")
			if err != nil {
				return err
			}
			inputPath = v
		case "--output":
			v, err := nextVal("--output")
			if err != nil {
				return err
			}
			outputPath = v
		case "--rank":
			v, err := nextVal("--rank")
			if err != nil {
				return err
			}
			n := 0
			for _, ch := range v {
				if ch < '0' || ch > '9' {
					return fmt.Errorf("--rank must be a positive integer, got %q", v)
				}
				n = n*10 + int(ch-'0')
			}
			if n <= 0 {
				return fmt.Errorf("--rank must be a positive integer, got %q", v)
			}
			rank = n
		default:
			if strings.HasPrefix(arg, "--") {
				return fmt.Errorf("unknown flag: %s", arg)
			}
		}
	}

	if inputPath == "" {
		return fmt.Errorf("--input is required")
	}
	if outputPath == "" {
		return fmt.Errorf("--output is required")
	}

	srcFile, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("open input: %w", err)
	}
	defer srcFile.Close() //nolint:errcheck

	srcInfo, err := srcFile.Stat()
	if err != nil {
		return fmt.Errorf("stat input: %w", err)
	}
	srcSize := srcInfo.Size()

	dstFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}
	defer dstFile.Close() //nolint:errcheck

	fmt.Fprintf(c.out, "Converting %s -> %s (rank=%d)\n", inputPath, outputPath, rank)

	opts := transmla.ConvertGGUFOptions{
		Rank: rank,
		OnLayerDone: func(layer, total int) {
			fmt.Fprintf(c.out, "  layer %d/%d decomposed\n", layer+1, total)
		},
	}

	if err := transmla.ConvertGGUF(srcFile, dstFile, opts); err != nil {
		// Clean up partial output on failure.
		dstFile.Close() //nolint:errcheck
		os.Remove(outputPath)
		return fmt.Errorf("conversion failed: %w", err)
	}

	// Report compression ratio.
	dstInfo, err := dstFile.Stat()
	if err != nil {
		return fmt.Errorf("stat output: %w", err)
	}
	dstSize := dstInfo.Size()

	ratio := float64(srcSize) / float64(dstSize)
	fmt.Fprintf(c.out, "Done. %s -> %s (%.2fx compression)\n",
		formatBytes(srcSize), formatBytes(dstSize), ratio)

	return nil
}

// Usage implements Command.Usage.
func (c *TransMLACommand) Usage() string {
	return `transmla [OPTIONS]

Convert MHA GGUF weights to multi-head latent attention (MLA) via truncated SVD.

OPTIONS:
  --input <path>     Input GGUF file path (required)
  --output <path>    Output GGUF file path (required)
  --rank <int>       Latent dimension / SVD rank (default: 512)`
}

// Examples implements Command.Examples.
func (c *TransMLACommand) Examples() []string {
	return []string{
		"transmla --input model.gguf --output model-mla.gguf",
		"transmla --rank 256 --input model.gguf --output model-mla.gguf",
	}
}

// Static interface assertion.
var _ Command = (*TransMLACommand)(nil)
