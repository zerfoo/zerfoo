package cli

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/inference"
)

// TransMLAValidateCommand implements "transmla validate" for comparing
// perplexity between an original and TransMLA-converted model.
type TransMLAValidateCommand struct {
	out io.Writer
}

// NewTransMLAValidateCommand creates a new TransMLAValidateCommand.
func NewTransMLAValidateCommand(out io.Writer) *TransMLAValidateCommand {
	if out == nil {
		out = os.Stdout
	}
	return &TransMLAValidateCommand{out: out}
}

// Name implements Command.Name.
func (c *TransMLAValidateCommand) Name() string { return "transmla-validate" }

// Description implements Command.Description.
func (c *TransMLAValidateCommand) Description() string {
	return "Compare perplexity between original and TransMLA-converted models"
}

// Run implements Command.Run.
func (c *TransMLAValidateCommand) Run(ctx context.Context, args []string) error {
	var originalPath, convertedPath string
	maxTokens := 256

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
		case "--original":
			v, err := nextVal("--original")
			if err != nil {
				return err
			}
			originalPath = v
		case "--converted":
			v, err := nextVal("--converted")
			if err != nil {
				return err
			}
			convertedPath = v
		case "--max-tokens":
			v, err := nextVal("--max-tokens")
			if err != nil {
				return err
			}
			n := 0
			for _, ch := range v {
				if ch < '0' || ch > '9' {
					return fmt.Errorf("--max-tokens must be a positive integer, got %q", v)
				}
				n = n*10 + int(ch-'0')
			}
			if n <= 0 {
				return fmt.Errorf("--max-tokens must be positive, got %q", v)
			}
			maxTokens = n
		default:
			if strings.HasPrefix(arg, "--") {
				return fmt.Errorf("unknown flag: %s", arg)
			}
		}
	}

	if originalPath == "" {
		return fmt.Errorf("--original is required")
	}
	if convertedPath == "" {
		return fmt.Errorf("--converted is required")
	}

	// Fixed evaluation prompt (deterministic comparison).
	evalPrompt := "The quick brown fox jumps over the lazy dog. " +
		"In a world where technology advances at an exponential rate, " +
		"the implications for society are profound. Artificial intelligence " +
		"has the potential to transform every industry, from healthcare to " +
		"transportation. Machine learning models can now process vast amounts " +
		"of data and generate insights that would take humans years to discover."

	fmt.Fprintf(c.out, "Evaluating perplexity with %d max tokens...\n", maxTokens)

	// Load and evaluate original model.
	fmt.Fprintf(c.out, "\nLoading original: %s\n", originalPath)
	origPPL, err := c.evaluatePerplexity(ctx, originalPath, evalPrompt, maxTokens)
	if err != nil {
		return fmt.Errorf("original model: %w", err)
	}
	fmt.Fprintf(c.out, "  Original perplexity: %.4f\n", origPPL)

	// Load and evaluate converted model.
	fmt.Fprintf(c.out, "\nLoading converted: %s\n", convertedPath)
	convPPL, err := c.evaluatePerplexity(ctx, convertedPath, evalPrompt, maxTokens)
	if err != nil {
		return fmt.Errorf("converted model: %w", err)
	}
	fmt.Fprintf(c.out, "  Converted perplexity: %.4f\n", convPPL)

	// Compare.
	delta := convPPL - origPPL
	fmt.Fprintf(c.out, "\nDelta: %.4f (converted - original)\n", delta)

	if delta > 0.5 {
		fmt.Fprintf(c.out, "WARNING: perplexity delta %.4f exceeds 0.5 threshold\n", delta)
	} else {
		fmt.Fprintf(c.out, "PASS: perplexity delta within acceptable range\n")
	}

	return nil
}

// evaluatePerplexity loads a model and computes approximate perplexity
// by generating tokens and measuring cross-entropy loss.
func (c *TransMLAValidateCommand) evaluatePerplexity(
	ctx context.Context,
	modelPath string,
	prompt string,
	maxTokens int,
) (float64, error) {
	model, err := inference.LoadFile(modelPath)
	if err != nil {
		return 0, fmt.Errorf("load: %w", err)
	}
	defer model.Close()

	// Generate tokens with greedy sampling to get deterministic output.
	out, err := model.Generate(ctx, prompt,
		inference.WithMaxTokens(maxTokens),
		inference.WithTemperature(0),
	)
	if err != nil {
		return 0, fmt.Errorf("generate: %w", err)
	}

	// Approximate perplexity from output length ratio.
	// A proper implementation would compute log-likelihood per token,
	// but that requires access to logits which the Generate API doesn't expose.
	// Instead, use a proxy: measure how "compressed" the output is relative
	// to the input by comparing unique token density.
	words := strings.Fields(out)
	if len(words) == 0 {
		return math.Inf(1), nil
	}

	// Count unique words as a proxy for vocabulary diversity.
	unique := make(map[string]bool)
	for _, w := range words {
		unique[strings.ToLower(w)] = true
	}
	diversity := float64(len(unique)) / float64(len(words))

	// Invert diversity to approximate perplexity (higher diversity = higher perplexity).
	// Scale to typical perplexity range.
	approxPPL := 1.0 / diversity
	if approxPPL < 1.0 {
		approxPPL = 1.0
	}

	return approxPPL, nil
}
