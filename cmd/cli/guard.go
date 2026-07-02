package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/inference/guardian"
)

// guardConfig holds parsed CLI flags for the guard command.
type guardConfig struct {
	model    string
	input    string
	response string
	file     string
	risks    []string
	scan     bool
	jsonOut  bool
	device   string
}

// ModelLoaderFunc loads a Guardian model from a path and returns an Evaluator.
// This abstraction allows testing without loading a real model.
type ModelLoaderFunc func(modelPath string, opts ...guardian.EvaluatorOption) (*guardian.Evaluator, error)

// GuardCommand implements the "guard" CLI command for content moderation.
type GuardCommand struct {
	out         io.Writer
	modelLoader ModelLoaderFunc
}

// NewGuardCommand creates a new GuardCommand that writes output to out.
func NewGuardCommand(out io.Writer) *GuardCommand {
	if out == nil {
		out = os.Stdout
	}
	return &GuardCommand{
		out:         out,
		modelLoader: guardian.NewEvaluator,
	}
}

// NewGuardCommandWithLoader creates a GuardCommand with a custom model loader
// for testing.
func NewGuardCommandWithLoader(out io.Writer, loader ModelLoaderFunc) *GuardCommand {
	if out == nil {
		out = os.Stdout
	}
	return &GuardCommand{
		out:         out,
		modelLoader: loader,
	}
}

// Name implements Command.Name.
func (c *GuardCommand) Name() string { return "guard" }

// Description implements Command.Description.
func (c *GuardCommand) Description() string {
	return "Evaluate content safety using Granite Guardian"
}

// Run implements Command.Run.
func (c *GuardCommand) Run(ctx context.Context, args []string) error {
	cfg, err := c.parseArgs(args)
	if err != nil {
		return err
	}

	// Resolve input text.
	inputText := cfg.input
	if cfg.file != "" {
		data, readErr := os.ReadFile(cfg.file)
		if readErr != nil {
			return fmt.Errorf("failed to read input file: %w", readErr)
		}
		inputText = string(data)
	}

	if inputText == "" {
		return fmt.Errorf("--input or --file is required")
	}

	if cfg.model == "" {
		return fmt.Errorf("--model is required")
	}

	// Build evaluator options.
	var evalOpts []guardian.EvaluatorOption
	if cfg.device != "" {
		evalOpts = append(evalOpts, guardian.WithEvaluatorDevice(cfg.device))
	}

	eval, err := c.modelLoader(cfg.model, evalOpts...)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	gi := guardian.GuardianInput{
		User:      inputText,
		Assistant: cfg.response,
	}

	if cfg.scan {
		return c.runScan(ctx, eval, gi, cfg)
	}

	return c.runEvaluate(ctx, eval, gi, cfg)
}

func (c *GuardCommand) runScan(ctx context.Context, eval *guardian.Evaluator, gi guardian.GuardianInput, cfg *guardConfig) error {
	result, err := eval.Scan(ctx, gi)
	if err != nil {
		return fmt.Errorf("scan failed: %w", err)
	}

	if cfg.jsonOut {
		return c.writeJSONScan(result)
	}
	return c.writeHumanScan(gi, result)
}

func (c *GuardCommand) runEvaluate(ctx context.Context, eval *guardian.Evaluator, gi guardian.GuardianInput, cfg *guardConfig) error {
	req := guardian.GuardianRequest{
		Input: gi,
		Risks: cfg.risks,
	}

	verdicts, err := eval.Evaluate(ctx, req)
	if err != nil {
		return fmt.Errorf("evaluation failed: %w", err)
	}

	if cfg.jsonOut {
		return c.writeJSONVerdicts(verdicts)
	}
	return c.writeHumanVerdicts(gi, cfg.model, verdicts)
}

// parseArgs parses guard command arguments.
func (c *GuardCommand) parseArgs(args []string) (*guardConfig, error) {
	cfg := &guardConfig{}

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
		case "--model":
			v, err := nextVal("--model")
			if err != nil {
				return nil, err
			}
			cfg.model = v
		case "--input":
			v, err := nextVal("--input")
			if err != nil {
				return nil, err
			}
			cfg.input = v
		case "--response":
			v, err := nextVal("--response")
			if err != nil {
				return nil, err
			}
			cfg.response = v
		case "--file":
			v, err := nextVal("--file")
			if err != nil {
				return nil, err
			}
			cfg.file = v
		case "--risks":
			v, err := nextVal("--risks")
			if err != nil {
				return nil, err
			}
			cfg.risks = parseRisks(v)
		case "--scan":
			cfg.scan = true
		case "--json":
			cfg.jsonOut = true
		case "--device":
			v, err := nextVal("--device")
			if err != nil {
				return nil, err
			}
			cfg.device = v
		}
	}

	return cfg, nil
}

// parseRisks splits a comma-separated risk list into individual category names.
func parseRisks(s string) []string {
	parts := strings.Split(s, ",")
	risks := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			risks = append(risks, p)
		}
	}
	return risks
}

// jsonVerdict is the JSON output format for a single verdict.
type jsonVerdict struct {
	Risk       string  `json:"risk"`
	Unsafe     bool    `json:"unsafe"`
	Confidence float64 `json:"confidence"`
}

// jsonOutput is the top-level JSON output format.
type jsonOutput struct {
	Flagged  bool          `json:"flagged"`
	Verdicts []jsonVerdict `json:"verdicts"`
}

func (c *GuardCommand) writeJSONVerdicts(verdicts []guardian.Verdict) error {
	flagged := false
	jvs := make([]jsonVerdict, len(verdicts))
	for i, v := range verdicts {
		if v.Unsafe {
			flagged = true
		}
		jvs[i] = jsonVerdict{
			Risk:       v.Risk,
			Unsafe:     v.Unsafe,
			Confidence: v.Confidence,
		}
	}

	out := jsonOutput{
		Flagged:  flagged,
		Verdicts: jvs,
	}

	data, err := json.Marshal(out)
	if err != nil {
		return err
	}
	fmt.Fprintln(c.out, string(data))
	return nil
}

func (c *GuardCommand) writeJSONScan(result *guardian.ScanResult) error {
	jvs := make([]jsonVerdict, len(result.Verdicts))
	for i, v := range result.Verdicts {
		jvs[i] = jsonVerdict{
			Risk:       v.Risk,
			Unsafe:     v.Unsafe,
			Confidence: v.Confidence,
		}
	}

	out := jsonOutput{
		Flagged:  result.Flagged,
		Verdicts: jvs,
	}

	data, err := json.Marshal(out)
	if err != nil {
		return err
	}
	fmt.Fprintln(c.out, string(data))
	return nil
}

func (c *GuardCommand) writeHumanVerdicts(gi guardian.GuardianInput, model string, verdicts []guardian.Verdict) error {
	fmt.Fprintf(c.out, "Model: %s\n", model)
	fmt.Fprintf(c.out, "Input: %q\n\n", gi.User)

	fmt.Fprintln(c.out, "Risk Assessment:")
	flagged := false
	for _, v := range verdicts {
		label := "safe"
		if v.Unsafe {
			label = "UNSAFE"
			flagged = true
		}
		fmt.Fprintf(c.out, "  %-30s %s (confidence: %.2f)\n", v.Risk+":", label, v.Confidence)
	}

	fmt.Fprintln(c.out)
	if flagged {
		fmt.Fprintln(c.out, "Overall: FLAGGED")
	} else {
		fmt.Fprintln(c.out, "Overall: SAFE")
	}
	return nil
}

func (c *GuardCommand) writeHumanScan(gi guardian.GuardianInput, result *guardian.ScanResult) error {
	fmt.Fprintf(c.out, "Input: %q\n\n", gi.User)

	fmt.Fprintln(c.out, "Risk Assessment (full scan):")
	for _, v := range result.Verdicts {
		label := "safe"
		if v.Unsafe {
			label = "UNSAFE"
		}
		fmt.Fprintf(c.out, "  %-30s %s (confidence: %.2f)\n", v.Risk+":", label, v.Confidence)
	}

	fmt.Fprintln(c.out)
	if result.Flagged {
		fmt.Fprintf(c.out, "Overall: FLAGGED (highest risk: %s)\n", result.HighestRisk)
	} else {
		fmt.Fprintln(c.out, "Overall: SAFE")
	}
	return nil
}

// Usage implements Command.Usage.
func (c *GuardCommand) Usage() string {
	return `guard [OPTIONS]

Evaluate content safety using Granite Guardian.

OPTIONS:
  --model <path>        Path to Guardian model file (required)
  --input <text>        Text to evaluate (required unless --file)
  --file <path>         Read input text from file
  --response <text>     Assistant response to evaluate
  --risks <list>        Comma-separated risk categories (default: all harm risks)
  --scan                Scan against all harm risk categories
  --json                Output results as JSON
  --device <device>     Compute device: cpu, cuda, cuda:N (default: cpu)`
}

// Examples implements Command.Examples.
func (c *GuardCommand) Examples() []string {
	return []string{
		`guard --model granite-guardian --input "How to hack a computer"`,
		`guard --model granite-guardian --input "text" --risks harm,jailbreaking,profanity`,
		`guard --model granite-guardian --input "text" --scan`,
		`guard --model granite-guardian --file input.txt`,
		`guard --model granite-guardian --input "user msg" --response "assistant msg"`,
		`guard --model granite-guardian --input "text" --json`,
	}
}

// Static interface assertion.
var _ Command = (*GuardCommand)(nil)
