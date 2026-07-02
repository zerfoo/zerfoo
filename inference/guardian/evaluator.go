package guardian

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/inference"
)

// GuardianRequest specifies content and risk categories to evaluate.
type GuardianRequest struct {
	Input  GuardianInput // content to evaluate
	Risks  []string      // risk categories to check (default: all harm risks)
	Format string        // output format: "3.0", "3.2", "3.3"
	Think  bool          // enable thinking (3.3 only)
}

// EvaluatorOption configures an Evaluator.
type EvaluatorOption func(*evaluatorOptions)

type evaluatorOptions struct {
	format   string
	device   string
	loadOpts []inference.Option
}

// WithDefaultFormat sets the default output format for the evaluator.
func WithDefaultFormat(format string) EvaluatorOption {
	return func(o *evaluatorOptions) {
		o.format = format
	}
}

// WithEvaluatorDevice sets the compute device for the evaluator model.
func WithEvaluatorDevice(device string) EvaluatorOption {
	return func(o *evaluatorOptions) {
		o.device = device
	}
}

// WithLoadOptions passes additional model loading options to the evaluator.
func WithLoadOptions(opts ...inference.Option) EvaluatorOption {
	return func(o *evaluatorOptions) {
		o.loadOpts = opts
	}
}

// ModelGenerator is the interface required by the Evaluator for text generation.
// This abstraction allows testing without loading a real model.
type ModelGenerator interface {
	Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) (string, error)
}

// Evaluator orchestrates Guardian safety evaluation by rendering prompts,
// running constrained generation, and parsing verdicts.
type Evaluator struct {
	model  ModelGenerator
	format string // default format
}

// NewEvaluator creates a new Evaluator by loading a Granite Guardian model
// from the given path.
func NewEvaluator(modelPath string, opts ...EvaluatorOption) (*Evaluator, error) {
	o := &evaluatorOptions{
		format: "3.2",
		device: "cpu",
	}
	for _, opt := range opts {
		opt(o)
	}

	loadOpts := append([]inference.Option{inference.WithDevice(o.device)}, o.loadOpts...)
	model, err := inference.LoadFile(modelPath, loadOpts...)
	if err != nil {
		return nil, fmt.Errorf("guardian: load model: %w", err)
	}

	return &Evaluator{
		model:  model,
		format: o.format,
	}, nil
}

// NewEvaluatorFromModel creates an Evaluator from a pre-loaded model.
// This is useful for testing or when the model is already loaded.
func NewEvaluatorFromModel(model ModelGenerator, opts ...EvaluatorOption) *Evaluator {
	o := &evaluatorOptions{
		format: "3.2",
	}
	for _, opt := range opts {
		opt(o)
	}
	return &Evaluator{
		model:  model,
		format: o.format,
	}
}

// Evaluate checks content against specified risk categories.
// For each risk category, it renders a prompt template, runs constrained
// generation (MaxTokens=50, Temperature=0), and parses the output into a Verdict.
func (e *Evaluator) Evaluate(ctx context.Context, req GuardianRequest) ([]Verdict, error) {
	risks := req.Risks
	if len(risks) == 0 {
		risks = HarmRiskCategories()
	}

	format := req.Format
	if format == "" {
		format = e.format
	}

	verdicts := make([]Verdict, 0, len(risks))
	for _, risk := range risks {
		select {
		case <-ctx.Done():
			return verdicts, ctx.Err()
		default:
		}

		prompt, err := RenderTemplate(req.Input, TemplateOptions{
			Risk:   risk,
			Format: format,
			Think:  req.Think,
		})
		if err != nil {
			return nil, fmt.Errorf("guardian: render template for %q: %w", risk, err)
		}

		output, err := e.model.Generate(ctx, prompt,
			inference.WithMaxTokens(50),
			inference.WithTemperature(0),
		)
		if err != nil {
			return nil, fmt.Errorf("guardian: generate for %q: %w", risk, err)
		}

		verdict := ParseVerdict(output, risk, nil)
		verdicts = append(verdicts, verdict)
	}

	return verdicts, nil
}

// BatchResult holds evaluation results for multiple inputs.
type BatchResult struct {
	Results []InputResult
}

// InputResult holds verdicts for a single input.
type InputResult struct {
	Index    int       // original index in the input slice
	Verdicts []Verdict // one per risk category
	Flagged  bool      // true if any verdict is Unsafe
}

// EvaluateBatch evaluates multiple inputs sequentially against the specified
// risk categories. Results are returned in the same order as the inputs.
func (e *Evaluator) EvaluateBatch(ctx context.Context, inputs []GuardianInput, risks []string) (*BatchResult, error) {
	result := &BatchResult{
		Results: make([]InputResult, len(inputs)),
	}

	for i, input := range inputs {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		default:
		}

		verdicts, err := e.Evaluate(ctx, GuardianRequest{
			Input: input,
			Risks: risks,
		})
		if err != nil {
			return nil, fmt.Errorf("guardian: batch input %d: %w", i, err)
		}

		flagged := false
		for _, v := range verdicts {
			if v.Unsafe {
				flagged = true
				break
			}
		}

		result.Results[i] = InputResult{
			Index:    i,
			Verdicts: verdicts,
			Flagged:  flagged,
		}
	}

	return result, nil
}

// ScanResult holds the aggregate result of scanning content against all
// harm-related risk categories.
type ScanResult struct {
	Flagged     bool      // true if any risk detected
	Verdicts    []Verdict // all individual verdicts
	HighestRisk string    // category with highest confidence unsafe
}

// Scan evaluates the input against all harm-related risk categories and
// returns an aggregate result.
func (e *Evaluator) Scan(ctx context.Context, input GuardianInput) (*ScanResult, error) {
	verdicts, err := e.Evaluate(ctx, GuardianRequest{
		Input: input,
		Risks: HarmRiskCategories(),
	})
	if err != nil {
		return nil, err
	}

	result := &ScanResult{
		Verdicts: verdicts,
	}

	var highestConf float64
	for _, v := range verdicts {
		if v.Unsafe {
			result.Flagged = true
			if result.HighestRisk == "" || v.Confidence > highestConf {
				result.HighestRisk = v.Risk
				highestConf = v.Confidence
			}
		}
	}

	return result, nil
}
