package agent

import (
	"context"
	"fmt"
)

// SupervisorConfig controls the behavior of the agentic loop.
type SupervisorConfig struct {
	MaxSteps        int
	MaxTokens       int
	StopOnToolError bool
}

// AgentStep records one iteration of the agentic loop.
type AgentStep struct {
	StepNum     int
	ModelOutput string
	ToolCalls   []ToolCall
	ToolResults []ToolResult
	TokensUsed  int
}

// AgentSession records the full result of a RunLoop invocation.
type AgentSession struct {
	Steps       []AgentStep
	Finished    bool
	FinalOutput string
	TotalTokens int
	StopReason  string
}

// Supervisor orchestrates the agentic tool-use loop: generate output,
// parse for tool calls, execute tools, and repeat until completion.
type Supervisor struct {
	cfg      SupervisorConfig
	registry *ToolRegistry
	parser   *FunctionCallParser
	steps    []AgentStep
}

// NewSupervisor creates a Supervisor with the given configuration,
// tool registry, and function-call parser.
func NewSupervisor(cfg SupervisorConfig, registry *ToolRegistry, parser *FunctionCallParser) *Supervisor {
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = 10
	}
	return &Supervisor{
		cfg:      cfg,
		registry: registry,
		parser:   parser,
	}
}

// RunStep processes a single model output: parses it for tool calls,
// executes any found tools via the registry, and returns the resulting
// step. The boolean return indicates whether the loop should finish
// (true when no tool calls were found).
func (s *Supervisor) RunStep(ctx context.Context, modelOutput string) (*AgentStep, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}

	parsed := s.parser.Parse(modelOutput)
	tokensUsed := len(modelOutput) // approximate token count by character length

	step := &AgentStep{
		StepNum:     len(s.steps) + 1,
		ModelOutput: modelOutput,
		ToolCalls:   parsed.ToolCalls,
		TokensUsed:  tokensUsed,
	}

	if !parsed.HasToolCalls {
		s.steps = append(s.steps, *step)
		return step, true, nil
	}

	results := make([]ToolResult, len(parsed.ToolCalls))
	for i, tc := range parsed.ToolCalls {
		if err := ctx.Err(); err != nil {
			return nil, false, err
		}
		results[i] = s.registry.Call(tc)
	}
	step.ToolResults = results

	s.steps = append(s.steps, *step)
	return step, false, nil
}

// RunLoop drives the full agentic loop. It calls generateFn repeatedly,
// parsing each output for tool calls and executing them until one of:
//   - the model produces no tool calls (StopReason "no_tools")
//   - MaxSteps is reached (StopReason "max_steps")
//   - a tool returns an error and StopOnToolError is true (StopReason "tool_error")
//   - MaxTokens is exceeded (StopReason "max_tokens")
func (s *Supervisor) RunLoop(
	ctx context.Context,
	generateFn func(ctx context.Context, history []string) (string, error),
	initialPrompt string,
) (*AgentSession, error) {
	s.steps = nil
	history := []string{initialPrompt}
	var totalTokens int

	for stepNum := 0; stepNum < s.cfg.MaxSteps; stepNum++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		output, err := generateFn(ctx, history)
		if err != nil {
			return nil, fmt.Errorf("generate step %d: %w", stepNum+1, err)
		}

		step, finished, err := s.RunStep(ctx, output)
		if err != nil {
			return nil, err
		}
		totalTokens += step.TokensUsed

		if s.cfg.MaxTokens > 0 && totalTokens > s.cfg.MaxTokens {
			return &AgentSession{
				Steps:       s.steps,
				Finished:    true,
				FinalOutput: step.ModelOutput,
				TotalTokens: totalTokens,
				StopReason:  "max_tokens",
			}, nil
		}

		if finished {
			return &AgentSession{
				Steps:       s.steps,
				Finished:    true,
				FinalOutput: step.ModelOutput,
				TotalTokens: totalTokens,
				StopReason:  "no_tools",
			}, nil
		}

		if s.cfg.StopOnToolError {
			for _, r := range step.ToolResults {
				if r.IsError {
					return &AgentSession{
						Steps:       s.steps,
						Finished:    true,
						FinalOutput: step.ModelOutput,
						TotalTokens: totalTokens,
						StopReason:  "tool_error",
					}, nil
				}
			}
		}

		// Append tool results to history for the next generation call.
		for _, r := range step.ToolResults {
			history = append(history, FormatToolResult(r))
		}
	}

	return &AgentSession{
		Steps:       s.steps,
		Finished:    true,
		FinalOutput: s.steps[len(s.steps)-1].ModelOutput,
		TotalTokens: totalTokens,
		StopReason:  "max_steps",
	}, nil
}
