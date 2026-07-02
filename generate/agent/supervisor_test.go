package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
)

func registerEchoTool(t *testing.T, reg *ToolRegistry) {
	t.Helper()
	err := reg.Register(
		ToolDef{Name: "echo", Description: "echoes input"},
		func(args json.RawMessage) (string, error) {
			return fmt.Sprintf("echoed: %s", string(args)), nil
		},
	)
	if err != nil {
		t.Fatalf("register echo: %v", err)
	}
}

func registerFailTool(t *testing.T, reg *ToolRegistry) {
	t.Helper()
	err := reg.Register(
		ToolDef{Name: "fail_tool", Description: "always fails"},
		func(args json.RawMessage) (string, error) {
			return "", fmt.Errorf("intentional error")
		},
	)
	if err != nil {
		t.Fatalf("register fail_tool: %v", err)
	}
}

func TestRunStep(t *testing.T) {
	reg := NewToolRegistry()
	registerEchoTool(t, reg)
	parser := NewFunctionCallParser()
	sup := NewSupervisor(SupervisorConfig{MaxSteps: 5}, reg, parser)

	modelOutput := `Let me call a tool. {"name": "echo", "arguments": {"msg": "hello"}}`

	step, finished, err := sup.RunStep(context.Background(), modelOutput)
	if err != nil {
		t.Fatalf("RunStep: %v", err)
	}
	if finished {
		t.Fatal("expected finished=false when tool calls are present")
	}
	if len(step.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(step.ToolCalls))
	}
	if step.ToolCalls[0].Name != "echo" {
		t.Fatalf("expected tool name 'echo', got %q", step.ToolCalls[0].Name)
	}
	if len(step.ToolResults) != 1 {
		t.Fatalf("expected 1 tool result, got %d", len(step.ToolResults))
	}
	if step.ToolResults[0].IsError {
		t.Fatalf("unexpected tool error: %s", step.ToolResults[0].Output)
	}
	if step.StepNum != 1 {
		t.Fatalf("expected StepNum=1, got %d", step.StepNum)
	}
}

func TestRunStepNoTools(t *testing.T) {
	reg := NewToolRegistry()
	parser := NewFunctionCallParser()
	sup := NewSupervisor(SupervisorConfig{MaxSteps: 5}, reg, parser)

	modelOutput := "The answer is 42."

	step, finished, err := sup.RunStep(context.Background(), modelOutput)
	if err != nil {
		t.Fatalf("RunStep: %v", err)
	}
	if !finished {
		t.Fatal("expected finished=true when no tool calls")
	}
	if len(step.ToolCalls) != 0 {
		t.Fatalf("expected 0 tool calls, got %d", len(step.ToolCalls))
	}
	if len(step.ToolResults) != 0 {
		t.Fatalf("expected 0 tool results, got %d", len(step.ToolResults))
	}
}

func TestRunLoopMaxSteps(t *testing.T) {
	reg := NewToolRegistry()
	registerEchoTool(t, reg)
	parser := NewFunctionCallParser()
	sup := NewSupervisor(SupervisorConfig{MaxSteps: 3}, reg, parser)

	callCount := 0
	generateFn := func(_ context.Context, _ []string) (string, error) {
		callCount++
		return `{"name": "echo", "arguments": {"n": 1}}`, nil
	}

	session, err := sup.RunLoop(context.Background(), generateFn, "go")
	if err != nil {
		t.Fatalf("RunLoop: %v", err)
	}
	if session.StopReason != "max_steps" {
		t.Fatalf("expected stop reason 'max_steps', got %q", session.StopReason)
	}
	if len(session.Steps) != 3 {
		t.Fatalf("expected 3 steps, got %d", len(session.Steps))
	}
	if callCount != 3 {
		t.Fatalf("expected generateFn called 3 times, got %d", callCount)
	}
	if !session.Finished {
		t.Fatal("expected session to be finished")
	}
}

func TestRunLoopStopOnToolError(t *testing.T) {
	reg := NewToolRegistry()
	registerFailTool(t, reg)
	parser := NewFunctionCallParser()
	sup := NewSupervisor(SupervisorConfig{MaxSteps: 10, StopOnToolError: true}, reg, parser)

	generateFn := func(_ context.Context, _ []string) (string, error) {
		return `{"name": "fail_tool", "arguments": {}}`, nil
	}

	session, err := sup.RunLoop(context.Background(), generateFn, "go")
	if err != nil {
		t.Fatalf("RunLoop: %v", err)
	}
	if session.StopReason != "tool_error" {
		t.Fatalf("expected stop reason 'tool_error', got %q", session.StopReason)
	}
	if len(session.Steps) != 1 {
		t.Fatalf("expected 1 step, got %d", len(session.Steps))
	}
}

func TestRunLoopCompletes(t *testing.T) {
	reg := NewToolRegistry()
	registerEchoTool(t, reg)
	parser := NewFunctionCallParser()
	sup := NewSupervisor(SupervisorConfig{MaxSteps: 10}, reg, parser)

	callCount := 0
	generateFn := func(_ context.Context, _ []string) (string, error) {
		callCount++
		if callCount < 3 {
			return `{"name": "echo", "arguments": {"step": true}}`, nil
		}
		return "All done, here is the final answer.", nil
	}

	session, err := sup.RunLoop(context.Background(), generateFn, "start")
	if err != nil {
		t.Fatalf("RunLoop: %v", err)
	}
	if session.StopReason != "no_tools" {
		t.Fatalf("expected stop reason 'no_tools', got %q", session.StopReason)
	}
	if len(session.Steps) != 3 {
		t.Fatalf("expected 3 steps, got %d", len(session.Steps))
	}
	if session.FinalOutput != "All done, here is the final answer." {
		t.Fatalf("unexpected final output: %q", session.FinalOutput)
	}
	if !session.Finished {
		t.Fatal("expected session to be finished")
	}
	if session.TotalTokens <= 0 {
		t.Fatal("expected positive TotalTokens")
	}
}
