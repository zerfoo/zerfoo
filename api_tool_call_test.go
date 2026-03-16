package zerfoo

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/zerfoo/zerfoo/serve"
)

func TestGenerate_ToolCallDetection(t *testing.T) {
	weatherTool := serve.Tool{
		Type: "function",
		Function: serve.ToolFunction{
			Name:        "get_weather",
			Description: "Get the weather for a location",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
		},
	}

	tests := []struct {
		name         string
		output       string
		tools        []serve.Tool
		toolChoice   *serve.ToolChoice
		wantToolCall bool
		wantFunc     string
	}{
		{
			name:         "JSON output detected as tool call",
			output:       `{"name":"get_weather","arguments":{"location":"Paris"}}`,
			tools:        []serve.Tool{weatherTool},
			wantToolCall: true,
			wantFunc:     "get_weather",
		},
		{
			name:         "plain text output not detected",
			output:       "The weather in Paris is sunny today.",
			tools:        []serve.Tool{weatherTool},
			wantToolCall: false,
		},
		{
			name:         "no tools configured means no detection",
			output:       `{"name":"get_weather","arguments":{"location":"Paris"}}`,
			tools:        nil,
			wantToolCall: false,
		},
		{
			name:         "tool choice none suppresses detection",
			output:       `{"name":"get_weather","arguments":{"location":"Paris"}}`,
			tools:        []serve.Tool{weatherTool},
			toolChoice:   &serve.ToolChoice{Mode: "none"},
			wantToolCall: false,
		},
		{
			name:         "single tool with bare JSON args",
			output:       `{"location":"Tokyo"}`,
			tools:        []serve.Tool{weatherTool},
			wantToolCall: true,
			wantFunc:     "get_weather",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{
				generateFunc: func(ctx context.Context, prompt string) (string, error) {
					return tt.output, nil
				},
			}

			var opts []GenerateOption
			if len(tt.tools) > 0 {
				opts = append(opts, WithTools(tt.tools...))
			}
			if tt.toolChoice != nil {
				opts = append(opts, WithToolChoice(*tt.toolChoice))
			}

			result, err := m.Generate(context.Background(), "test prompt", opts...)
			if err != nil {
				t.Fatalf("Generate returned error: %v", err)
			}

			if tt.wantToolCall {
				if len(result.ToolCalls) == 0 {
					t.Fatal("expected ToolCalls to be populated, got none")
				}
				if result.ToolCalls[0].FunctionName != tt.wantFunc {
					t.Errorf("ToolCalls[0].FunctionName = %q, want %q", result.ToolCalls[0].FunctionName, tt.wantFunc)
				}
				if result.ToolCalls[0].ID == "" {
					t.Error("ToolCalls[0].ID should not be empty")
				}
				if len(result.ToolCalls[0].Arguments) == 0 {
					t.Error("ToolCalls[0].Arguments should not be empty")
				}
			} else if len(result.ToolCalls) != 0 {
				t.Errorf("expected no ToolCalls, got %d: %+v", len(result.ToolCalls), result.ToolCalls)
			}

			// Text should always be set regardless of tool call detection.
			if result.Text != tt.output {
				t.Errorf("Text = %q, want %q", result.Text, tt.output)
			}
		})
	}
}
