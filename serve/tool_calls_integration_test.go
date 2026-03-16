package serve

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestToolCallIntegration(t *testing.T) {
	weatherTool := Tool{
		Type: "function",
		Function: ToolFunction{
			Name:        "get_weather",
			Description: "Get current weather for a location",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		},
	}
	timeTool := Tool{
		Type: "function",
		Function: ToolFunction{
			Name:        "get_time",
			Description: "Get current time in a timezone",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"tz":{"type":"string"}}}`),
		},
	}

	tests := []struct {
		name           string
		modelOutput    string // what the model generates (single token via whitespace tokenizer)
		tools          []Tool
		toolChoice     string // raw JSON for tool_choice field, empty means omit
		wantStatus     int
		wantFinish     string
		wantToolCalls  int
		wantFuncName   string
		wantHasContent bool // true if message.content should be non-empty
	}{
		{
			name:           "auto tool choice with JSON output returns tool_calls",
			modelOutput:    `{"name":"get_weather","arguments":{"city":"NYC"}}`,
			tools:          []Tool{weatherTool, timeTool},
			toolChoice:     `"auto"`,
			wantStatus:     http.StatusOK,
			wantFinish:     "tool_calls",
			wantToolCalls:  1,
			wantFuncName:   "get_weather",
			wantHasContent: false,
		},
		{
			name:           "auto tool choice with non-JSON output returns regular response",
			modelOutput:    "", // use default model (plain text output)
			tools:          []Tool{weatherTool},
			toolChoice:     `"auto"`,
			wantStatus:     http.StatusOK,
			wantFinish:     "stop",
			wantToolCalls:  0,
			wantHasContent: true,
		},
		{
			name:           "forced tool choice with JSON output returns tool_calls",
			modelOutput:    `{"city":"London"}`,
			tools:          []Tool{weatherTool, timeTool},
			toolChoice:     `{"type":"function","function":{"name":"get_weather"}}`,
			wantStatus:     http.StatusOK,
			wantFinish:     "tool_calls",
			wantToolCalls:  1,
			wantFuncName:   "get_weather",
			wantHasContent: false,
		},
		{
			name:           "forced tool choice with non-JSON output still returns tool_calls",
			modelOutput:    "", // use default model (plain text output)
			tools:          []Tool{weatherTool, timeTool},
			toolChoice:     `{"type":"function","function":{"name":"get_weather"}}`,
			wantStatus:     http.StatusOK,
			wantFinish:     "tool_calls",
			wantToolCalls:  1,
			wantFuncName:   "get_weather",
			wantHasContent: false,
		},
		{
			name:        "invalid tool name in request returns 400",
			modelOutput: "",
			tools: []Tool{{
				Type: "function",
				Function: ToolFunction{
					Name:        "invalid name with spaces!",
					Description: "Bad tool",
				},
			}},
			wantStatus: http.StatusBadRequest,
		},
		{
			name:        "forced tool choice referencing unknown function returns 400",
			modelOutput: "",
			tools:       []Tool{weatherTool},
			toolChoice:  `{"type":"function","function":{"name":"nonexistent"}}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:           "tool_choice none with JSON output returns regular response",
			modelOutput:    `{"name":"get_weather","arguments":{"city":"NYC"}}`,
			tools:          []Tool{weatherTool},
			toolChoice:     `"none"`,
			wantStatus:     http.StatusOK,
			wantFinish:     "stop",
			wantToolCalls:  0,
			wantHasContent: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var srv *Server
			if tt.modelOutput != "" {
				model := buildToolCallTestModel(t, []string{tt.modelOutput})
				srv = NewServer(model)
			} else {
				model := buildTestModel(t)
				srv = NewServer(model)
			}
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()

			// Build request body.
			reqObj := map[string]interface{}{
				"model":    "test-model",
				"messages": []map[string]string{{"role": "user", "content": "test prompt"}},
			}

			// Add tools as raw JSON to preserve structure.
			toolsJSON, err := json.Marshal(tt.tools)
			if err != nil {
				t.Fatal(err)
			}
			reqObj["tools"] = json.RawMessage(toolsJSON)

			if tt.toolChoice != "" {
				reqObj["tool_choice"] = json.RawMessage(tt.toolChoice)
			}

			body, err := json.Marshal(reqObj)
			if err != nil {
				t.Fatal(err)
			}

			req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(string(body)))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != tt.wantStatus {
				var errBody map[string]interface{}
				json.NewDecoder(resp.Body).Decode(&errBody)
				t.Fatalf("status=%d, want %d; body=%v", resp.StatusCode, tt.wantStatus, errBody)
			}

			// For error responses, no further validation needed.
			if tt.wantStatus != http.StatusOK {
				return
			}

			var result ChatCompletionResponse
			if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
				t.Fatalf("decode response: %v", err)
			}

			if len(result.Choices) != 1 {
				t.Fatalf("choices=%d, want 1", len(result.Choices))
			}

			choice := result.Choices[0]
			if choice.FinishReason != tt.wantFinish {
				t.Fatalf("finish_reason=%q, want %q", choice.FinishReason, tt.wantFinish)
			}

			if len(choice.ToolCalls) != tt.wantToolCalls {
				t.Fatalf("tool_calls=%d, want %d", len(choice.ToolCalls), tt.wantToolCalls)
			}

			if tt.wantToolCalls > 0 {
				tc := choice.ToolCalls[0]
				if tc.Type != "function" {
					t.Fatalf("tool_call type=%q, want function", tc.Type)
				}
				if tc.Function.Name != tt.wantFuncName {
					t.Fatalf("function name=%q, want %q", tc.Function.Name, tt.wantFuncName)
				}
				if tc.ID == "" {
					t.Fatal("tool_call ID is empty")
				}
				if !json.Valid([]byte(tc.Function.Arguments)) {
					t.Fatalf("tool_call arguments is not valid JSON: %q", tc.Function.Arguments)
				}
			}

			if tt.wantHasContent && choice.Message.Content == "" {
				t.Fatal("expected non-empty message content")
			}
			if !tt.wantHasContent && tt.wantToolCalls > 0 && choice.Message.Content != "" {
				t.Fatalf("expected empty message content for tool call, got %q", choice.Message.Content)
			}
		})
	}
}

func TestForcedToolChoiceArguments(t *testing.T) {
	tool := Tool{
		Type: "function",
		Function: ToolFunction{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  json.RawMessage(`{"type":"object"}`),
		},
	}

	t.Run("valid JSON output used as arguments", func(t *testing.T) {
		model := buildToolCallTestModel(t, []string{`{"city":"Paris"}`})
		srv := NewServer(model)
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()

		body := `{
			"model": "test-model",
			"messages": [{"role": "user", "content": "weather"}],
			"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}}],
			"tool_choice": {"type": "function", "function": {"name": "get_weather"}}
		}`

		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		var result ChatCompletionResponse
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			t.Fatal(err)
		}

		tc := result.Choices[0].ToolCalls[0]
		var args map[string]string
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			t.Fatalf("unmarshal arguments: %v", err)
		}
		if args["city"] != "Paris" {
			t.Fatalf("args[city]=%q, want Paris", args["city"])
		}
	})

	t.Run("non-JSON output gives empty object arguments", func(t *testing.T) {
		// buildTestModel generates plain text ("foo bar"), not JSON.
		model := buildTestModel(t)
		srv := NewServer(model)
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()

		toolJSON, _ := json.Marshal([]Tool{tool})
		choiceJSON := `{"type":"function","function":{"name":"get_weather"}}`
		reqBody := `{
			"model": "test-model",
			"messages": [{"role": "user", "content": "weather"}],
			"tools": ` + string(toolJSON) + `,
			"tool_choice": ` + choiceJSON + `
		}`

		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		var result ChatCompletionResponse
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			t.Fatal(err)
		}

		tc := result.Choices[0].ToolCalls[0]
		if tc.Function.Arguments != "{}" {
			t.Fatalf("arguments=%q, want {}", tc.Function.Arguments)
		}
		if tc.Function.Name != "get_weather" {
			t.Fatalf("function name=%q, want get_weather", tc.Function.Name)
		}
	})
}
