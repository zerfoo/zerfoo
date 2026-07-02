package serve

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestToolChoiceUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
		mode    string
		fnName  string
	}{
		{
			name:  "auto string",
			input: `"auto"`,
			mode:  "auto",
		},
		{
			name:  "none string",
			input: `"none"`,
			mode:  "none",
		},
		{
			name:    "invalid string value",
			input:   `"required"`,
			wantErr: true,
		},
		{
			name:   "function object",
			input:  `{"type":"function","function":{"name":"get_weather"}}`,
			mode:   "function",
			fnName: "get_weather",
		},
		{
			name:    "object with invalid type",
			input:   `{"type":"tool","function":{"name":"foo"}}`,
			wantErr: true,
		},
		{
			name:    "object with missing function name",
			input:   `{"type":"function","function":{}}`,
			wantErr: true,
		},
		{
			name:    "object with null function",
			input:   `{"type":"function"}`,
			wantErr: true,
		},
		{
			name:    "invalid JSON",
			input:   `{bad}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var tc ToolChoice
			err := json.Unmarshal([]byte(tt.input), &tc)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.Mode != tt.mode {
				t.Errorf("mode = %q, want %q", tc.Mode, tt.mode)
			}
			if tt.fnName != "" {
				if tc.Function == nil {
					t.Fatal("expected function, got nil")
				}
				if tc.Function.Name != tt.fnName {
					t.Errorf("function name = %q, want %q", tc.Function.Name, tt.fnName)
				}
			}
		})
	}
}

func TestToolChoiceMarshalJSON(t *testing.T) {
	tests := []struct {
		name string
		tc   ToolChoice
		want string
	}{
		{name: "auto", tc: ToolChoice{Mode: "auto"}, want: `"auto"`},
		{name: "none", tc: ToolChoice{Mode: "none"}, want: `"none"`},
		{
			name: "function",
			tc:   ToolChoice{Mode: "function", Function: &ToolChoiceFunction{Name: "foo"}},
			want: `{"type":"function","function":{"name":"foo"}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.tc)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if string(data) != tt.want {
				t.Errorf("got %s, want %s", data, tt.want)
			}
		})
	}
}

func TestValidateTools(t *testing.T) {
	tests := []struct {
		name    string
		tools   []Tool
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid tool",
			tools: []Tool{{
				Type: "function",
				Function: ToolFunction{
					Name:        "get_weather",
					Description: "Get the weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
				},
			}},
		},
		{
			name: "valid tool without parameters",
			tools: []Tool{{
				Type:     "function",
				Function: ToolFunction{Name: "list_items", Description: "List items"},
			}},
		},
		{
			name: "valid tool name with hyphens and underscores",
			tools: []Tool{{
				Type:     "function",
				Function: ToolFunction{Name: "get-weather_v2", Description: "Get weather"},
			}},
		},
		{
			name: "invalid type",
			tools: []Tool{{
				Type:     "tool",
				Function: ToolFunction{Name: "foo"},
			}},
			wantErr: true,
			errMsg:  "must be \"function\"",
		},
		{
			name: "missing function name",
			tools: []Tool{{
				Type:     "function",
				Function: ToolFunction{Description: "No name"},
			}},
			wantErr: true,
			errMsg:  "name is required",
		},
		{
			name: "name too long",
			tools: []Tool{{
				Type:     "function",
				Function: ToolFunction{Name: strings.Repeat("a", 65)},
			}},
			wantErr: true,
			errMsg:  "is invalid",
		},
		{
			name: "name with special chars",
			tools: []Tool{{
				Type:     "function",
				Function: ToolFunction{Name: "get weather!"},
			}},
			wantErr: true,
			errMsg:  "is invalid",
		},
		{
			name: "invalid JSON in parameters",
			tools: []Tool{{
				Type:     "function",
				Function: ToolFunction{Name: "foo", Parameters: json.RawMessage(`{invalid}`)},
			}},
			wantErr: true,
			errMsg:  "invalid JSON",
		},
		{
			name: "multiple tools valid",
			tools: []Tool{
				{Type: "function", Function: ToolFunction{Name: "foo"}},
				{Type: "function", Function: ToolFunction{Name: "bar"}},
			},
		},
		{
			name: "second tool invalid",
			tools: []Tool{
				{Type: "function", Function: ToolFunction{Name: "foo"}},
				{Type: "function", Function: ToolFunction{Name: ""}},
			},
			wantErr: true,
			errMsg:  "tools[1]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateTools(tt.tools)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errMsg)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestValidateToolChoice(t *testing.T) {
	tools := []Tool{
		{Type: "function", Function: ToolFunction{Name: "get_weather"}},
	}

	tests := []struct {
		name    string
		choice  *ToolChoice
		tools   []Tool
		wantErr bool
		errMsg  string
	}{
		{name: "nil choice", choice: nil, tools: nil},
		{name: "auto with tools", choice: &ToolChoice{Mode: "auto"}, tools: tools},
		{name: "none with tools", choice: &ToolChoice{Mode: "none"}, tools: tools},
		{
			name:   "function choice matching tool",
			choice: &ToolChoice{Mode: "function", Function: &ToolChoiceFunction{Name: "get_weather"}},
			tools:  tools,
		},
		{
			name:    "choice set but no tools",
			choice:  &ToolChoice{Mode: "auto"},
			tools:   nil,
			wantErr: true,
			errMsg:  "no tools are provided",
		},
		{
			name:    "choice set but empty tools",
			choice:  &ToolChoice{Mode: "auto"},
			tools:   []Tool{},
			wantErr: true,
			errMsg:  "no tools are provided",
		},
		{
			name:    "function choice not in tools",
			choice:  &ToolChoice{Mode: "function", Function: &ToolChoiceFunction{Name: "unknown"}},
			tools:   tools,
			wantErr: true,
			errMsg:  "not found in tools",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateToolChoice(tt.choice, tt.tools)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errMsg)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestChatCompletionRequestToolsParsing(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantErr    bool
		toolCount  int
		choiceMode string
	}{
		{
			name: "request with tools and auto choice",
			body: `{
				"model": "test",
				"messages": [{"role":"user","content":"hi"}],
				"tools": [{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object"}}}],
				"tool_choice": "auto"
			}`,
			toolCount:  1,
			choiceMode: "auto",
		},
		{
			name: "request with tools no choice",
			body: `{
				"model": "test",
				"messages": [{"role":"user","content":"hi"}],
				"tools": [{"type":"function","function":{"name":"foo"}}]
			}`,
			toolCount: 1,
		},
		{
			name: "request without tools",
			body: `{
				"model": "test",
				"messages": [{"role":"user","content":"hi"}]
			}`,
			toolCount: 0,
		},
		{
			name: "request with function choice object",
			body: `{
				"model": "test",
				"messages": [{"role":"user","content":"hi"}],
				"tools": [{"type":"function","function":{"name":"foo"}}],
				"tool_choice": {"type":"function","function":{"name":"foo"}}
			}`,
			toolCount:  1,
			choiceMode: "function",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req ChatCompletionRequest
			err := json.Unmarshal([]byte(tt.body), &req)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(req.Tools) != tt.toolCount {
				t.Errorf("tool count = %d, want %d", len(req.Tools), tt.toolCount)
			}
			if tt.choiceMode != "" {
				if req.ToolChoice == nil {
					t.Fatal("expected tool_choice, got nil")
				}
				if req.ToolChoice.Mode != tt.choiceMode {
					t.Errorf("tool_choice mode = %q, want %q", req.ToolChoice.Mode, tt.choiceMode)
				}
			}
		})
	}
}

func TestHandleChatCompletionsToolValidation(t *testing.T) {
	model := buildTestModel(t)
	srv := NewServer(model)

	tests := []struct {
		name       string
		body       string
		wantStatus int
	}{
		{
			name: "valid tools returns 200",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather"}}]
			}`,
			wantStatus: http.StatusOK,
		},
		{
			name: "no tools returns 200",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}]
			}`,
			wantStatus: http.StatusOK,
		},
		{
			name: "invalid tool name returns 400",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"type":"function","function":{"name":"bad name!","description":"desc"}}]
			}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "missing tool function name returns 400",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"type":"function","function":{"description":"no name"}}]
			}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "invalid tool type returns 400",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"type":"retrieval","function":{"name":"foo"}}]
			}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "invalid tool_choice string returns 400",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"type":"function","function":{"name":"foo"}}],
				"tool_choice":"required"
			}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "tool_choice set but empty tools returns 400",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[],
				"tool_choice":"auto"
			}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "tool_choice function not in tools returns 400",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"type":"function","function":{"name":"foo"}}],
				"tool_choice":{"type":"function","function":{"name":"bar"}}
			}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "invalid JSON in parameters returns 400",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"type":"function","function":{"name":"foo","parameters":{invalid}}}]
			}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "valid tools with auto choice returns 200",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"type":"function","function":{"name":"foo"}}],
				"tool_choice":"auto"
			}`,
			wantStatus: http.StatusOK,
		},
		{
			name: "valid tools with none choice returns 200",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"type":"function","function":{"name":"foo"}}],
				"tool_choice":"none"
			}`,
			wantStatus: http.StatusOK,
		},
		{
			name: "valid function choice returns 200",
			body: `{
				"model":"test",
				"messages":[{"role":"user","content":"hi"}],
				"tools":[{"type":"function","function":{"name":"foo"}}],
				"tool_choice":{"type":"function","function":{"name":"foo"}}
			}`,
			wantStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()

			srv.handleChatCompletions(rec, req)

			if rec.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d; body: %s", rec.Code, tt.wantStatus, rec.Body.String())
			}
		})
	}
}
