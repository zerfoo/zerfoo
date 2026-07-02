package agent

import (
	"encoding/json"
	"testing"
)

func TestParseToolCall(t *testing.T) {
	parser := NewFunctionCallParser()

	text := `Sure, let me look that up. {"name": "GetMarketData", "arguments": {"symbol": "BTCUSD"}} Here is the result.`

	resp := parser.Parse(text)

	if !resp.HasToolCalls {
		t.Fatal("expected HasToolCalls=true")
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}

	tc := resp.ToolCalls[0]
	if tc.ID != "call_0" {
		t.Errorf("expected ID call_0, got %q", tc.ID)
	}
	if tc.Name != "GetMarketData" {
		t.Errorf("expected name GetMarketData, got %q", tc.Name)
	}

	var args map[string]string
	if err := json.Unmarshal(tc.Arguments, &args); err != nil {
		t.Fatalf("unmarshal arguments: %v", err)
	}
	if args["symbol"] != "BTCUSD" {
		t.Errorf("expected symbol BTCUSD, got %q", args["symbol"])
	}

	if resp.Text != "Sure, let me look that up.  Here is the result." {
		t.Errorf("unexpected remaining text: %q", resp.Text)
	}
}

func TestParseMultipleToolCalls(t *testing.T) {
	parser := NewFunctionCallParser()

	text := `First call: {"name": "GetPrice", "arguments": {"ticker": "AAPL"}} and second: {"tool": "GetNews", "parameters": {"query": "earnings"}} done.`

	resp := parser.Parse(text)

	if !resp.HasToolCalls {
		t.Fatal("expected HasToolCalls=true")
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(resp.ToolCalls))
	}

	if resp.ToolCalls[0].Name != "GetPrice" {
		t.Errorf("first call: expected GetPrice, got %q", resp.ToolCalls[0].Name)
	}
	if resp.ToolCalls[0].ID != "call_0" {
		t.Errorf("first call: expected call_0, got %q", resp.ToolCalls[0].ID)
	}

	if resp.ToolCalls[1].Name != "GetNews" {
		t.Errorf("second call: expected GetNews, got %q", resp.ToolCalls[1].Name)
	}
	if resp.ToolCalls[1].ID != "call_1" {
		t.Errorf("second call: expected call_1, got %q", resp.ToolCalls[1].ID)
	}
}

func TestParseNoToolCalls(t *testing.T) {
	parser := NewFunctionCallParser()

	text := "This is just a plain text response with no tool calls at all."

	resp := parser.Parse(text)

	if resp.HasToolCalls {
		t.Fatal("expected HasToolCalls=false")
	}
	if len(resp.ToolCalls) != 0 {
		t.Fatalf("expected 0 tool calls, got %d", len(resp.ToolCalls))
	}
	if resp.Text != text {
		t.Errorf("expected text to be preserved, got %q", resp.Text)
	}
}

func TestParsePartialJSON(t *testing.T) {
	parser := NewFunctionCallParser()

	tests := []struct {
		name string
		text string
	}{
		{
			name: "unclosed brace",
			text: `Here is some {"name": "Foo", "arguments": {"bar": "baz"`,
		},
		{
			name: "missing arguments key",
			text: `Look: {"name": "Foo", "other": "value"} done`,
		},
		{
			name: "arguments not an object",
			text: `Look: {"name": "Foo", "arguments": "string"} done`,
		},
		{
			name: "empty name",
			text: `Look: {"name": "", "arguments": {"x": 1}} done`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := parser.Parse(tt.text)
			if resp.HasToolCalls {
				t.Fatal("expected HasToolCalls=false for malformed input")
			}
			if resp.Text != tt.text {
				t.Errorf("expected original text preserved, got %q", resp.Text)
			}
		})
	}
}

func TestFormatToolResult(t *testing.T) {
	result := ToolResult{
		CallID:  "call_0",
		Output:  "price: 150.25",
		IsError: false,
	}

	got := FormatToolResult(result)

	var parsed struct {
		ToolCallID string `json:"tool_call_id"`
		Output     string `json:"output"`
		IsError    bool   `json:"is_error"`
	}
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if parsed.ToolCallID != "call_0" {
		t.Errorf("expected tool_call_id call_0, got %q", parsed.ToolCallID)
	}
	if parsed.Output != "price: 150.25" {
		t.Errorf("expected output 'price: 150.25', got %q", parsed.Output)
	}
	if parsed.IsError {
		t.Error("expected is_error=false")
	}

	// Test with error.
	errResult := ToolResult{
		CallID:  "call_1",
		Output:  "connection refused",
		IsError: true,
	}
	got2 := FormatToolResult(errResult)
	var parsed2 struct {
		IsError bool `json:"is_error"`
	}
	if err := json.Unmarshal([]byte(got2), &parsed2); err != nil {
		t.Fatalf("unmarshal error result: %v", err)
	}
	if !parsed2.IsError {
		t.Error("expected is_error=true for error result")
	}
}

func TestParseToolPattern(t *testing.T) {
	parser := NewFunctionCallParser()

	// Test the {"tool": ..., "parameters": ...} pattern specifically.
	text := `{"tool": "SearchDocs", "parameters": {"query": "install guide"}}`

	resp := parser.Parse(text)

	if !resp.HasToolCalls {
		t.Fatal("expected HasToolCalls=true for tool/parameters pattern")
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].Name != "SearchDocs" {
		t.Errorf("expected SearchDocs, got %q", resp.ToolCalls[0].Name)
	}
}

func TestParseNestedJSON(t *testing.T) {
	parser := NewFunctionCallParser()

	// Tool call with nested JSON in arguments.
	text := `Result: {"name": "CreateConfig", "arguments": {"config": {"key": "value", "nested": {"deep": true}}}} end`

	resp := parser.Parse(text)

	if !resp.HasToolCalls {
		t.Fatal("expected HasToolCalls=true")
	}
	if resp.ToolCalls[0].Name != "CreateConfig" {
		t.Errorf("expected CreateConfig, got %q", resp.ToolCalls[0].Name)
	}

	var args map[string]json.RawMessage
	if err := json.Unmarshal(resp.ToolCalls[0].Arguments, &args); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := args["config"]; !ok {
		t.Error("expected config key in arguments")
	}
}

func TestParseStringWithBraces(t *testing.T) {
	parser := NewFunctionCallParser()

	// JSON string values containing braces should not confuse the parser.
	text := `{"name": "Echo", "arguments": {"msg": "hello {world}"}}`

	resp := parser.Parse(text)

	if !resp.HasToolCalls {
		t.Fatal("expected HasToolCalls=true")
	}
	if resp.ToolCalls[0].Name != "Echo" {
		t.Errorf("expected Echo, got %q", resp.ToolCalls[0].Name)
	}
}
