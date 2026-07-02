package serve

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ToolCallResult holds a detected tool call.
type ToolCallResult struct {
	ID           string          // e.g. "call_1234567890"
	FunctionName string
	Arguments    json.RawMessage // raw JSON of arguments
}

// DetectToolCall examines generated text to determine if it is a tool call.
// Returns (result, true) if a tool call is detected, (nil, false) otherwise.
//
// Detection heuristic:
//  1. If tool_choice is "none": never detect
//  2. Trim whitespace from text
//  3. If text starts with '{' and parses as valid JSON object: it is a tool call
//  4. If tool_choice forces a specific function: use that name
//  5. Otherwise: look for "name" field in the JSON to match a tool
func DetectToolCall(text string, tools []Tool, choice ToolChoice) (*ToolCallResult, bool) {
	// Never detect if choice is "none".
	if choice.Mode == "none" {
		return nil, false
	}

	// No tools means nothing to detect.
	if len(tools) == 0 {
		return nil, false
	}

	trimmed := strings.TrimSpace(text)
	if trimmed == "" || trimmed[0] != '{' {
		return nil, false
	}

	// Must parse as valid JSON.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(trimmed), &obj); err != nil {
		return nil, false
	}

	result := &ToolCallResult{
		ID: generateCallID(),
	}

	// If tool_choice forces a specific function, use that name.
	if choice.Mode == "function" && choice.Function != nil {
		result.FunctionName = choice.Function.Name
		result.Arguments = json.RawMessage(trimmed)
		return result, true
	}

	// Look for a "name" field in the JSON to match a tool.
	if nameRaw, ok := obj["name"]; ok {
		var name string
		if err := json.Unmarshal(nameRaw, &name); err == nil {
			for _, t := range tools {
				if t.Function.Name == name {
					result.FunctionName = name
					// Use "arguments" field if present, otherwise the whole object.
					if argsRaw, ok := obj["arguments"]; ok {
						result.Arguments = argsRaw
					} else {
						result.Arguments = json.RawMessage(trimmed)
					}
					return result, true
				}
			}
		}
	}

	// If only one tool is available, assume the JSON is arguments for it.
	if len(tools) == 1 {
		result.FunctionName = tools[0].Function.Name
		result.Arguments = json.RawMessage(trimmed)
		return result, true
	}

	return nil, false
}

// generateCallID creates a unique tool call ID.
func generateCallID() string {
	return fmt.Sprintf("call_%d", time.Now().UnixNano())
}
