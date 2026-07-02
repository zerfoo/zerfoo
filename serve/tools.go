package serve

import (
	"encoding/json"
	"fmt"
	"regexp"
)

// toolNamePattern matches valid OpenAI tool names: 1-64 alphanumeric, underscore, or hyphen.
var toolNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// Tool represents an OpenAI-compatible tool definition.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction holds the function definition within a tool.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolChoice represents the tool_choice field.
// It can be the string "auto", "none", or an object {"type":"function","function":{"name":"..."}}.
type ToolChoice struct {
	Mode     string // "auto", "none", or "function"
	Function *ToolChoiceFunction
}

// ToolChoiceFunction identifies a specific function in a tool_choice object.
type ToolChoiceFunction struct {
	Name string `json:"name"`
}

// toolChoiceObject is the JSON structure for the object form of tool_choice.
type toolChoiceObject struct {
	Type     string              `json:"type"`
	Function *ToolChoiceFunction `json:"function"`
}

// UnmarshalJSON handles both string and object forms of tool_choice.
func (tc *ToolChoice) UnmarshalJSON(data []byte) error {
	// Try string first.
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		switch s {
		case "auto", "none":
			tc.Mode = s
			return nil
		default:
			return fmt.Errorf("invalid tool_choice value: %q", s)
		}
	}

	// Try object form.
	var obj toolChoiceObject
	if err := json.Unmarshal(data, &obj); err != nil {
		return fmt.Errorf("tool_choice must be a string or object: %w", err)
	}
	if obj.Type != "function" {
		return fmt.Errorf("invalid tool_choice type: %q", obj.Type)
	}
	if obj.Function == nil || obj.Function.Name == "" {
		return fmt.Errorf("tool_choice function name is required")
	}
	tc.Mode = "function"
	tc.Function = obj.Function
	return nil
}

// MarshalJSON encodes ToolChoice back to its JSON representation.
func (tc ToolChoice) MarshalJSON() ([]byte, error) {
	if tc.Mode == "auto" || tc.Mode == "none" {
		return json.Marshal(tc.Mode)
	}
	return json.Marshal(toolChoiceObject{
		Type:     "function",
		Function: tc.Function,
	})
}

// validateTools checks that all tools in the slice have valid definitions.
func validateTools(tools []Tool) error {
	for i, t := range tools {
		if t.Type != "function" {
			return fmt.Errorf("tools[%d].type must be \"function\", got %q", i, t.Type)
		}
		if t.Function.Name == "" {
			return fmt.Errorf("tools[%d].function.name is required", i)
		}
		if !toolNamePattern.MatchString(t.Function.Name) {
			return fmt.Errorf("tools[%d].function.name %q is invalid: must match [a-zA-Z0-9_-]{1,64}", i, t.Function.Name)
		}
		if len(t.Function.Parameters) > 0 && !json.Valid(t.Function.Parameters) {
			return fmt.Errorf("tools[%d].function.parameters contains invalid JSON", i)
		}
	}
	return nil
}

// validateToolChoice checks that tool_choice is consistent with the tools array.
func validateToolChoice(choice *ToolChoice, tools []Tool) error {
	if choice == nil {
		return nil
	}
	if len(tools) == 0 {
		return fmt.Errorf("tool_choice is set but no tools are provided")
	}
	if choice.Mode == "function" {
		for _, t := range tools {
			if t.Function.Name == choice.Function.Name {
				return nil
			}
		}
		return fmt.Errorf("tool_choice function %q not found in tools", choice.Function.Name)
	}
	return nil
}
