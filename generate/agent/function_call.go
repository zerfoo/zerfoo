package agent

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ParsedResponse holds the result of parsing model output for tool calls.
type ParsedResponse struct {
	Text         string
	ToolCalls    []ToolCall
	HasToolCalls bool
}

// FunctionCallParser scans model output text for embedded JSON tool-call
// objects and extracts them into structured ToolCall values.
type FunctionCallParser struct{}

// NewFunctionCallParser creates a new FunctionCallParser.
func NewFunctionCallParser() *FunctionCallParser {
	return &FunctionCallParser{}
}

// Parse scans text for JSON objects matching tool-call patterns and extracts
// them as ToolCall values. Two patterns are recognised:
//
//	{"name": "...", "arguments": {...}}
//	{"tool": "...", "parameters": {...}}
//
// Extracted JSON blocks are removed from the text. If no valid tool calls are
// found, HasToolCalls is false and Text contains the original input.
func (p *FunctionCallParser) Parse(text string) ParsedResponse {
	var calls []ToolCall
	var cleaned strings.Builder
	i := 0
	callIndex := 0

	for i < len(text) {
		// Look for the start of a JSON object.
		idx := strings.IndexByte(text[i:], '{')
		if idx < 0 {
			cleaned.WriteString(text[i:])
			break
		}

		// Write everything before the '{'.
		cleaned.WriteString(text[i : i+idx])

		// Try to extract a balanced JSON object starting at this position.
		jsonStr, end, ok := extractJSONObject(text, i+idx)
		if !ok {
			// Not a balanced object; emit the '{' and advance.
			cleaned.WriteByte('{')
			i = i + idx + 1
			continue
		}

		// Try to decode as a tool call.
		if tc, ok := decodeToolCall(jsonStr, callIndex); ok {
			calls = append(calls, tc)
			callIndex++
			i = end
			continue
		}

		// Valid JSON but not a tool call pattern; leave it in text.
		cleaned.WriteString(jsonStr)
		i = end
	}

	if len(calls) == 0 {
		return ParsedResponse{Text: text, HasToolCalls: false}
	}

	return ParsedResponse{
		Text:         strings.TrimSpace(cleaned.String()),
		ToolCalls:    calls,
		HasToolCalls: true,
	}
}

// FormatToolResult serialises a ToolResult as a JSON string suitable for
// injection back into the model context.
func FormatToolResult(result ToolResult) string {
	out := struct {
		ToolCallID string `json:"tool_call_id"`
		Output     string `json:"output"`
		IsError    bool   `json:"is_error"`
	}{
		ToolCallID: result.CallID,
		Output:     result.Output,
		IsError:    result.IsError,
	}
	b, _ := json.Marshal(out)
	return string(b)
}

// extractJSONObject finds a balanced brace-delimited object starting at pos
// in text. It returns the substring, the index one past the closing brace,
// and whether extraction succeeded.
func extractJSONObject(text string, pos int) (string, int, bool) {
	if pos >= len(text) || text[pos] != '{' {
		return "", 0, false
	}

	depth := 0
	inString := false
	escaped := false

	for i := pos; i < len(text); i++ {
		ch := text[i]

		if escaped {
			escaped = false
			continue
		}

		if ch == '\\' && inString {
			escaped = true
			continue
		}

		if ch == '"' {
			inString = !inString
			continue
		}

		if inString {
			continue
		}

		switch ch {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return text[pos : i+1], i + 1, true
			}
		}
	}

	return "", 0, false
}

// decodeToolCall attempts to unmarshal jsonStr as a tool call. It recognises
// {"name": ..., "arguments": ...} and {"tool": ..., "parameters": ...}.
func decodeToolCall(jsonStr string, index int) (ToolCall, bool) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(jsonStr), &raw); err != nil {
		return ToolCall{}, false
	}

	name, args, ok := extractNameArgs(raw, "name", "arguments")
	if !ok {
		name, args, ok = extractNameArgs(raw, "tool", "parameters")
	}
	if !ok {
		return ToolCall{}, false
	}

	return ToolCall{
		ID:        fmt.Sprintf("call_%d", index),
		Name:      name,
		Arguments: args,
	}, true
}

// extractNameArgs pulls a string value and a raw JSON object from a decoded
// map using the given key names.
func extractNameArgs(raw map[string]json.RawMessage, nameKey, argsKey string) (string, json.RawMessage, bool) {
	nameRaw, ok := raw[nameKey]
	if !ok {
		return "", nil, false
	}
	argsRaw, ok := raw[argsKey]
	if !ok {
		return "", nil, false
	}

	var name string
	if err := json.Unmarshal(nameRaw, &name); err != nil {
		return "", nil, false
	}
	if name == "" {
		return "", nil, false
	}

	// Verify arguments is a JSON object.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(argsRaw, &obj); err != nil {
		return "", nil, false
	}

	return name, argsRaw, true
}
