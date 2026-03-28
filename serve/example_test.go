package serve_test

import (
	"encoding/json"
	"fmt"

	"github.com/zerfoo/zerfoo/serve"
)

func ExampleDetectToolCall() {
	tools := []serve.Tool{{
		Type: "function",
		Function: serve.ToolFunction{
			Name:        "get_weather",
			Description: "Get weather for a city",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		},
	}}

	text := `{"name":"get_weather","arguments":{"city":"Paris"}}`
	result, ok := serve.DetectToolCall(text, tools, serve.ToolChoice{Mode: "auto"})
	if ok {
		fmt.Printf("%s %s\n", result.FunctionName, result.Arguments)
	}
	// Output: get_weather {"city":"Paris"}
}
