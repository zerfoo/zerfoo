// Command agentic-tool-use demonstrates function calling (tool use) with a
// language model using the zerfoo one-line API.
//
// The model is given a set of tools (functions) it can invoke. When the model
// decides to call a tool, the application executes the function locally and
// feeds the result back for a follow-up generation. This is the core pattern
// behind agentic systems that combine LLM reasoning with real-world actions.
//
// Usage:
//
//	go build -o agentic-tool-use ./examples/agentic-tool-use/
//	./agentic-tool-use --model path/to/model.gguf
//	./agentic-tool-use --model path/to/model.gguf --query "What is the weather in Tokyo?"
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo"
	"github.com/zerfoo/zerfoo/serve"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	query := flag.String("query", "What is the weather in San Francisco?", "user query")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: agentic-tool-use --model <path> [--query <text>]")
		os.Exit(1)
	}

	// Load the model.
	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	// Define available tools. These follow the OpenAI function-calling schema.
	tools := []serve.Tool{
		{
			Type: "function",
			Function: serve.ToolFunction{
				Name:        "get_weather",
				Description: "Get the current weather for a given city.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}`),
			},
		},
		{
			Type: "function",
			Function: serve.ToolFunction{
				Name:        "calculate",
				Description: "Evaluate a mathematical expression.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string","description":"Math expression to evaluate"}},"required":["expression"]}`),
			},
		},
	}

	fmt.Fprintf(os.Stderr, "Query: %s\n", *query)
	fmt.Fprintf(os.Stderr, "Available tools: get_weather, calculate\n\n")

	// Step 1: Ask the model to decide which tool to call (if any).
	result, err := m.Generate(context.Background(), *query,
		zerfoo.WithTools(tools...),
		zerfoo.WithGenMaxTokens(256),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	// Step 2: Check if the model invoked a tool.
	if len(result.ToolCalls) == 0 {
		// No tool call -- the model answered directly.
		fmt.Println("Direct answer:", result.Text)
		return
	}

	// Step 3: Execute the tool call locally.
	tc := result.ToolCalls[0]
	fmt.Printf("Tool call: %s(%s)\n", tc.FunctionName, string(tc.Arguments))

	toolResult := executeToolCall(tc.FunctionName, tc.Arguments)
	fmt.Printf("Tool result: %s\n\n", toolResult)

	// Step 4: Feed the tool result back to the model for a natural-language answer.
	followUp := fmt.Sprintf(
		"The user asked: %s\n\nYou called %s and got: %s\n\nProvide a helpful answer based on this result.",
		*query, tc.FunctionName, toolResult,
	)

	finalResult, err := m.Generate(context.Background(), followUp, zerfoo.WithGenMaxTokens(256))
	if err != nil {
		fmt.Fprintf(os.Stderr, "follow-up generate: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Answer:", finalResult.Text)
}

// executeToolCall dispatches a tool call to its local implementation.
// In a real application, these would call APIs, databases, or system tools.
func executeToolCall(name string, args json.RawMessage) string {
	switch name {
	case "get_weather":
		var params struct {
			City string `json:"city"`
		}
		if err := json.Unmarshal(args, &params); err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		// Simulated weather response.
		return fmt.Sprintf(`{"city":%q,"temperature":"18°C","condition":"partly cloudy","humidity":"65%%"}`, params.City)

	case "calculate":
		var params struct {
			Expression string `json:"expression"`
		}
		if err := json.Unmarshal(args, &params); err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		// Simulated calculator -- in production, use a real expression evaluator.
		return fmt.Sprintf(`{"expression":%q,"result":"computed"}`, params.Expression)

	default:
		return fmt.Sprintf(`{"error":"unknown tool: %s"}`, name)
	}
}
