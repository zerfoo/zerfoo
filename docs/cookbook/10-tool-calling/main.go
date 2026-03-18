// Recipe 10: Tool / Function Calling
//
// Let the model decide when to invoke tools (functions). The application
// defines a set of tools, the model generates a tool call, the application
// executes it, and feeds the result back for a final natural-language answer.
//
// This is the core pattern behind agentic systems that combine LLM reasoning
// with real-world actions (API calls, database queries, calculations, etc.).
//
// Usage:
//
//	go run ./docs/cookbook/10-tool-calling/ --model path/to/model.gguf
//	go run ./docs/cookbook/10-tool-calling/ --model path/to/model.gguf --query "What time is it in London?"
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/zerfoo/zerfoo"
	"github.com/zerfoo/zerfoo/serve"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	query := flag.String("query", "What time is it in Tokyo?", "user query")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: tool-calling --model <path> [--query <text>]")
		os.Exit(1)
	}

	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	// Define tools using the OpenAI function-calling schema.
	tools := []serve.Tool{
		{
			Type: "function",
			Function: serve.ToolFunction{
				Name:        "get_current_time",
				Description: "Get the current time in a given timezone.",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {
						"timezone": {
							"type": "string",
							"description": "IANA timezone name, e.g. America/New_York"
						}
					},
					"required": ["timezone"]
				}`),
			},
		},
		{
			Type: "function",
			Function: serve.ToolFunction{
				Name:        "lookup_word",
				Description: "Look up the definition of an English word.",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {
						"word": {"type": "string", "description": "The word to look up"}
					},
					"required": ["word"]
				}`),
			},
		},
	}

	fmt.Fprintf(os.Stderr, "Query: %s\n\n", *query)

	// Step 1: Ask the model to decide which tool to call.
	result, err := m.Generate(context.Background(), *query,
		zerfoo.WithTools(tools...),
		zerfoo.WithGenMaxTokens(256),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	// Step 2: If no tool call, the model answered directly.
	if len(result.ToolCalls) == 0 {
		fmt.Println("Direct answer:", result.Text)
		return
	}

	// Step 3: Execute the tool call.
	tc := result.ToolCalls[0]
	fmt.Printf("Tool call: %s(%s)\n", tc.FunctionName, string(tc.Arguments))

	toolResult := dispatch(tc.FunctionName, tc.Arguments)
	fmt.Printf("Tool result: %s\n\n", toolResult)

	// Step 4: Feed the result back for a natural-language answer.
	followUp := fmt.Sprintf(
		"The user asked: %s\nYou called %s and got: %s\nProvide a helpful answer.",
		*query, tc.FunctionName, toolResult,
	)
	final, err := m.Generate(context.Background(), followUp, zerfoo.WithGenMaxTokens(256))
	if err != nil {
		fmt.Fprintf(os.Stderr, "follow-up: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Answer:", final.Text)
}

// dispatch executes a tool call locally.
func dispatch(name string, args json.RawMessage) string {
	switch name {
	case "get_current_time":
		var p struct {
			Timezone string `json:"timezone"`
		}
		if err := json.Unmarshal(args, &p); err != nil {
			return fmt.Sprintf(`{"error":%q}`, err)
		}
		loc, err := time.LoadLocation(p.Timezone)
		if err != nil {
			return fmt.Sprintf(`{"error":%q}`, err)
		}
		return fmt.Sprintf(`{"timezone":%q,"time":%q}`, p.Timezone, time.Now().In(loc).Format(time.RFC3339))

	case "lookup_word":
		var p struct {
			Word string `json:"word"`
		}
		if err := json.Unmarshal(args, &p); err != nil {
			return fmt.Sprintf(`{"error":%q}`, err)
		}
		// Placeholder dictionary.
		return fmt.Sprintf(`{"word":%q,"definition":"(definition would come from a real dictionary API)"}`, p.Word)

	default:
		return fmt.Sprintf(`{"error":"unknown tool: %s"}`, name)
	}
}
