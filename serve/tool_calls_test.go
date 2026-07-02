package serve

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/model/registry"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	tokenizer "github.com/zerfoo/ztoken"
)

func TestDetectToolCall(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "get_weather",
				Description: "Get weather for a city",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			},
		},
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "get_time",
				Description: "Get current time",
			},
		},
	}

	tests := []struct {
		name       string
		text       string
		tools      []Tool
		choice     ToolChoice
		wantDetect bool
		wantFunc   string
	}{
		{
			name:       "valid JSON with name field matching tool",
			text:       `{"name":"get_weather","arguments":{"city":"NYC"}}`,
			tools:      tools,
			choice:     ToolChoice{Mode: "auto"},
			wantDetect: true,
			wantFunc:   "get_weather",
		},
		{
			name:       "non-JSON text",
			text:       "The weather in NYC is sunny today.",
			tools:      tools,
			choice:     ToolChoice{Mode: "auto"},
			wantDetect: false,
		},
		{
			name:       "tool_choice none never detects",
			text:       `{"name":"get_weather","arguments":{"city":"NYC"}}`,
			tools:      tools,
			choice:     ToolChoice{Mode: "none"},
			wantDetect: false,
		},
		{
			name:       "specific tool_choice forces function name",
			text:       `{"city":"NYC"}`,
			tools:      tools,
			choice:     ToolChoice{Mode: "function", Function: &ToolChoiceFunction{Name: "get_weather"}},
			wantDetect: true,
			wantFunc:   "get_weather",
		},
		{
			name:       "empty tools never detects",
			text:       `{"name":"get_weather","arguments":{"city":"NYC"}}`,
			tools:      nil,
			choice:     ToolChoice{Mode: "auto"},
			wantDetect: false,
		},
		{
			name:       "whitespace around JSON still detected",
			text:       `  {"name":"get_weather","arguments":{"city":"NYC"}}  `,
			tools:      tools,
			choice:     ToolChoice{Mode: "auto"},
			wantDetect: true,
			wantFunc:   "get_weather",
		},
		{
			name:       "invalid JSON not detected",
			text:       `{"name":"get_weather", broken}`,
			tools:      tools,
			choice:     ToolChoice{Mode: "auto"},
			wantDetect: false,
		},
		{
			name:       "JSON with unmatched name not detected with multiple tools",
			text:       `{"name":"unknown_func","arguments":{}}`,
			tools:      tools,
			choice:     ToolChoice{Mode: "auto"},
			wantDetect: false,
		},
		{
			name:       "single tool with no name field uses that tool",
			text:       `{"city":"NYC"}`,
			tools:      tools[:1],
			choice:     ToolChoice{Mode: "auto"},
			wantDetect: true,
			wantFunc:   "get_weather",
		},
		{
			name:       "arguments field extracted when present",
			text:       `{"name":"get_time","arguments":{"tz":"UTC"}}`,
			tools:      tools,
			choice:     ToolChoice{Mode: "auto"},
			wantDetect: true,
			wantFunc:   "get_time",
		},
		{
			name:       "empty text not detected",
			text:       "",
			tools:      tools,
			choice:     ToolChoice{Mode: "auto"},
			wantDetect: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, detected := DetectToolCall(tt.text, tt.tools, tt.choice)
			if detected != tt.wantDetect {
				t.Fatalf("DetectToolCall detected=%v, want %v", detected, tt.wantDetect)
			}
			if detected {
				if result == nil {
					t.Fatal("detected but result is nil")
				}
				if result.FunctionName != tt.wantFunc {
					t.Fatalf("FunctionName=%q, want %q", result.FunctionName, tt.wantFunc)
				}
				if result.ID == "" {
					t.Fatal("ID is empty")
				}
				if len(result.Arguments) == 0 {
					t.Fatal("Arguments is empty")
				}
			}
		})
	}
}

func TestDetectToolCallArgumentsExtraction(t *testing.T) {
	tools := []Tool{
		{Type: "function", Function: ToolFunction{Name: "get_weather"}},
	}

	// When name matches and arguments field exists, use just the arguments.
	text := `{"name":"get_weather","arguments":{"city":"NYC"}}`
	result, ok := DetectToolCall(text, tools, ToolChoice{Mode: "auto"})
	if !ok {
		t.Fatal("expected detection")
	}
	// Arguments should be the inner object, not the whole JSON.
	var args map[string]string
	if err := json.Unmarshal(result.Arguments, &args); err != nil {
		t.Fatalf("unmarshal arguments: %v", err)
	}
	if args["city"] != "NYC" {
		t.Fatalf("args[city]=%q, want NYC", args["city"])
	}
}

// buildToolCallTestModel creates a test model whose output is composed of
// the given words (whitespace-separated tokens). Each word is registered as
// a token in the whitespace tokenizer, so the generator can produce them.
func buildToolCallTestModel(t *testing.T, words []string) *inference.Model {
	t.Helper()
	vocabSize := 4 + len(words) // 0=PAD, 1=BOS, 2=EOS, 3=UNK, then custom
	tok := tokenizer.NewWhitespaceTokenizer()

	// Register each word as a token (IDs start at 4).
	tokenIDs := make([]int, len(words))
	for i, w := range words {
		id := tok.AddToken(w)
		tokenIDs[i] = id
	}
	tokenIDs = append(tokenIDs, 2) // EOS

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})

	node := &fixedLogitsNode{
		vocabSize:     vocabSize,
		tokenSequence: tokenIDs,
	}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	gen := generate.NewGenerator(g, tok, engine, generate.ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  512,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  0,
	})

	return inference.NewTestModel(gen, tok, engine,
		inference.ModelMetadata{
			VocabSize:  vocabSize,
			NumLayers:  1,
			EOSTokenID: 2,
			BOSTokenID: 1,
		},
		&registry.ModelInfo{ID: "test-model", Path: "/tmp/test"},
	)
}

func TestHandlerToolCallResponse(t *testing.T) {
	// Model generates a single token that is valid JSON matching a tool.
	// The whitespace tokenizer treats the whole blob as one word token.
	jsonBlob := `{"name":"get_weather","arguments":{"city":"NYC"}}`
	model := buildToolCallTestModel(t, []string{jsonBlob})
	srv := NewServer(model)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "weather in NYC"}],
		"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}}]
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

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d, want 200", resp.StatusCode)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if len(result.Choices) != 1 {
		t.Fatalf("choices=%d, want 1", len(result.Choices))
	}

	choice := result.Choices[0]
	if choice.FinishReason != "tool_calls" {
		t.Fatalf("finish_reason=%q, want tool_calls", choice.FinishReason)
	}
	if len(choice.ToolCalls) != 1 {
		t.Fatalf("tool_calls=%d, want 1", len(choice.ToolCalls))
	}
	tc := choice.ToolCalls[0]
	if tc.Type != "function" {
		t.Fatalf("tool_call type=%q, want function", tc.Type)
	}
	if tc.Function.Name != "get_weather" {
		t.Fatalf("function name=%q, want get_weather", tc.Function.Name)
	}
	if tc.ID == "" {
		t.Fatal("tool_call ID is empty")
	}
	if choice.Message.Content != "" {
		t.Fatalf("message content=%q, want empty for tool call", choice.Message.Content)
	}
}

func TestHandlerNonToolCallResponse(t *testing.T) {
	// Model generates plain text (not JSON).
	model := buildTestModel(t)
	srv := NewServer(model)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{
		"model": "test-model",
		"messages": [{"role": "user", "content": "hello"}],
		"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather"}}]
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

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d, want 200", resp.StatusCode)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if len(result.Choices) != 1 {
		t.Fatalf("choices=%d, want 1", len(result.Choices))
	}

	choice := result.Choices[0]
	if choice.FinishReason != "stop" {
		t.Fatalf("finish_reason=%q, want stop", choice.FinishReason)
	}
	if len(choice.ToolCalls) != 0 {
		t.Fatalf("tool_calls=%d, want 0", len(choice.ToolCalls))
	}
}
