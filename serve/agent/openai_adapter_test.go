package agent

import (
	"encoding/json"
	"testing"

	genagent "github.com/zerfoo/zerfoo/generate/agent"
	"github.com/zerfoo/zerfoo/serve"
)

func TestConvertTools(t *testing.T) {
	tools := []serve.Tool{
		{
			Type: "function",
			Function: serve.ToolFunction{
				Name:        "get_weather",
				Description: "Get weather for a city",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			},
		},
		{
			Type: "function",
			Function: serve.ToolFunction{
				Name:        "get_time",
				Description: "Get current time",
			},
		},
	}

	defs := ConvertTools(tools)
	if len(defs) != 2 {
		t.Fatalf("len(defs)=%d, want 2", len(defs))
	}
	if defs[0].Name != "get_weather" {
		t.Fatalf("defs[0].Name=%q, want get_weather", defs[0].Name)
	}
	if defs[0].Description != "Get weather for a city" {
		t.Fatalf("defs[0].Description=%q, want 'Get weather for a city'", defs[0].Description)
	}
	if string(defs[0].Parameters) != `{"type":"object","properties":{"city":{"type":"string"}}}` {
		t.Fatalf("defs[0].Parameters=%s, unexpected", defs[0].Parameters)
	}
	if defs[1].Name != "get_time" {
		t.Fatalf("defs[1].Name=%q, want get_time", defs[1].Name)
	}
}

func TestConvertToolsEmpty(t *testing.T) {
	defs := ConvertTools(nil)
	if len(defs) != 0 {
		t.Fatalf("len(defs)=%d, want 0", len(defs))
	}
}

func TestToolCallFromAgent(t *testing.T) {
	tc := genagent.ToolCall{
		ID:        "call_123",
		Name:      "get_weather",
		Arguments: json.RawMessage(`{"city":"NYC"}`),
	}

	result := ToolCallFromAgent(tc)
	if result.ID != "call_123" {
		t.Fatalf("ID=%q, want call_123", result.ID)
	}
	if result.Type != "function" {
		t.Fatalf("Type=%q, want function", result.Type)
	}
	if result.Function.Name != "get_weather" {
		t.Fatalf("Function.Name=%q, want get_weather", result.Function.Name)
	}
	if result.Function.Arguments != `{"city":"NYC"}` {
		t.Fatalf("Function.Arguments=%q, unexpected", result.Function.Arguments)
	}
}

func TestToolCallsFromStep(t *testing.T) {
	step := &genagent.AgentStep{
		ToolCalls: []genagent.ToolCall{
			{ID: "call_1", Name: "func_a", Arguments: json.RawMessage(`{}`)},
			{ID: "call_2", Name: "func_b", Arguments: json.RawMessage(`{"x":1}`)},
		},
	}

	calls := ToolCallsFromStep(step)
	if len(calls) != 2 {
		t.Fatalf("len(calls)=%d, want 2", len(calls))
	}
	if calls[0].Function.Name != "func_a" {
		t.Fatalf("calls[0].Function.Name=%q, want func_a", calls[0].Function.Name)
	}
	if calls[1].Function.Name != "func_b" {
		t.Fatalf("calls[1].Function.Name=%q, want func_b", calls[1].Function.Name)
	}
}

func TestToolCallsFromStepNil(t *testing.T) {
	if calls := ToolCallsFromStep(nil); calls != nil {
		t.Fatalf("expected nil, got %v", calls)
	}
	step := &genagent.AgentStep{}
	if calls := ToolCallsFromStep(step); calls != nil {
		t.Fatalf("expected nil for empty step, got %v", calls)
	}
}

func TestResponseFromSessionWithToolCalls(t *testing.T) {
	session := &genagent.AgentSession{
		Steps: []genagent.AgentStep{
			{
				StepNum:     1,
				ModelOutput: "",
				ToolCalls: []genagent.ToolCall{
					{ID: "call_42", Name: "get_weather", Arguments: json.RawMessage(`{"city":"NYC"}`)},
				},
				TokensUsed: 50,
			},
		},
		Finished:    true,
		FinalOutput: "",
		TotalTokens: 50,
		StopReason:  "no_tools",
	}

	resp := ResponseFromSession(session, "test-model")
	if resp.Object != "chat.completion" {
		t.Fatalf("Object=%q, want chat.completion", resp.Object)
	}
	if resp.Model != "test-model" {
		t.Fatalf("Model=%q, want test-model", resp.Model)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("len(Choices)=%d, want 1", len(resp.Choices))
	}

	choice := resp.Choices[0]
	if choice.FinishReason != "tool_calls" {
		t.Fatalf("FinishReason=%q, want tool_calls", choice.FinishReason)
	}
	if choice.Message.Content != "" {
		t.Fatalf("Message.Content=%q, want empty", choice.Message.Content)
	}
	if choice.Message.Role != "assistant" {
		t.Fatalf("Message.Role=%q, want assistant", choice.Message.Role)
	}
	if len(choice.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls)=%d, want 1", len(choice.ToolCalls))
	}
	if choice.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("ToolCalls[0].Function.Name=%q, want get_weather", choice.ToolCalls[0].Function.Name)
	}
}

func TestResponseFromSessionWithoutToolCalls(t *testing.T) {
	session := &genagent.AgentSession{
		Steps: []genagent.AgentStep{
			{
				StepNum:     1,
				ModelOutput: "Hello, how can I help?",
				TokensUsed:  30,
			},
		},
		Finished:    true,
		FinalOutput: "Hello, how can I help?",
		TotalTokens: 30,
		StopReason:  "no_tools",
	}

	resp := ResponseFromSession(session, "test-model")
	choice := resp.Choices[0]
	if choice.FinishReason != "stop" {
		t.Fatalf("FinishReason=%q, want stop", choice.FinishReason)
	}
	if choice.Message.Content != "Hello, how can I help?" {
		t.Fatalf("Message.Content=%q, unexpected", choice.Message.Content)
	}
	if len(choice.ToolCalls) != 0 {
		t.Fatalf("len(ToolCalls)=%d, want 0", len(choice.ToolCalls))
	}
}

func TestResponseFromSessionEmptySteps(t *testing.T) {
	session := &genagent.AgentSession{
		Finished:    true,
		FinalOutput: "done",
		TotalTokens: 10,
	}

	resp := ResponseFromSession(session, "m")
	choice := resp.Choices[0]
	if choice.FinishReason != "stop" {
		t.Fatalf("FinishReason=%q, want stop", choice.FinishReason)
	}
	if choice.Message.Content != "done" {
		t.Fatalf("Message.Content=%q, want done", choice.Message.Content)
	}
}

func TestBuildToolCallChunks(t *testing.T) {
	step := &genagent.AgentStep{
		ToolCalls: []genagent.ToolCall{
			{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"NYC"}`)},
			{ID: "call_2", Name: "get_time", Arguments: json.RawMessage(`{}`)},
		},
	}

	chunks := BuildToolCallChunks(step, "test-model")

	// Expect: 1 role chunk + 2 tool call chunks + 1 finish chunk = 4
	if len(chunks) != 4 {
		t.Fatalf("len(chunks)=%d, want 4", len(chunks))
	}

	// First chunk: role.
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Fatalf("chunk[0] role=%q, want assistant", chunks[0].Choices[0].Delta.Role)
	}

	// Second chunk: first tool call.
	tc0 := chunks[1].Choices[0].Delta.ToolCalls
	if len(tc0) != 1 {
		t.Fatalf("chunk[1] tool_calls=%d, want 1", len(tc0))
	}
	if tc0[0].Index != 0 {
		t.Fatalf("tc0 index=%d, want 0", tc0[0].Index)
	}
	if tc0[0].ID != "call_1" {
		t.Fatalf("tc0 ID=%q, want call_1", tc0[0].ID)
	}
	if tc0[0].Type != "function" {
		t.Fatalf("tc0 type=%q, want function", tc0[0].Type)
	}
	if tc0[0].Function.Name != "get_weather" {
		t.Fatalf("tc0 function.name=%q, want get_weather", tc0[0].Function.Name)
	}
	if tc0[0].Function.Arguments != `{"city":"NYC"}` {
		t.Fatalf("tc0 function.arguments=%q, unexpected", tc0[0].Function.Arguments)
	}

	// Third chunk: second tool call.
	tc1 := chunks[2].Choices[0].Delta.ToolCalls
	if len(tc1) != 1 {
		t.Fatalf("chunk[2] tool_calls=%d, want 1", len(tc1))
	}
	if tc1[0].Index != 1 {
		t.Fatalf("tc1 index=%d, want 1", tc1[0].Index)
	}
	if tc1[0].ID != "call_2" {
		t.Fatalf("tc1 ID=%q, want call_2", tc1[0].ID)
	}

	// Fourth chunk: finish_reason.
	fr := chunks[3].Choices[0].FinishReason
	if fr == nil || *fr != "tool_calls" {
		t.Fatalf("finish_reason=%v, want tool_calls", fr)
	}

	// All chunks share the same ID and model.
	for i, c := range chunks {
		if c.Object != "chat.completion.chunk" {
			t.Fatalf("chunk[%d] object=%q, want chat.completion.chunk", i, c.Object)
		}
		if c.Model != "test-model" {
			t.Fatalf("chunk[%d] model=%q, want test-model", i, c.Model)
		}
		if c.ID != chunks[0].ID {
			t.Fatalf("chunk[%d] ID=%q, want %q", i, c.ID, chunks[0].ID)
		}
	}
}

func TestBuildToolCallChunksNil(t *testing.T) {
	if chunks := BuildToolCallChunks(nil, "m"); chunks != nil {
		t.Fatalf("expected nil, got %d chunks", len(chunks))
	}
	step := &genagent.AgentStep{}
	if chunks := BuildToolCallChunks(step, "m"); chunks != nil {
		t.Fatalf("expected nil for empty step, got %d chunks", len(chunks))
	}
}

func TestMarshalChunk(t *testing.T) {
	chunk := StreamChunk{
		ID:      "chatcmpl-test",
		Object:  "chat.completion.chunk",
		Created: 1000,
		Model:   "m",
		Choices: []StreamChunkChoice{{
			Index: 0,
			Delta: StreamDelta{
				ToolCalls: []StreamToolCallDelta{{
					Index: 0,
					ID:    "call_1",
					Type:  "function",
					Function: &StreamFunctionDelta{
						Name:      "f",
						Arguments: `{}`,
					},
				}},
			},
		}},
	}

	data, err := MarshalChunk(chunk)
	if err != nil {
		t.Fatalf("MarshalChunk: %v", err)
	}

	// Verify it round-trips.
	var decoded StreamChunk
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if decoded.ID != "chatcmpl-test" {
		t.Fatalf("ID=%q, want chatcmpl-test", decoded.ID)
	}
	if len(decoded.Choices) != 1 {
		t.Fatalf("len(Choices)=%d, want 1", len(decoded.Choices))
	}
	tc := decoded.Choices[0].Delta.ToolCalls
	if len(tc) != 1 {
		t.Fatalf("len(ToolCalls)=%d, want 1", len(tc))
	}
	if tc[0].Function.Name != "f" {
		t.Fatalf("Function.Name=%q, want f", tc[0].Function.Name)
	}
}

// TestOpenAIToolsAPI is the acceptance test required by the task definition.
// It verifies end-to-end conversion: OpenAI tools -> agent defs -> tool calls
// -> OpenAI response with tool_calls, and streaming delta events.
func TestOpenAIToolsAPI(t *testing.T) {
	// 1. Convert OpenAI tools to agent tool defs.
	tools := []serve.Tool{
		{
			Type: "function",
			Function: serve.ToolFunction{
				Name:        "search",
				Description: "Search the web",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`),
			},
		},
	}

	defs := ConvertTools(tools)
	if len(defs) != 1 || defs[0].Name != "search" {
		t.Fatalf("ConvertTools produced unexpected defs: %+v", defs)
	}

	// 2. Simulate a supervisor step that produces a tool call.
	step := &genagent.AgentStep{
		StepNum:     1,
		ModelOutput: `{"name":"search","arguments":{"query":"zerfoo"}}`,
		ToolCalls: []genagent.ToolCall{
			{
				ID:        "call_999",
				Name:      "search",
				Arguments: json.RawMessage(`{"query":"zerfoo"}`),
			},
		},
		TokensUsed: 20,
	}

	// 3. Build non-streaming response.
	session := &genagent.AgentSession{
		Steps:       []genagent.AgentStep{*step},
		Finished:    true,
		FinalOutput: "",
		TotalTokens: 20,
		StopReason:  "no_tools",
	}

	resp := ResponseFromSession(session, "test-model")
	if len(resp.Choices) != 1 {
		t.Fatalf("len(Choices)=%d, want 1", len(resp.Choices))
	}
	choice := resp.Choices[0]
	if choice.FinishReason != "tool_calls" {
		t.Fatalf("FinishReason=%q, want tool_calls", choice.FinishReason)
	}
	if len(choice.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls)=%d, want 1", len(choice.ToolCalls))
	}
	if choice.ToolCalls[0].ID != "call_999" {
		t.Fatalf("ToolCall.ID=%q, want call_999", choice.ToolCalls[0].ID)
	}
	if choice.ToolCalls[0].Type != "function" {
		t.Fatalf("ToolCall.Type=%q, want function", choice.ToolCalls[0].Type)
	}
	if choice.ToolCalls[0].Function.Name != "search" {
		t.Fatalf("ToolCall.Function.Name=%q, want search", choice.ToolCalls[0].Function.Name)
	}
	if choice.ToolCalls[0].Function.Arguments != `{"query":"zerfoo"}` {
		t.Fatalf("ToolCall.Function.Arguments=%q, unexpected", choice.ToolCalls[0].Function.Arguments)
	}

	// 4. Build streaming tool call chunks.
	chunks := BuildToolCallChunks(step, "test-model")
	if len(chunks) != 3 { // role + 1 tool call + finish
		t.Fatalf("len(chunks)=%d, want 3", len(chunks))
	}

	// Verify role chunk.
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Fatalf("chunk[0] role=%q, want assistant", chunks[0].Choices[0].Delta.Role)
	}

	// Verify tool call delta.
	tcDelta := chunks[1].Choices[0].Delta.ToolCalls
	if len(tcDelta) != 1 {
		t.Fatalf("chunk[1] tool_calls=%d, want 1", len(tcDelta))
	}
	if tcDelta[0].Function.Name != "search" {
		t.Fatalf("delta function.name=%q, want search", tcDelta[0].Function.Name)
	}

	// Verify finish chunk.
	fr := chunks[2].Choices[0].FinishReason
	if fr == nil || *fr != "tool_calls" {
		t.Fatalf("finish_reason=%v, want tool_calls", fr)
	}

	// 5. Verify chunks serialize as valid JSON.
	for i, c := range chunks {
		data, err := MarshalChunk(c)
		if err != nil {
			t.Fatalf("MarshalChunk(%d): %v", i, err)
		}
		if !json.Valid(data) {
			t.Fatalf("chunk[%d] produced invalid JSON", i)
		}
	}
}
