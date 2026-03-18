// Package agent adapts the generate/agent agentic loop to the
// OpenAI-compatible chat completions API, translating between OpenAI
// tool definitions and the internal ToolRegistry/Supervisor types.
package agent

import (
	"encoding/json"
	"fmt"
	"time"

	genagent "github.com/zerfoo/zerfoo/generate/agent"
	"github.com/zerfoo/zerfoo/serve"
)

// ConvertTools converts OpenAI Tool definitions into generate/agent ToolDef
// values suitable for registration in a ToolRegistry.
func ConvertTools(tools []serve.Tool) []genagent.ToolDef {
	defs := make([]genagent.ToolDef, len(tools))
	for i, t := range tools {
		defs[i] = genagent.ToolDef{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		}
	}
	return defs
}

// ToolCallFromAgent converts a generate/agent ToolCall into an OpenAI-format
// serve.ToolCall suitable for inclusion in a ChatCompletionChoice.
func ToolCallFromAgent(tc genagent.ToolCall) serve.ToolCall {
	return serve.ToolCall{
		ID:   tc.ID,
		Type: "function",
		Function: serve.ToolCallFunction{
			Name:      tc.Name,
			Arguments: string(tc.Arguments),
		},
	}
}

// ToolCallsFromStep extracts all tool calls from an AgentStep and converts
// them to the OpenAI response format.
func ToolCallsFromStep(step *genagent.AgentStep) []serve.ToolCall {
	if step == nil || len(step.ToolCalls) == 0 {
		return nil
	}
	calls := make([]serve.ToolCall, len(step.ToolCalls))
	for i, tc := range step.ToolCalls {
		calls[i] = ToolCallFromAgent(tc)
	}
	return calls
}

// ResponseFromSession builds a ChatCompletionResponse from a completed
// AgentSession. The last step's tool calls (if any) are included in the
// response choice. If the session ended with tool calls, the finish_reason
// is "tool_calls"; otherwise it is "stop".
func ResponseFromSession(session *genagent.AgentSession, model string) serve.ChatCompletionResponse {
	choice := serve.ChatCompletionChoice{
		Index:        0,
		FinishReason: "stop",
	}

	if len(session.Steps) > 0 {
		last := session.Steps[len(session.Steps)-1]
		choice.Message = serve.ChatMessage{
			Role:    "assistant",
			Content: last.ModelOutput,
		}

		if len(last.ToolCalls) > 0 {
			choice.FinishReason = "tool_calls"
			choice.Message.Content = ""
			choice.ToolCalls = ToolCallsFromStep(&last)
		}
	} else {
		choice.Message = serve.ChatMessage{
			Role:    "assistant",
			Content: session.FinalOutput,
		}
	}

	return serve.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []serve.ChatCompletionChoice{choice},
		Usage: serve.UsageInfo{
			PromptTokens:     0,
			CompletionTokens: session.TotalTokens,
			TotalTokens:      session.TotalTokens,
		},
	}
}

// StreamToolCallDelta represents an incremental tool call update for
// SSE streaming, following the OpenAI delta format.
type StreamToolCallDelta struct {
	Index    int                     `json:"index"`
	ID       string                  `json:"id,omitempty"`
	Type     string                  `json:"type,omitempty"`
	Function *StreamFunctionDelta    `json:"function,omitempty"`
}

// StreamFunctionDelta holds incremental function name/arguments in a
// streaming tool call delta.
type StreamFunctionDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// StreamChunk is the SSE data payload for a streaming chat completion
// chunk that may contain tool call deltas.
type StreamChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []StreamChunkChoice  `json:"choices"`
}

// StreamChunkChoice is a single choice within a streaming chunk.
type StreamChunkChoice struct {
	Index        int                  `json:"index"`
	Delta        StreamDelta          `json:"delta"`
	FinishReason *string              `json:"finish_reason"`
}

// StreamDelta holds the incremental content or tool calls for one
// streaming chunk.
type StreamDelta struct {
	Role      string                `json:"role,omitempty"`
	Content   string                `json:"content,omitempty"`
	ToolCalls []StreamToolCallDelta `json:"tool_calls,omitempty"`
}

// BuildToolCallChunks builds the sequence of SSE chunks needed to stream
// tool calls from an AgentStep. Per the OpenAI spec, the first chunk for
// each tool call includes id, type, and function name; subsequent chunks
// carry argument fragments. This implementation emits one initial chunk
// per tool call (with the full arguments in a single fragment).
func BuildToolCallChunks(step *genagent.AgentStep, model string) []StreamChunk {
	if step == nil || len(step.ToolCalls) == 0 {
		return nil
	}

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	now := time.Now().Unix()
	chunks := make([]StreamChunk, 0, len(step.ToolCalls)+1)

	// Initial role chunk.
	chunks = append(chunks, StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: now,
		Model:   model,
		Choices: []StreamChunkChoice{{
			Index: 0,
			Delta: StreamDelta{Role: "assistant"},
		}},
	})

	// One chunk per tool call with id, type, name, and arguments.
	for i, tc := range step.ToolCalls {
		chunks = append(chunks, StreamChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: now,
			Model:   model,
			Choices: []StreamChunkChoice{{
				Index: 0,
				Delta: StreamDelta{
					ToolCalls: []StreamToolCallDelta{{
						Index: i,
						ID:    tc.ID,
						Type:  "function",
						Function: &StreamFunctionDelta{
							Name:      tc.Name,
							Arguments: string(tc.Arguments),
						},
					}},
				},
			}},
		})
	}

	// Final chunk with finish_reason.
	reason := "tool_calls"
	chunks = append(chunks, StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: now,
		Model:   model,
		Choices: []StreamChunkChoice{{
			Index:        0,
			Delta:        StreamDelta{},
			FinishReason: &reason,
		}},
	})

	return chunks
}

// MarshalChunk serialises a StreamChunk as a JSON byte slice suitable
// for an SSE "data:" line.
func MarshalChunk(chunk StreamChunk) ([]byte, error) {
	return json.Marshal(chunk)
}
