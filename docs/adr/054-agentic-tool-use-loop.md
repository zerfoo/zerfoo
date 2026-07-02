# ADR 054: Agentic Tool-Use and Loop Supervisor

## Status
Accepted

## Date
2026-03-17

## Context
Wolf's decision engine will evolve from signal-based trading to agentic trading:
models that autonomously decide when to gather information (call tools), reason
about market state, and execute multi-step plans. OpenAI function-calling and
Anthropic tool-use are the emerging standards. Gartner predicts 33% of enterprise
software will embed agentic AI by 2028.

Zerfoo's serve package already produces OpenAI-compatible chat completions. The
extension needed is: (1) structured tool-call output in the token stream, and (2)
an agentic loop supervisor that manages multi-turn tool execution.

## Decision
Implement agentic infrastructure in generate/agent/ and serve/agent/:

Tool Registry (generate/agent/tools.go):
- ToolSpec: JSON schema for tool name, description, and parameter schema
- ToolRegistry: maps tool names to Go functions (func(params json.RawMessage) (string, error))
- Wolf pre-registers tools: GetMarketData, GetOrderBook, GetPortfolio, SubmitOrder,
  GetEarningsCalendar, SearchNews

Function-Call Decoding (generate/agent/function_call.go):
- Grammar-constrained decoding: when model outputs "<tool_call>" token, switch to
  JSON grammar for the rest of the generation (integrates with existing grammar-guided
  decoding in ADR-038)
- Parses JSON tool call: {"name": "GetMarketData", "arguments": {"symbol": "BTCUSD"}}
- Returns ToolCall struct to the agentic loop

Agentic Loop Supervisor (generate/agent/supervisor.go):
- MaxIterations: configurable limit (default 10) prevents infinite loops
- Step: Forward pass -> detect tool call or EOS -> if tool call, execute tool,
  append result to context, continue; if EOS, return final response
- Timeout: per-step timeout (default 30s); global timeout (default 5min)
- Budget tracking: token budget decrements each step; halts if exceeded

Wolf Integration (generate/agent/wolf_agent.go):
- Pre-built Wolf trading agent: system prompt + Wolf tool registry
- Observability: each agent step logged with tool name, latency, result summary
- Safety gate: SubmitOrder tool requires Wolf risk system approval before execution
  (synchronous gRPC call to Wolf's risk service)

OpenAI Compatibility (serve/agent/openai_adapter.go):
- /v1/chat/completions with tools parameter activates agentic mode
- Response format matches OpenAI function calling JSON schema
- Stream mode: tool_calls delta events interleaved with content deltas

## Consequences
Positive:
- Wolf agents can autonomously gather market data, reason, and execute trades
  without hardcoded decision logic
- OpenAI-compatible tool-use means Wolf can use any model supported by Zerfoo
- MaxIterations + timeout + budget provide safety bounds on runaway agents

Negative:
- Agentic loops have unpredictable latency; incompatible with hard real-time trading
  constraints (use for pre-market analysis, not HFT)
- Tool execution is synchronous; async tool calls require significant additional
  complexity (deferred to Year 5)
- LLM-based trading agents increase regulatory risk (MiFID II, SEC algorithmic
  trading rules); legal review required before live deployment
