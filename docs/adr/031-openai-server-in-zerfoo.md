# ADR 031: OpenAI-Compatible Inference Server Lives in Zerfoo

## Status
Accepted

## Date
2026-03-12

## Context
The project needs an OpenAPI-compatible inference server. Two locations were
considered: the Zerfoo repository (github.com/zerfoo/zerfoo) or the Zonnx
repository (github.com/zerfoo/zonnx). The question is which offers a cleaner
architecture for hosting the server.

Key facts:
- Zerfoo already has a `serve/` package with OpenAI-compatible endpoints:
  POST /v1/chat/completions, POST /v1/completions, GET /v1/models.
- The server supports SSE streaming, batch scheduling, and speculative decoding.
- It is wired into the CLI via `cmd/cli/serve.go` (the `serve` subcommand).
- Zonnx is an ONNX model converter/optimizer. It must not import Zerfoo
  (architectural boundary verified by `make verify-architecture`).
- The inference API (Load, Generate, Chat, Embed) lives in Zerfoo's
  `inference/` package.

## Decision
The OpenAI-compatible inference server stays in Zerfoo's `serve/` package.

Zonnx is architecturally unsuitable because:
1. It cannot import Zerfoo (the boundary is enforced by CI).
2. It has no inference runtime -- it converts models to ZMF format.
3. Placing a server there would require duplicating or re-importing the entire
   inference stack.

The existing `serve/` package already implements the core OpenAI endpoints.
The remaining work is to add missing OpenAI API surface: embeddings, model
deletion, health, and an OpenAPI spec document.

## Consequences
Positive:
- Zero new dependencies or architectural changes.
- Reuses proven code (server, batch scheduler, streaming).
- Single binary: `zerfoo serve` handles both inference and API.
- CLI already has the `serve` subcommand wired up.

Negative:
- Zonnx users who want a server must also install Zerfoo (this is expected).
- The serve package grows as more OpenAI endpoints are added.
