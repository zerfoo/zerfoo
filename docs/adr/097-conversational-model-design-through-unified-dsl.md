# ADR 097: Shared component registry and model-definition DSL

Status: Accepted. Product-specific design and orchestration are maintained separately.

The open-source framework provides a shared component registry, versioned model
definitions, allocation-free shape validation, and compilation through existing
Zerfoo graph builders. The registry distinguishes construction, inference,
training, device, precision and execution-mode support. Registration is not
qualification, and inference support does not imply backward support.

Definitions describe named ports, branches, merges and shared parameters. The
compiler rejects unsupported combinations rather than silently changing them.
GGUF remains the weights format; model metadata binds the canonical definition
to its tensors. See [DSL coverage](../model-definition-dsl.md) for implemented
operators, grammar limits and executable evidence.

The conversational application, its planning prompts and research retrieval are
not part of this public repository. The application consumes these framework APIs.
