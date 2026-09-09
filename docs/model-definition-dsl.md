# Model definitions and executable coverage

ADR 097 introduces `dsl.Definition` version 1 as the composition contract.
`zerfoo-create plan_create` accepts it as `definition`; the service stores a
version 2 plan and its `definition_sha256`. The calling agent designs the graph.
The service validates and compiles it through registered Zerfoo graph builders.

## Example

This linear classifier consumes two numeric features and produces three logits:

```json
{
  "version": 1,
  "name": "linear_classifier",
  "inputs": [{"name": "features", "dtype": "float32", "shape": [-1, 2]}],
  "parameters": [
    {"name": "weight", "dtype": "float32", "shape": [2, 3], "initializer": "he_normal"},
    {"name": "bias", "dtype": "float32", "shape": [3], "initializer": "zeros"}
  ],
  "nodes": [{
    "name": "head", "operator": "Dense", "version": 1,
    "inputs": {"x": {"node": "features", "port": "value"}},
    "parameters": {"weights": "weight", "bias": "bias"}
  }],
  "outputs": [{"name": "logits", "source": {"node": "head", "port": "y"}}]
}
```

Connections name both node and port. Reusing a global parameter name in different
nodes shares the actual parameter and sums its gradients; optimizers see it once.
Branches and merges execute every edge. Multiple graph outputs are returned by
name; `Backward` accepts named output seeds and sums their contributions. Calls
on an executable are sequential; returned tensors live until its next invocation.

Canonical ordering removes declaration-order differences. SHA-256 binds the
canonical JSON, including initializer and component versions. Definitions cannot
supply code, shell commands or dynamic plugins. Scalar attributes must match the
registered component schema. Diagnostics expose code, path and message through
MCP error `structuredContent` and Go `DiagnosticError`.

## Grammar and limits

Version 1 supports float32, deterministic rank-2 operators and one tensor output
per operator. A graph may declare up to 16 inputs and 16 outputs, 256 nodes and
512 parameters. Only the first input axis may be dynamic (`-1`), with a runtime
batch of 1–256. Elementwise shapes must match exactly; broadcasting is rejected.
The serialized definition is limited to 1 MiB and its estimated tensor storage
to 16 million elements, counting parameter values and gradients. These are
preflight bounds, not a hard process-memory cap or a runtime-workspace estimate.

The classifier service further requires one feature input and one logits output
with the dataset's feature and class widths. A final Softmax is rejected because
the existing loss consumes logits. `hidden_dims` generates Dense/ReLU definitions;
explicit definitions cannot also specify hidden widths. Legacy DSL execution
rejects multi-parent or multi-output graphs it previously silently truncated.

## Coverage and evidence

The registry distinguishes discovery, implemented support and qualification.
`ValidateExecution` checks operation, device, precision and execution mode before
a conversational training plan is accepted. The capability response is version 2.
This change advertises CPU float32 eager composition/training as **implemented**;
it does not mark every composition or device verified from fixture tests.

| Route | Implemented behavior | Executable evidence |
| --- | --- | --- |
| Linear, Dense, ReLU, Add, Mul, Sub, MatMul, Softmax | Named ports, shape preflight, existing node builders | `TestComposableOperatorValues` checks eight scalar forward references |
| Residual Dense with tied parameters and multiple outputs | Forward and accumulated backward gradients | `TestCompileResidualSharedParametersAndOutputs` |
| Nonlinear tied residual graph | Dense/ReLU/Mul/Sub/Softmax backward | `TestResidualGradientIndependentReference` compares a separate float64 expression using central differences |
| Classifier DSL training and bundle | Existing cross-entropy/AdamW, exact architecture reload | `TestDSLClassifierLearnsAndReloadsResidualGraph`, creation service DSL test and subprocess lifecycle |
| Generic composition GGUF | Canonical definition, unique tensors, hashes and named output reconstruction | `model/dsl/gguf_test.go` |
| Registered architecture reference | Identity-bound decoded tensors passed to the existing inference factory | `inference/definition_test.go` uses a real tiny Llama forward |
| Other registered operators/architectures | Discoverable as unverified; no implicit backward or composition support | Existing architecture-specific tests do not qualify this new DSL route |

A generic `Executable.WriteGGUF` records `zerfoo.dsl.v1`, the canonical definition
and identity, unique F32 tensors and supplied lifecycle metadata. `ReadGGUF`
checks content identity when supplied, tensor contracts and bounded input size.
Classifier bundles also bind their existing preprocessing, labels, training and
evaluation metadata to the definition identity. GGUF remains the weight format;
this is not ZMF and it is not an optimizer/RNG checkpoint.

`dsl.ArchitectureReference` and `inference.CompileArchitectureReference` provide
the registered-architecture route. A reference pins configuration, component
version and decoded parameter-set identity, then invokes the existing builder.
Builder-specific configuration validation still applies. This API is currently
an inference construction route, not a generic conversational training tool.

## Expansion gates

Add an operator by registering its existing builder and a versioned descriptor
with named ports, parameter slots, scalar attributes and allocation-free shape
rules. Application capability discovery needs no additional recipe. Research
cards use those identifiers; eligibility is not paper-reproduction evidence.

Remaining qualification includes GPU/mixed precision, native multi-output
operators, stochastic blocks, reusable block syntax, architecture-specific
configuration/resource contracts, and training/export coverage for the remaining
architectures. Each needs operation-specific numerical and lifecycle evidence
before claiming support. No GPU, checkpoint/resume, web chat or hosted deployment
coverage is added by this implementation.
