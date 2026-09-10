# ADR 097: Conversational model design through a shared registry and DSL

Status: Accepted. Shared registry, bounded composition compiler, conversational
plan migration and GGUF reconstruction implemented. Architecture-wide training
and device qualification remain incremental; see [coverage](../model-definition-dsl.md).

Date: 2026-09-09

## Context

Conversational model creation should let a developer describe a task and have
an agent design, train, evaluate and eventually serve a model on user-controlled
hardware. The developer's distilled research library will describe architectures
and methods supported by Zerfoo and grow as executable support grows.

The initial `zerfoo-create` application exposes MCP and JSON CLI operations over
persistent projects, dataset snapshots, plans and background jobs. Its model
specification is currently a classifier configuration with hidden-layer widths;
its capability and research-component lists are maintained in the application.
Extending this approach architecture by architecture would create a second,
incomplete account of what Zerfoo supports.

Zerfoo already has `model/dsl`, with layer definitions and graph connections.
However, the current DSL recognizes a small fixed layer set, receives input and
output dimensions separately, and has execution paths that consume only the
first parent and return the last output. Its dedicated execution/training
implementation is not a general representation of every supported architecture.
It is a foundation to extend, not evidence that arbitrary compositions work.

Architecture definition and serialized weights are separate concerns. The DSL
is not the retired ZMF format. GGUF remains the sole model weight format.

## Decision

Use a shared, versioned capability registry and a complete, validated DSL as the
model-design contract for conversational creation. Compile definitions through
Zerfoo's existing operators, graph mechanisms and architecture builders. Do not
maintain an independent implementation of model mathematics in the application
or a parallel DSL runtime.

### 1. One authoritative capability registry

Extend and reconcile existing operator and architecture registration mechanisms
rather than adding a competing application-owned catalog. Each descriptor must
identify its stable name/version, configuration schema, named input/output ports,
tensor types and shape constraints, parameter names and sharing rules, and the
builder that implements it.

Describe support by operation and execution conditions: construction, inference,
training/backward, export/reload, checkpoint/resume where available, device,
precision and execution mode. Attach references to verification evidence and its
runtime/component revision. Inference support must never imply training support;
registration alone must never imply qualification. Unverified paths remain
explicitly unverified and unavailable for claims requiring verified support.

The MCP and CLI capability tools derive their answers from this registry. The
research adapter uses the same identifiers to determine executable eligibility.
Adding a supported architecture must not require another hard-coded application
recipe or tool handler.

### 2. A complete architecture definition

Extend the DSL to represent named tensor inputs and outputs, types and shapes
(including explicitly constrained dynamic dimensions), typed configuration,
connections to named ports, branching and merge operations, residual paths,
shared parameters, and reusable blocks or registered architecture references.

Every edge and declared output must have defined execution semantics. Reject
unsupported graph structures instead of ignoring additional parents or outputs.
Validate configuration, shape compatibility, parameter identity, connectivity,
resource bounds and execution requirements before allocating large models or
starting a run. The supported graph grammar must be explicit; declaring a DSL
construct does not make it executable.

Definitions must have a versioned serialization and canonical content identity.
They contain no executable shell commands, arbitrary Go source or dynamically
loaded code supplied by the agent. Custom operators require a separate reviewed
implementation and registration step.

### 3. Compile to existing Zerfoo execution

Resolve DSL components through registered builders into Zerfoo's computation
graph. Reuse existing layers, training/autograd and optimizer infrastructure;
all tensor arithmetic continues through `compute.Engine`. Eliminate duplicated
DSL math as its supported paths migrate, retaining compatibility adapters where
needed. Do not silently alter the meaning of existing definitions.

There are two creation routes:

- Configure a registered architecture using its validated configuration schema.
- Compose registered operators and blocks into a custom DSL architecture.

Both routes produce an inspectable, versioned definition and use the same
validation, execution and artifact contracts. Novel combinations require
composition-level verification even when their individual operators are tested.

### 4. The agent proposes; the service validates

The existing coding agent interprets the user's objective, inspects data and
hardware, retrieves applicable evidence, and proposes a concrete DSL definition
plus a training/evaluation plan. The service returns structured diagnostics for
unsupported components, shape errors, missing training/device support and
resource violations so the agent can revise the proposal before execution.

Research cards must reference stable component or architecture identifiers and
pin source provenance, assumptions and limitations. Persist the cited evidence
snapshot with the plan. Paper inclusion is not executable support or proof of
reproduction. Retrieved text is untrusted data and cannot override authorization,
resource bounds or validation. Unknown requirements become explicit capability
gaps; the agent may address them through a separate coding task, not silently
substitute a different architecture and report success.

The service need not use a second LLM to perform planning. MCP and CLI remain
adapters to one application service. A web chat interface is out of the current
implementation scope.

### 5. Bind architecture to the complete model lifecycle

Keep the architecture definition distinct from learned tensors and run state.
A versioned bundle manifest binds its canonical identity to GGUF weights,
preprocessing/tokenization references as applicable, input/output schemas and
label mappings, component/runtime versions, training configuration and evaluation
records. A DSL definition alone is not a trained model or a resumable checkpoint.

The loader must reconstruct the exact selected architecture and parameter
sharing, validate tensor names/shapes/types and content identities, and reject
missing or incompatible components. Custom compositions must not depend on an
unrecorded local code change. Optimizer and RNG state belong to the checkpoint
contract; weights alone cannot satisfy resume claims.

Model-definition metadata does not introduce a new weight format or revive ZMF.
GGUF and the surrounding manifest carry the supported deployment contract.

### 6. Verify advertised coverage

Build a capability matrix and executable tests linking each advertised
architecture/operation/device combination to evidence. Cover definition
validation, shape inference, actual graph construction, inference, and, where
claimed, gradients/training, export/reload, prediction equivalence and recovery.
Include multi-input/output and parameter-sharing cases, unsupported combinations,
malformed definitions, incompatible artifacts and bounded allocation failures.

Keep independent numerical references and the existing gradient/oracle gates.
A falling loss or matching CPU/GPU result alone does not establish correctness.
Require named executed tests with nonzero counts; absent or skipped required
checks leave support unverified. Claims are scoped to tested component versions,
compositions and execution modes, not generalized from one classifier fixture.

## Implementation sequence

1. Inventory existing operator registries, architecture builders and DSL paths;
   document the supported operation/device matrix and gaps without inflating it.
2. Establish shared descriptors, versioned DSL schemas and validation diagnostics.
3. Implement the unified compiler and prove graph semantics with a small verified
   composition set, including the existing classifier path.
4. Migrate conversational plan/capability tools and research eligibility onto the
   registry and DSL. Preserve readable existing plans and explicit versioning.
5. Extend architecture/bundle reconstruction and train/export qualification across
   the remaining registered architectures. Expand advertised capabilities only
   as their applicable tests pass.

This sequencing replaces the approach of growing conversational creation through
an application-maintained list of individual recipes. It does not waive the
training isolation, artifact integrity or verification requirements in ADR 096.

## Alternatives considered

- **Keep adding application recipes:** useful for the first working slice, but
  duplicates capability knowledge and cannot express general architectures.
- **Use the current DSL unchanged:** insufficient graph semantics, execution
  coverage and lifecycle integration; would overstate present support.
- **Generate arbitrary model code at run time:** harder to validate, reproduce
  and constrain; new operator implementation remains a separate coding workflow.
- **Define a new model/weight format:** unnecessary; extend architecture metadata
  while preserving GGUF as the weight format.

## Consequences

Conversational creation can grow with Zerfoo's actual capabilities through one
model-definition path. Designs become inspectable and reproducible, and failures
can identify precise missing components instead of generic unsupported recipes.

This requires substantive DSL/compiler and registry work before claiming broad
architecture coverage. Some existing inference-only architectures will remain
ineligible for training until backward and lifecycle support are verified. The
initial classifier workflow remains usable during migration; accepting this ADR
does not declare those broader capabilities implemented.

Packaging stays focused on a binary or user-run container. Users choose local
or cloud hardware; Zerfoo does not become a hosted compute or deployment platform
through this decision. A usable prediction endpoint remains a lifecycle goal,
with broader infrastructure setup left to the user's coding agent.

## References

- [Model-definition DSL](../../model/dsl/dsl.go)
- [Current DSL graph builder](../../model/dsl/graph.go)
- [Conversational creation implementation](../conversational-model-creation.md)
- [Bounded creation lifecycle contract](096-model-creation-reference-contract.md)
- [Gradient and independent-oracle verification](091-gradcheck-pytorch-oracle-verification.md)
