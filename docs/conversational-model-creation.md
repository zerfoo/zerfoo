# Model creation from a coding agent

`zerfoo-create` is a local application executable exposing the same operations
through MCP stdio and JSON CLI calls. The calling coding agent is the model
designer: it interprets the user's objective, inspects the dataset, retrieves
eligible research, and submits a concrete architecture and training plan.
There is no web chat, hosted compute, or second LLM API dependency.

## Build and connect

```sh
go build -o zerfoo-create ./cmd/zerfoo-create
./zerfoo-create --state ./projects --data-root ./datasets mcp
```

Configure your MCP host to launch the built executable with those arguments.
Use absolute paths in host configuration. Standard output contains only
newline-delimited JSON-RPC messages. The transport follows the
[MCP stdio specification](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports).
The server negotiates protocol version `2025-11-25` and advertises tools.
A real subprocess protocol client is tested; named coding-agent host/version
compatibility has not yet been qualified.

An initial user request can be:

> Inspect iris.csv and create a model that predicts species. Explain the
> architecture and evidence, then train within the supported limits. Report
> validation quality and show a prediction using the saved model.

The agent uses this sequence:

1. `capabilities` establishes the actual executable components and limits.
2. `project_create` records the objective and returns a project ID.
3. `dataset_inspect` snapshots a relative CSV path under `--data-root`, records
   target/schema and isolated partitions, and returns a dataset ID. Optional
   group/time/split fields select the existing split policies. Defaults use
   stratification; seed is an explicit tool argument.
4. `research_search` retrieves eligible evidence. An empty catalog means no
   retrieved evidence, not permission to invent sources.
5. `plan_create` submits dataset/project IDs, rationale, evidence IDs, hidden
   widths, epochs, batch size, learning rate and seed. The service validates
   the model and stores an immutable version 1 plan with evidence snapshots.
6. `run_start` takes the plan ID and an idempotency key. Repeating the same
   request returns the same run; changing the plan under that key fails.
7. `run_status` reports progress, terminal status, validation metrics and the
   saved bundle identity. `run_cancel` requests cooperative cancellation.
8. `model_predict` takes a successful run ID and 1–256 raw numeric rows in the
   recorded feature order, applying the saved preprocessing and exact bundle.

Read any project/plan again with `project_get`/`plan_get`. IDs and records remain
in the state directory across MCP and CLI sessions. Keep the returned IDs in
the agent's project notes.

Every tool is also a CLI operation, with exactly one JSON argument:

```sh
./zerfoo-create --state ./projects capabilities '{}'
./zerfoo-create --state ./projects project_create '{"objective":"Predict species"}'
./zerfoo-create --state ./projects run_status '{"id":"RUN_ID"}'
```

## Supported model designs

The first executable capability is numeric classification with linear layers,
up to four hidden ReLU layers (width 1–1024), softmax, cross-entropy and AdamW.
The agent chooses topology and supported training parameters; the service does
not translate arbitrary paper descriptions into new operators. Unsupported
fields and evidence IDs fail validation. The growing Zerfoo capability set
must be registered and verified before new architectures become executable.

Plans allow 1–200 epochs, batch size 1–256 and learning rate in (0,1]. Training
has a two-minute cooperative time limit and one worker per state directory.
These limits are not an OS memory cap. Run the binary in a user-configured
container/cgroup for hard memory/process limits. This initial worker uses CPU;
GPU and remote worker dispatch are not exposed by this application yet.

Validation results are measured on held-out validation rows. No test partition
is read by this workflow, and a successful training run is not a production
qualification verdict. A prediction endpoint and automated deployment are not
included; `model_predict` provides local prediction through MCP/CLI.

## Distilled paper catalog adapter

Pass `--library /path/to/catalog.json` before the operation. Until the external
library schema is available, the adapter accepts this versioned format:

```json
{
  "version": 1,
  "cards": [
    {
      "id": "YOUR_STABLE_PAPER_ID",
      "title": "TITLE_FROM_YOUR_LIBRARY",
      "url": "https://your-source.example/paper",
      "version": "PINNED_SOURCE_VERSION",
      "section": "SUPPORTING_SECTION",
      "summary": "Distilled claim, assumptions and limitations",
      "components": ["linear", "relu", "softmax", "cross_entropy", "adamw"]
    }
  ]
}
```

This is a schema illustration, not a supplied research source. No papers are
bundled or fabricated. The adapter searches title/summary text and filters out
cards requiring unsupported components. Source URL/version/section and stable
IDs are required. Matched evidence remains untrusted data; it is never executed
or given authority to change paths, limits, or model capabilities. The host
agent is responsible for applicability reasoning; retrieval alone does not
verify a scientific claim or establish reproduction of a paper.

## Persistence and interruption

The application stores metadata in bbolt, immutable CSV snapshots and GGUF
bundles on the user's filesystem. CSV reads are constrained by `os.Root` to
the configured data root; replay verifies the snapshot against the manifest.
The state directory is trusted, private application storage for one OS user.
MCP exposes no network listener or arbitrary command execution.

A separate worker process performs training; normal MCP stdin closure or CLI
exit does not cancel it. An OS lock serializes workers. Closing a terminal or
container in a way that kills all its processes can still kill the worker.
`run_recover` marks an abandoned queued/running job interrupted only when its
worker lock is free. It does not resume optimizer state. Starting over requires
a new idempotency key; stale requests cannot silently launch another run.
A queued launch interrupted before worker startup also needs this explicit
recovery operation. No automatic restart or checkpoint/resume is claimed.

Back up the whole state directory, not just the database. Model publication
inherits the existing bundle's atomic-rename behavior; power-loss durability
of dataset/bundle files remains an outstanding refinement task.

## Verification

```sh
go test -race -count=1 ./cmd/zerfoo-create
go build -o /tmp/zerfoo-create ./cmd/zerfoo-create
python3 cmd/zerfoo-create/testdata/lifecycle.py /tmp/zerfoo-create
```

The subprocess check uses the committed Iris fixture, initializes MCP, discovers
all 12 tools, starts a real plan, retries the start, closes the MCP process,
waits from fresh CLI processes, and checks the saved model predicts Iris-setosa.
It verifies artifact identity and reports actual validation metrics. It does
not substitute a mocked trainer or claim host-specific integration coverage.
