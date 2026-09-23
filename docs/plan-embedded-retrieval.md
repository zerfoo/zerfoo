# Generic embedded retrieval work plan

**Owner direction:** Put generic retrieval in Zerfoo so Go systems can embed
it. Skill and tool-operation search are example consumers. The one-skill agent
interface, catalog importers, and tool authorization stay in consumer code.

| Stage | Deliverable | Acceptance | State |
|---|---|---|---|
| R1 | In-process search, fetch, and evaluation API | Stable IDs, BM25, optional dense encoder and reranker, bounded shortlist, deterministic results, ranking metrics, unit evidence | Implemented in `retrieval`; not yet benchmarked at catalog scale |
| R2 | Generic benchmark harness and example adapters | Reproducible SkillRet v1.1 and ToolRet/operation inputs without domain types in `retrieval`; source hashes, held-out qrels, latency and memory report | Planned |
| R3 | Contextual inference | Load a pinned encoder checkpoint, expose final token states and configured pooling/prompts, match reference vectors and rankings on CPU and Spark GPU | Planned |
| R4 | Native contrastive training | In-batch/multi-positive InfoNCE, hard negatives, explicit query/doc formatting, checkpoint/resume, held-out source and API-family results | Engine-based in-batch loss implemented; encoder training and qualification remain planned |
| R5 | Advanced ranking | Train and benchmark a reranker; test late-interaction token vectors only if R1–R4 errors justify it | Planned |
| R6 | One-skill open-weights specialization | Train from licensed skill data using the generic pipeline; version checkpoint, model card, provenance, no-skill evaluation, and a Go loading example | Experimental Qwen3 adapter released separately in `zerfoo/skillrouter`; independent queries, no-skill evaluation, and Go loading remain planned |
| R7 | Consumer integrations | Example one-skill search/fetch adapter and Zatiti integration outside the generic package; end-to-end task success and no-result calibration | Planned, pending consumer contract |

## Immediate evidence and constraints

- `inference/inference.go` currently mean-pools token lookup rows in
  `Model.Embed`. This is a non-contextual baseline only.
- `inference/arch_bert.go` already builds a bidirectional attention graph for
  classification. It is a candidate for extracting contextual hidden states,
  but its current output is classification logits, not retrieval vectors.
- `training/default_trainer.go`, `training/optimizer`, and `training/lora`
  provide reusable training pieces. `training/loss.InBatchContrastive` now
  supplies a generic multi-positive in-batch objective, but no qualified text
  encoder training pipeline was found.
- The first consumer can use the R1 API with local BM25 while R2–R5 are built.
  No pretrained-model parity or quality claim follows from that API alone.
- A DGX Spark run trained the separate `zerfoo/skillrouter` Qwen3 adapter on
  description-derived pairs. Its early held-out results establish the Python
  training path only; they do not validate native Zerfoo inference or real
  task-worded retrieval.

## Next implementation slice

Pin SkillRet dataset revision and reference Qwen3-Embedding checkpoint; build
an evaluator that records exact source IDs, splits, model versions, candidate
sets, ranking metrics, latency, and memory. Run BM25 first. Then add contextual
encoder output and test its vectors against a Python reference before fitting
or shipping a model. Train the one-skill checkpoint only after the generic
path works, and release weights only with a reviewable model card and license
audit. Coordinate the Go call shape and latency target with the
active Zatiti session through `ajent.social`.
