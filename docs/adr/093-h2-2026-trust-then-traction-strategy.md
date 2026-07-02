# ADR 093: H2 2026 Product Strategy -- Trust, then Traction

## Status
Accepted

## Date
2026-07-02

## Context

A full product review on 2026-07-02 (plan.md, devlog Apr-Jun, benchmarks,
ADRs 068-092, the issue tracker, the codebase, and all satellite repos) found:

1. Engineering capability is world-class and ahead of the 10-year vision's
   Year-1 targets: ~45 architecture builders, 241 tok/s Gemma 3 1B (1.28x
   Ollama), over-RAM 229B inference via mmap, a hardened GPU training stack
   (plan-gpu-training-hardening.md closed its end goal 2026-06-11), and
   per-op verification harnesses (ADR-091).
2. Adoption is zero: 6 GitHub stars, an empty website (zerfoo.github.io is a
   0-byte index.html), launch posts and pricing sheets drafted since March
   2026 but never published, no external contributors or users. The March
   2026 progress report diagnosed accessibility -- not capability -- as the
   bottleneck; the following four months went into capability anyway.
3. Correctness debt blocks a credible launch: CUDA-graph capture-replay
   training silently produces wrong gradients (#878); flash paths have
   replay-lifetime bugs (#870, #865); Gemma 4 edge decode has never produced
   correct output (#757), stalled 10 weeks with a specified fix candidate
   (#766).
4. Operational drag: ~75% of open issues are noise (generated business
   fiction under the retracted E94 epic, "(COMPLETE)" epics left open);
   plan.md's summary contradicts its body; DGX validation chronically
   deferred because the purego darwin-to-linux/arm64 cross-compile blocker
   was never institutionalized away.

## Decision

Adopt the strategy documented in docs/product-strategy-2026-H2.md:
**Trust, then Traction**, executed in phases:

- **P1 Trust:** make every public claim true. Close the capture/replay bug
  cluster (#865 -> #870 -> #878), time-box the Gemma 4 edge fix (demote to
  experimental if the specified candidate fails), finish the kernel-numerics
  tail (#847), and publish an honest verified-model matrix (~10 models with
  parity + benchmark evidence). Verification gates marketing claims.
- **P2 Traction:** launch for real. Website + docs + examples + reproducible
  benchmarks, then publish the drafted Show HN / r/golang / newsletter posts;
  CFPs; contributor pipeline. Success is measured in external users.
- **P3 Moat:** deepen only differentiated capability -- mmap/over-RAM (E125),
  LTX-2 DiT-first (ADR-092) as the SINGLE capability-expansion front, and
  fast-follow frontier model releases through the verified matrix.

Operating rules (full text in the strategy doc, Part 5):
1. Verification is the gate for claims.
2. One capability-expansion front at a time.
3. Time-box correctness hunts; demote to experimental rather than
   hypothesis-hunt indefinitely.
4. The tracker is the truth; aspirational planning lives in strategy docs,
   not GitHub issues.
5. Consumer workloads harden the framework at the contract level (unchanged).

Additionally: docs/plan.md is scoped to ONE phase at a time. Each phase's
plan ends with a task to plan the next phase, reading the strategy doc as
the higher-level source of truth. This keeps the plan executable and
prevents the 4,400-line drift observed in the previous plan.

## Consequences

Positive:
- Launch happens against a foundation whose claims survive scrutiny.
- The issue tracker and plan become legible to new contributors and future
  sessions.
- Explicit non-goals (no new backends, no enterprise motion, no throughput
  moonshots in H2) protect focus; parked work is labeled, not deleted.

Negative / accepted costs:
- LTX-2 progress slows (it becomes Phase 3 of H2 instead of the active
  front).
- Gemma 4 edge may ship as "experimental" rather than fixed if the
  time-boxed candidate fails, sacrificing a marketing claim for honesty.
- Revenue remains $0 by design through 2026; VISION.md Year-3 economics
  start only after adoption exists.

Related: docs/product-strategy-2026-H2.md (the full strategy),
docs/VISION.md (unchanged 10-year thesis), ADR-090 (enterprise extraction,
ships with the Phase 2 major bump), ADR-091 (verification gates), ADR-092
(LTX-2 scope).
