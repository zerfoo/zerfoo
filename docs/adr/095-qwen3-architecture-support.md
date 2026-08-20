# ADR 095: Add Qwen 3 architecture support during Phase 1

## Status
Accepted

## Date
2026-08-20

## Context

Zerfoo supports the `qwen2` and `qwen_vl` architectures. It has no Qwen 3
support of any kind: there is no `qwen3` string in the Go source of zerfoo or
zonnx, `DefaultArchConfigRegistry` (`inference/arch_config.go`) registers no
such parser, and `buildArchGraph` (`inference/load_gguf.go`) has no such case.
A Qwen3 GGUF therefore falls through to the `default:` branch and fails with
`unsupported architecture "qwen3"`. That failure is clean -- there is no
silent misload through the `qwen2` builder -- but the capability is absent.

This was surfaced by a direct founder question on 2026-08-20 ("do we have
Qwen 3.8 27B support?"). Two things complicate the ask:

1. **The named model could not be verified to exist.** Qwen's released line
   runs Qwen1.5 -> Qwen2 -> Qwen2.5 -> Qwen3; no "3.8" version is known, and
   27B is a Gemma 3 size class rather than a Qwen size. The requested name may
   be a conflation. Building against an unverified model name would be
   building against nothing.
2. **ADR-093 forbids this work right now.** The ruling H2 2026 strategy is
   "Trust, then Traction": one expansion front at a time, and Phase 1 lists
   "new model classes" as an explicit non-goal (parked). Phase 1 is roughly 90
   percent complete and its remaining critical path -- the verified-model
   matrix, T136.6 -> T136.3 -> T136.4 -- contends for the same single GB10 that
   Qwen 3 verification needs.

The seat's recommendation was a cheap 1-2 hour sizing spike, not a build. The
founder was given the ADR-093 conflict explicitly as the reason to decline and
chose to build anyway.

## Decision

Add Qwen 3 support now, as epic E151, scoped in three specific ways:

1. **Support the architecture, not a parameter size.** Implement whatever
   architecture string current Qwen3 GGUFs actually declare in their metadata.
   Architecture support covers every size in the family, which makes the
   unverified "3.8 27B" naming moot rather than blocking.
2. **Resolve reality before writing code.** The first deliverable is a report
   of which Qwen3 models actually exist and whether anything matching the
   requested name is real. Findings are not to be bent to fit the request, and
   no model is to be invented to satisfy it.
3. **Verify the architecture delta against real GGUF tensor names, not model
   recollection.** The reported Qwen3 delta from Qwen2 (per-head QK RMSNorm,
   dropped QKV attention bias) is treated as a hypothesis to check, not a
   specification. Correctness is proven on the smallest Qwen3 (minutes) before
   any large download is attempted; a 27B-class Q4_K_M is roughly 16GB on a
   demonstrably slow link.

The gemma4e precedent governs the honesty bar: if greedy decode is degenerate
on a real GGUF, it is reported as degenerate and the architecture is marked
experimental, not quietly shipped.

## Consequences

**Positive.** The capability gap closes, and closing it at the architecture
level is cheap relative to its reach -- the `qwen2` builder already exists and
the delta is believed small. Scoping to the architecture string also removes
the dependency on resolving an unverifiable model name.

**Negative, and accepted knowingly.** This violates ADR-093's one-expansion-
front rule and Phase 1's stated non-goal. It introduces GB10 contention with
the verified-model matrix runs, which are the phase's core deliverable and
already twice-delayed. If E151 begins delaying milestone M-P1-4, the seat is
to surface the trade-off explicitly rather than let Phase 1 exit slip
silently.

**Precedent risk.** ADR-093's value comes from being binding; overriding it by
request weakens it for the next ask. This ADR exists so the override is
recorded as a deliberate, reasoned exception with a named cost, rather than
becoming an undocumented habit.

**Trust-surface obligation.** Per ADR-093 rule 1, Qwen 3 does not become a
public claim on merge. It enters `docs/verified-models.md` only with parity
and benchmark evidence attached, like every other row.

## Provenance

Founder-direct (David, 2026-08-20), overriding the seat's recommendation of a
sizing spike. Reasoning and rejected alternatives recorded in the hq decision
ledger at `.dira/entries/dec-0050.md`.
