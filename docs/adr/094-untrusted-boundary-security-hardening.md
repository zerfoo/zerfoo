# ADR 094: Untrusted-boundary security hardening

## Status
Accepted

## Date
2026-07-03

## Context

Deep-review 002 (docs/deep-reviews/002-full-codebase.md, 2026-07-03, HEAD 5817d590)
found zerfoo's overall security posture "above average" (maturity ~3.9/5) but
identified 9 High-severity findings clustered in three untrusted boundaries the
framework had not treated as first-class attack surfaces:

1. **GGUF model files** (F1/F2/F3): downloaded from the internet and explicitly
   untrusted by design, yet the loader's overflow/offset guards check *after*
   arithmetic that can already have wrapped, so a crafted file panics the process
   instead of returning an error. The identical bug was copy-pasted across four
   loader sites.
2. **The distributed-training wire** (DIST-1/DIST-2): the worker gRPC server binds
   with no TLS credentials and the shipped example binds `0.0.0.0`; the mTLS
   machinery in `distributed/tlsconfig.go` is correct but has no caller.
3. **Native library loading** (CUDA-1/CUDA-2): `internal/cuda/purego.go` includes
   a CWD-relative `./libkernels.so` dlopen candidate, and vendor CUDA/HIP/OpenCL
   libraries load by bare soname, both hijackable via a writable working
   directory or `LD_LIBRARY_PATH`.

A recurring theme across the review was **security code that exists but is never
wired in**: the rate limiter, scoped keystore, incident responder, and mTLS
config are all correct in `serve/security/` and `distributed/`, but the shipped
CLI connects none of them by default.

This sits squarely inside Phase 1 ("Trust, then Traction", ADR-093): the phase's
exit criterion is zero known silent-correctness bugs, and an untrusted input that
crashes the process or a wire that accepts unauthenticated writes are trust
failures of the same class as the capture/replay correctness bugs already in this
phase's scope.

## Decision

1. **Treat the GGUF file and the distributed-training wire as first-class
   untrusted inputs**, on par with any network-facing HTTP request. Bounds-check
   before arithmetic that can overflow (check-then-multiply, not
   multiply-then-check); validate offsets as unsigned before any signed
   conversion; return errors, never panic, on malformed input. Extract the
   four-times-duplicated GGUF loader loop into one shared, tested helper so the
   fix (and any future fix) exists in exactly one place.
2. **Native library loading uses only vetted absolute paths.** No
   CWD-relative or bare-soname dlopen candidates. A configurable path must be an
   absolute path, and should be validated as non-world-writable before use.
3. **The distributed wire fails closed, not open.** A worker or coordinator may
   bind loopback without TLS (single-host dev use); any routable bind must
   present the already-implemented mTLS credentials or refuse to start.
4. **Ship the defenses you write.** When a security capability
   (rate limiter, keystore, mTLS, incident responder) is implemented and correct,
   it must be reachable from the CLI (a flag, a documented default), not left as
   an unwired library API. An implemented-but-unwired control is treated as an
   open finding, not a mitigated one.
5. Remediation follows the deep-review's own four-tier roadmap (fix immediately /
   this sprint / this quarter / tech debt) rather than being re-triaged from
   scratch; see docs/plan.md epics E139-E145 for the task-level breakdown.

## Consequences

**Positive:**
- Malformed model files (accidental corruption or supply-chain tampering) become
  a clean load error instead of a process crash -- important for any auto-reload
  or auto-pull deployment.
- Multi-host distributed training gets a real authentication/encryption boundary
  instead of an implicit trust-the-LAN assumption.
- Removing CWD/bare-soname dlopen candidates closes a local-privilege-escalation
  primitive with no loss of functionality for standard installs (the trusted
  `/opt/zerfoo/lib` path was already first in the candidate list).
- Consolidating the GGUF overflow-guard loop removes a landmine class: future
  contributors can no longer fix the bug in one of four sites and leave the
  other three vulnerable.

**Negative / costs:**
- Multi-host distributed training now requires certificate provisioning
  (`--tls-*` flags) where it previously worked with zero configuration; this is
  an intentional posture change documented in E140, with a cert-gen helper to
  soften the migration.
- Dev builds that relied on dropping `libkernels.so` next to the binary (CWD)
  must switch to an absolute path or the vetted default location.
- The tier-3/tech-debt findings (SERVE-3 caps, CONC-M1 cleanup ticker, CI/CD
  permissions hardening, SLSA signing) add real but bounded engineering cost;
  per ADR-093 rule 3 discipline, tech-debt-tier items not closed within Phase 1
  are filed as tracked issues rather than open-endedly pursued.
