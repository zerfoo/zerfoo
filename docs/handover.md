# Handover -- 2026-08-11T14:xx UTC, session zerfoo-phase1-closeout

## TL;DR
Picked up a stalled T145.1 merge, found and fixed two things blocking `main`'s
CI (a security bug + a CVE), closed the security-review objective, then did
DGX GGUF model provisioning + the gemma4e correctness disposition, finding
and fixing two more real GPU bugs along the way. Session ended clean: no
in-flight work, everything merged, `main` green.

## Done & VERIFIED
- T145.1 (rate-limit/keystore CLI flags) merged: PR #977.
- F4 GGUF-parser stack-overflow bug found + fixed: PR #978. Verified via
  `TestParse_ArrayNestingDepthExceeded`/`...AtMax`, 3x clean fuzz runs.
- GO-2026-6061 grpc CVE fixed (v1.79.3 -> v1.82.1): PR #979. Verified via
  `go test ./distributed/...` + govulncheck no longer reporting it.
- T145.2 (deep-review 002 closeout): Objective 6 / D7 closed. All 9 High
  findings + CICD-1/2 individually re-run and passing on `main`. Remediation
  status table in `docs/deep-reviews/002-full-codebase.md`.
- T136.2 (DGX GGUF model provisioning): 9/11 flagship models staged in
  `/var/lib/zerfoo/models` (~32GB), size-verified against each file's
  `Content-Length` (not just curl exit code -- see Landmines). `ls -la
  /var/lib/zerfoo/models/*.gguf` on the DGX confirms.
- T134.1/T134.2 (gemma4e disposition): H21 fix candidate implemented, run
  twice in isolation against the real staged model, deterministic degenerate
  output both times -- refuted. `gemma4`/`gemma4e`/`gemma4moe` now log
  `slog.Warn("loading experimental architecture", ...)` at load
  (`inference/load_gguf.go`), verified firing. `zerfoo#766` closed as
  attempted, `zerfoo#757` re-scoped to `parked`. PRs #980, #985.
- Two real GPU bugs found + fixed (PR #980): `TestGPUParity_Conv1D` test-shape
  bug (now `maxDiff=0.0`), and a SIGSEGV in `TestGPUParity_GQA` from
  `ztensor`'s fused `RepeatInterleaveF32` kernel (affects any GQA model --
  mitigated by disabling the fused path, upstream issue `zerfoo/ztensor#180`
  filed with full repro).
- Full `go build/vet/test -short ./...` green on `main` at `b6f53b3b`
  (excluding nothing now -- the earlier wrong "no GPU" exclusion was
  corrected and re-verified this session).

## Done but UNVERIFIED
- None known. Everything claimed above was directly observed (test output,
  file sizes, CI check results) this session, not inferred.

## In flight
None. No claimed lanes, no open PRs, no uncommitted work beyond this
handover branch itself.

## Blocked
None currently open.

## Not done, intentionally deferred (pick these up next)
- **T136.3/T136.4** (parity + benchmark runs against the 9 newly-staged
  models): genuinely unblocked now, not yet attempted. Natural next step.
- **T136.5** (surface the matrix) and **T138.1** (plan Phase 2): downstream
  of T136.3/T136.4.
- **Llama 4 Scout (~65GB) and MiniMax-M2 (~129GB)**: not fetched -- would
  take multiple hours at this network's demonstrated throughput. Use the
  `curl -4 --http1.1 --fail -C -` pattern from `docs/lore.md` L-0016 if
  picking these up; don't trust plain `curl -sL` or `zerfoo pull` here.
- **Three real bugs found, triaged as pre-existing (not caused by this
  session's changes, confirmed via revert-and-rerun), filed but not fixed:**
  `zerfoo#981` (KV-cache multi-head key offset bug), `zerfoo#982`
  (intermittent `cudaMemcpy invalid argument` in Gather), `zerfoo#983`
  (PatchTST tiny-training GPU convergence failure). Each issue has a full
  repro.
- **`ztensor/ztensor#180`**: the actual upstream root cause of the GQA crash
  mitigated in this repo. Needs a real fix + release + zerfoo bump to
  re-enable the fused RepeatInterleave perf path.

## Running processes left alive
None. All background model-download tasks completed and were resolved
(verified or retried) before this handover.

## Landmines & context
- **This session's sandbox IS the DGX host** (`aitopatom-bfc8`, real GB10
  GPU) -- not a remote environment. Don't assume you need SSH or that
  GPU-touching tests are unavailable; check `nvidia-smi`/`hostname` directly
  before concluding otherwise. Two earlier-session claims of exactly that
  wrong shape ("SSH is broken," "no GPU here") had to be corrected this
  session -- see `docs/devlog.md` 2026-08-10 entry for the full account.
- **`docs/lore.md` L-0016**: huggingface.co downloads from this host
  silently truncate over plain `curl -sL` (exit 0, wrong size) unless you
  add `--fail`; also force `-4` (broken IPv6 path) and `--http1.1` (HTTP/2
  stream resets). Always verify final size against `Content-Length`.
- **`docs/lore.md` L-0017**: `nohup ... & disown` background processes on
  this host got silently reaped between agent tool-call boundaries (not
  OOM -- confirmed via `dmesg`/`journalctl`). Use the harness's own
  background-task tracking for anything that must survive across turns.
- **`docs/lore.md` L-0013**: this box has a known CUDA-context-poisoning
  hazard class -- don't run two GPU-touching processes concurrently
  (a Spark validation pod + a local decode test collided once this session
  and produced an untrustworthy result that had to be discarded and re-run
  in isolation).
- **Local Go toolchain quirk**: `GOROOT` env defaults to a stale `1.26.0`
  install while `go` on PATH resolves to `1.26.1`; export
  `GOROOT=/home/ndungu/go-sdk/go` (or wherever the newer install lives) or
  every build fails with a toolchain-version-mismatch error. Not a repo
  issue.
- **THE ORG PROTOCOL** (`~/.claude/CLAUDE.md`) now governs sessions on
  David's machines -- confirmed legitimate by David directly this session
  after I flagged it as an unverified-looking instruction (the referenced
  `~/Code/dndungu/hq` "seat" repo doesn't exist on this DGX host; the
  protocol itself says that's expected since the seat runs on the mini).
  `docs/roadmap.md` (new this session) is this repo's status-board fallback
  for machines where `SendMessage to seat` isn't available.

## How to resume
1. `git fetch origin` from the primary checkout (`/home/ndungu/Work/zerfoo/zerfoo`, already on `main`, already up to date as of this handover).
2. Read this file, then `docs/roadmap.md` (status board) and `docs/plan.md`'s hand-off section (SESSION HANDOFF 2026-08-10) for full task-level context.
3. Read `docs/devlog.md`'s 2026-08-09 and 2026-08-10 entries for the full investigation narrative behind everything above.
4. Next concrete lane: T136.3 (parity runs against the 9 staged models) -- no claim system is in active use on this repo currently (checked: `refs/claims/*` empty on origin), so just pick it up per `docs/plan.md`.
5. This `handover` branch is a marker only -- it points at the same commit as `main` (`b6f53b3b` at push time plus this notes commit); there is no unmerged WIP to reconcile. Safe to ignore once read.
