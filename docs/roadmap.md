# Roadmap / Status Board

Cross-session status entries for the org protocol (`~/.claude/CLAUDE.md`,
"THE ORG PROTOCOL"). Detailed task tracking lives in `docs/plan.md`;
detailed investigation history lives in `docs/devlog.md`. This file is
the lightweight, sweepable status surface for the seat.

---

## 2026-08-11 -- STATUS

- **Session:** zerfoo Phase 1 closeout (security review + DGX model provisioning + gemma4e disposition), spanning 2026-08-09 through 2026-08-11.
- **Repo + lane:** `zerfoo/zerfoo`, `main`. Phase 1 ("Trust"), Objective 6 (deep-review 002 security closeout) and the T136.2/T134.x (model provisioning / gemma4e) chain.
- **Done and verified (all merged to `main`):**
  - Security review closeout (T145.2): Objective 6 / D7 closed. 9 High findings re-verified, remediation status documented.
  - Two pre-existing CI blockers found and fixed: a GGUF-parser stack-overflow bug (PR #978) and a grpc CVE (PR #979); T145.1 CLI-flag work merged (#977).
  - DGX GGUF model provisioning (T136.2): 9 of 11 flagship models staged (~32GB in `/var/lib/zerfoo/models`); 2 largest deliberately deferred (multi-hour downloads).
  - gemma4e decode-quality disposition (T134.1/T134.2): fix candidate implemented, GPU-verified, refuted; architecture demoted to experimental (PR #980, #985).
  - Two real GPU bugs found and fixed: a test-fixture bug, and a SIGSEGV crash in an upstream `ztensor` kernel affecting any GQA model (Llama/Mistral/Qwen/Gemma) -- mitigated here, filed upstream (`zerfoo/ztensor#180`).
  - Three more real bugs found, triaged as pre-existing, filed rather than fixed: `#981`, `#982`, `#983`.
- **In-flight:** none. Session wound down via `/handover`; full notes at `docs/handover.md` on the `handover` branch (origin, commit `231ec5dd`).
- **Planned (unblocked, unclaimed):** T136.3 (parity runs) and T136.4 (benchmark re-runs) against the 9 newly-staged models -- both ready to pick up, see `docs/handover.md`.
- **Uncommitted/unpushed:** none. `git status` clean on `main`; all work pushed.
- **Blockers:** none currently open.
- **Founder questions currently held:** none pending -- all prior questions this session were answered directly.
- **Fleet bookkeeping note:** no `~/.claude/bus/zerfoo/coordination.md` team bus exists yet (only an unrelated `sirerun` one found) and no Engineering Portfolio Notion row was mirrored -- both steps in the handover skill's fleet-bookkeeping phase were skipped for lack of a real target rather than fabricated.
