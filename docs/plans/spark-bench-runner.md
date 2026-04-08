# Spark Bench Runner Deployment

## Context

**Problem.** Today (2026-04-07) we discovered two regressions on `main` while running PatchTST GPU training benchmarks on the DGX Spark host (192.168.86.250):

1. A convergence regression introduced by PR #348 (E85 GPU memory leak fix) where the AdamW step writes to a throwaway CPU copy of GPU tensor memory. Fix in flight on branch `fix/gpu-train-cpu-mirror` (commit `0750c440`).
2. A separate hang in 25K x 20ch x 10-epoch GPU training runs that surfaces only on longer runs and produces no output. Still under investigation.

**Bigger problem revealed by the incident.** Every benchmark we ran today went via interactive `ssh ndungu@dgx 'bench_train ...'`, which holds the SSH channel open for the duration of the job. This terminal's bash tool auto-backgrounds long SSH commands but does not release the channels, so connections piled up: 60+ user sessions, load avg 13+, sshd starved, host eventually unreachable, **DGX had to be rebooted**. The bench process itself was clean (resource-wise) — the SSH session leak from the client side was the killer.

**Decision.** Use Spark for all benches going forward. Spark is the in-house single-node pod orchestrator at `../../feza-ai/spark` (separate repo, currently at `v1.6.0`) that accepts Kubernetes manifests and runs them via Podman. Submitting a job to Spark over its HTTP API does not hold any persistent SSH connection — the client posts a manifest, polls a job ID, and pulls logs on demand. Combined with Podman cgroups (`memory.max`, `cpu.max`, `pids.max`), this eliminates both failure modes that bit us today: SSH session accumulation and unbounded resource usage.

### Objectives
- Spark `v1.6.0` (or latest stable) installed and running as a systemd service on DGX 192.168.86.250.
- A reusable bench manifest template (`docs/bench/manifests/patchtst-train.yaml`) that runs `bench_train` inside a Podman container with cgroup limits.
- A small helper script `scripts/bench-spark.sh` that submits the manifest, polls until the pod completes, and prints logs — replacing today's `ssh ... bench_train ...` pattern.
- Documented procedure in CLAUDE.md so future sessions know to use Spark instead of interactive SSH.

### Non-Goals
- NOT building a NATS-based bench client (HTTP is sufficient).
- NOT generalizing to all DGX workloads — only zerfoo benches.
- NOT modifying the Spark codebase. We consume the released `.deb` package as-is.
- NOT addressing the hang in `trainWindowedGPU` 10-epoch runs — that is tracked separately and is the next benchmark we'll run *through* the new Spark infrastructure, not a prerequisite for it.

### Constraints / Assumptions
- DGX is `aarch64` Ubuntu, so we need the `linux_arm64.deb` asset from the Spark release.
- Spark `v1.6.0` ships these features we depend on: HTTP REST API, GPU device assignment, init containers, container port mapping, security context, structured JSON logging, pod logs/events via HTTP.
- Podman is already present on DGX (it runs CUDA workloads). Confirmed by `setup-dgx.sh` which assumes podman.
- The bench container needs the CUDA runtime libraries available either via volume mount or via a base image with CUDA libraries pre-installed. We will mount `/usr/local/cuda` from the host into the container as a read-only volume.
- The bench binary itself is built on the host (not inside the container). We will mount `/tmp/bench_train` (or a stable path under `/var/lib/zerfoo/bin/`) into the container.

### Success Metrics
- One-shot 25K x 20ch x 10-epoch bench can be submitted, executed, and its logs retrieved end-to-end without holding any interactive SSH session for more than 2 seconds.
- A runaway bench process is killable via `curl -X DELETE http://dgx:8080/api/v1/pods/<id>` without affecting other host processes.
- Memory cap of 32 GB on the bench container is enforced (verified by manually exceeding it and observing OOM at the cgroup boundary, not the host).
- No SSH session accumulation: after running 10 consecutive benches the host's user-session count is unchanged from before the first run.

## Discovery Summary

**Work type:** mixed engineering + operations.

**Engineering signal:** Authoring a manifest and a helper shell script in this repo, plus a CLAUDE.md update. No new use cases — this is infrastructure that supports existing benchmark use cases (UC-TS01 PatchTST training).

**Operations signal:** Installing a systemd service on a remote host, defining a process for how benchmarks are submitted going forward, retiring the interactive-SSH pattern.

### Spark Release Inventory (verified 2026-04-07)
- Repo: `github.com/feza-ai/spark`, local checkout at `/Users/dndungu/Code/feza-ai/spark`.
- Latest tag: `v1.6.0` (released 2026-03-21).
- Release assets include `spark_1.6.0_linux_arm64.deb` (4.7 MB) — exactly what we need for DGX (aarch64).
- Existing deploy artifacts in the spark repo: `deploy/setup-dgx.sh` (binary install), `deploy/install.sh` (cross-build + scp), `deploy/spark.service`, `deploy/spark.env`, `deploy/nats-server.service`. These were written for the v1.0 era (raw binary install). The `.deb` package is newer and includes its own postinstall hook (`deploy/nfpm/postinstall.sh`) that handles user/group/dirs/systemd reload.
- nfpm postinstall handles: user/group creation, `/etc/spark/`, `/var/lib/spark/`, `/etc/spark/manifests/`, systemd daemon-reload.
- Spark requires NATS for messaging. Existing `nats-server.service` unit in `spark` repo handles installation. NATS is a one-time bootstrap.

### DGX Current State (verified 2026-04-07 19:41 UTC after reboot)
- `uptime`: clean reboot, load avg 0.07, 1 user.
- GPU: NVIDIA GB10 (unified memory architecture, no MIG).
- Repo: `/home/ndungu/zerfoo` checked out, on branch `fix/gpu-train-cpu-mirror`.
- Go toolchain: `/usr/local/go/bin/go`.
- Existing bench binary: `/tmp/bench_train` (built today from `cmd/bench_train`).
- Spark not installed. NATS not installed.
- Podman: not yet verified. Likely needs `apt install podman` if absent.

### Why Spark vs alternatives
| Option | SSH session leak? | Resource limits? | GPU support? | Ops cost |
|---|---|---|---|---|
| `ssh dgx 'bench ...'` (current) | YES — root cause of today's outage | None | Direct | Zero — just `ssh` |
| `ssh dgx 'nohup bench &'` + poll log file | No (channel closes <1s) | None | Direct | One-line wrapper |
| systemd-run --user --scope | Same as ssh, plus cgroup limits | Yes (cgroup v2) | Direct | Slightly higher |
| **Spark + Podman manifest** (chosen) | No (HTTP submit) | Yes (Podman cgroups) | Yes (NVIDIA_VISIBLE_DEVICES) | Higher: install + manifest authoring, but reusable |
| Kubernetes (k3s/microk8s) | No | Yes | Yes | Highest: full k8s overhead on a single node |

Spark wins because it gives us **both** the no-SSH-leak property AND cgroup-enforced resource caps in one package, with the bonus of speaking k8s manifests so the bench definition is portable. The `nohup` workaround would have prevented today's incident but offers no resource caps, no central queue, no kill API, no pod history.

### Reference: docs/adr/017-dgx-spark-hardware-validation.md
ADR 017 covers DGX hardware validation strategy and is the closest existing decision record. This plan adds a new ADR (next sequential number) covering the Spark-as-bench-runner choice specifically.

## Scope and Deliverables

### In Scope
- Install Spark `v1.6.0` (or newest stable at execution time) and NATS on DGX 192.168.86.250 via the `.deb` package.
- Create `docs/bench/manifests/patchtst-train.yaml` — a Pod manifest that runs `bench_train` with configurable args, GPU device assignment, memory/CPU caps, and host volume mounts for the binary and CUDA libs.
- Create `scripts/bench-spark.sh` — submits the manifest via Spark HTTP API, polls until the pod terminates, fetches logs, prints them, exits with the pod's status code.
- Update `CLAUDE.md` (root and zerfoo subrepo) to document the new bench submission procedure and to deprecate interactive `ssh ... bench_train ...`.
- Write a new ADR documenting the decision and the architecture (Spark on DGX, manifest convention, helper script).
- Sanity-check the new pipeline by running the same 1K / 5K / 25K bench progression we ran today, this time through Spark, and confirm:
  - The convergence regression manifests on `main` (proves the path works end-to-end and reaches GPU code).
  - The fix branch `fix/gpu-train-cpu-mirror` produces non-frozen losses (validates the fix).

### Out of Scope
- Investigating or fixing the 10-epoch hang.
- Any zerfoo source changes other than the helper script + CLAUDE.md update.
- Multi-node Spark or NATS clustering.
- Auth tokens / RBAC on the Spark HTTP API (single-node trusted LAN).
- Migrating other zerfoo benchmarks (`bench_batch`, `bench_tps`, etc.) — they can adopt the same manifest pattern later but are not blocking.

### Deliverables Table
| ID | Description | Owner | Acceptance Criteria |
|---|---|---|---|
| D1 | Spark v1.6.0 service running on DGX | TBD | `curl http://192.168.86.250:8080/healthz` returns 200 OK; `systemctl status spark` reports active. NATS service active. |
| D2 | PatchTST bench manifest committed | TBD | `docs/bench/manifests/patchtst-train.yaml` exists, contains GPU spec + cgroup limits + host mounts; passes `kubeval` (or equivalent yaml lint). |
| D3 | Bench submission helper script committed | TBD | `scripts/bench-spark.sh -- -samples 5000 -channels 10 -epochs 3` runs to completion and prints loss history. |
| D4 | CLAUDE.md procedure documented | TBD | Both `CLAUDE.md` files document the new procedure; `ssh ... bench_train` pattern is marked deprecated. |
| D5 | ADR documenting the decision | TBD | New ADR file in `docs/adr/` referenced from this plan. |
| D6 | End-to-end smoke run via Spark | TBD | 5K x 10ch x 3-epoch run on `fix/gpu-train-cpu-mirror` produces a decreasing loss in the pod logs, retrieved via the helper script. |

## Checkable Work Breakdown

### Epic E1: Spark host install on DGX

- [ ] T1.1 Verify podman is installed on DGX  Owner: TBD  Est: 0.25h  delivers: [DGX podman version recorded in plan progress log]
  Acceptance: `ssh dgx 'podman --version'` returns a version. If not present, `sudo apt install -y podman` and rerun.

- [ ] T1.2 Install NATS server v2 (latest) on DGX  Owner: TBD  Est: 0.5h  delivers: [nats-server.service active on DGX]
  Deps: T1.1
  Acceptance: `systemctl status nats-server` shows active. `ss -tln | grep 4222` shows listener.
  Note: reuse `deploy/nats-server.service` from spark repo. Install nats binary via `curl -sf https://binaries.nats.dev/nats-io/nats-server/v2@latest | sh`.

- [ ] T1.3 Download Spark v1.6.0 arm64 .deb from GitHub release  Owner: TBD  Est: 0.25h  delivers: [spark_1.6.0_linux_arm64.deb on DGX /tmp]
  Deps: none
  Acceptance: `sha256sum /tmp/spark_1.6.0_linux_arm64.deb` matches the digest from the release manifest (`bd1aa380abe19b0ae4df8c5c492694f2553e69a1241df254834e2b87e8834dae`).
  Command: `gh -R feza-ai/spark release download v1.6.0 -p 'spark_1.6.0_linux_arm64.deb' -D /tmp`, then `scp /tmp/spark_1.6.0_linux_arm64.deb ndungu@dgx:/tmp/`.

- [ ] T1.4 Install Spark .deb on DGX  Owner: TBD  Est: 0.25h  delivers: [/usr/local/bin/spark installed via dpkg]
  Deps: T1.2, T1.3
  Acceptance: `dpkg -l spark` shows v1.6.0. nfpm postinstall hook will create the user/group, dirs, and reload systemd.
  Command: `ssh dgx 'sudo dpkg -i /tmp/spark_1.6.0_linux_arm64.deb'`.

- [ ] T1.5 Configure Spark env for DGX (system reserve, GPU max)  Owner: TBD  Est: 0.5h  delivers: [/etc/spark/spark.env tuned for GB10]
  Deps: T1.4
  Acceptance: `/etc/spark/spark.env` contains `SPARK_GPU_MAX=1`, `SPARK_SYSTEM_RESERVE_CPU=4000` (4 cores reserved for sshd/system), `SPARK_SYSTEM_RESERVE_MEMORY=8192` (8 GB reserved). HTTP bind on `:8080`. Source: copy and adapt `deploy/spark.env` from the spark repo.

- [ ] T1.6 Enable and start spark.service  Owner: TBD  Est: 0.25h  delivers: [spark.service active on DGX]
  Deps: T1.5
  Acceptance: `systemctl status spark` active; `curl http://localhost:8080/healthz` returns 200; `curl http://localhost:8080/api/v1/node-info` reports the GB10 GPU.

- [ ] T1.7 Verify Spark API is reachable from this Mac  Owner: TBD  Est: 0.25h  delivers: [curl http://192.168.86.250:8080/healthz from local works]
  Deps: T1.6
  Acceptance: `curl -sf http://192.168.86.250:8080/healthz` from the Mac returns 200. If firewall blocks, open port 8080 inbound on DGX.

### Epic E2: Bench manifest and helper

- [ ] T2.1 Author `docs/bench/manifests/patchtst-train.yaml`  Owner: TBD  Est: 1h  delivers: [committed Pod manifest with GPU + cgroup limits]
  Deps: T1.6
  Acceptance: Manifest is a v1 Pod with:
    - `spec.containers[0].image`: a CUDA-capable base image (initially `docker.io/library/ubuntu:24.04` since we mount the host bench binary; can switch to a CUDA base image later if we need libs not on host).
    - `spec.containers[0].command`: `["/var/lib/zerfoo/bin/bench_train"]`.
    - `spec.containers[0].args`: parameterizable via env substitution at submission time (we substitute samples/channels/epochs/out before posting the manifest).
    - `spec.containers[0].resources.limits`: `memory: 32Gi`, `cpu: "8"`, `nvidia.com/gpu: 1`.
    - `spec.volumes`: hostPath mounts for `/var/lib/zerfoo/bin` (read-only), `/usr/local/cuda` (read-only), `/var/lib/zerfoo/bench-out` (read-write) to capture the `-out` file.
    - `spec.containers[0].env`: `LD_LIBRARY_PATH=/usr/local/cuda/lib64`.

- [ ] T2.2 Author `scripts/bench-spark.sh`  Owner: TBD  Est: 1.5h  delivers: [shell helper that submits + polls + fetches logs]
  Deps: T2.1, T1.7
  Acceptance: Plain bash + curl, no external deps. Behavior:
    1. Reads the manifest template, substitutes args from `$@`.
    2. POSTs to `http://192.168.86.250:8080/api/v1/pods` with the manifest.
    3. Captures the returned pod name/UID.
    4. Polls `/api/v1/pods/{name}` every 3 seconds until phase is `Succeeded` or `Failed`.
    5. Fetches `/api/v1/pods/{name}/logs` and prints to stdout.
    6. Exits 0 on Succeeded, non-zero on Failed.
    7. Supports a `-cleanup` flag that deletes the pod after fetching logs (default: keep for inspection).
    Make script idempotent and POSIX-bash compatible.

- [ ] T2.3 Add unit-style smoke test for the helper script  Owner: TBD  Est: 0.5h  delivers: [tests/bench-spark/smoke.sh that runs the helper against a 1K bench]
  Deps: T2.2
  Acceptance: A trivial test that runs `scripts/bench-spark.sh -samples 1000 -channels 5 -epochs 2` and asserts the output contains `total:` and 2 epoch lines. Used as the post-deploy validation step.

- [ ] T2.4 Lint the helper script and manifest  Owner: TBD  Est: 0.25h  verifies: [infrastructure]
  Deps: T2.2
  Acceptance: `shellcheck scripts/bench-spark.sh` clean. `kubeval docs/bench/manifests/patchtst-train.yaml` (or equivalent) clean.

### Epic E3: Documentation and decision record

- [ ] T3.1 Create ADR for Spark bench runner  Owner: TBD  Est: 0.5h  delivers: [docs/adr/NNN-spark-bench-runner.md]
  Deps: T2.2
  Acceptance: New ADR file using the next sequential number (currently 083 — next = 083, since highest existing is 082-composition-remediation-strategy.md). Captures: context (today's incident), decision (Spark + Podman manifests), alternatives evaluated (table from this plan), consequences. Self-contained and understandable without reading this plan.

- [ ] T3.2 Update root CLAUDE.md with new bench procedure  Owner: TBD  Est: 0.25h  delivers: [/Users/dndungu/Code/zerfoo/CLAUDE.md updated]
  Deps: T3.1
  Acceptance: A new "Benchmarks" subsection or update to "Hardware" pointing to `scripts/bench-spark.sh` and the manifest path. Marks `ssh ... bench_train` as deprecated. Notes the resource limits enforced by the manifest.

- [ ] T3.3 Update zerfoo subrepo CLAUDE.md  Owner: TBD  Est: 0.25h  delivers: [/Users/dndungu/Code/zerfoo/zerfoo/CLAUDE.md updated]
  Deps: T3.1
  Acceptance: Same content as T3.2 but scoped to the zerfoo project. References the helper script and manifest paths relative to repo root.

### Epic E4: Validation through Spark

- [ ] T4.1 Smoke run on `main` via Spark  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Deps: T1.7, T2.2, T2.3
  Acceptance: `scripts/bench-spark.sh -samples 5000 -channels 10 -epochs 3` (against `main`) returns frozen-loss output exactly matching today's broken behavior (`0.268357` repeated). This proves the new infrastructure correctly reproduces a known regression and that the GPU code path actually runs through Spark.

- [ ] T4.2 Validate fix branch via Spark  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Deps: T4.1
  Acceptance: After checking out `fix/gpu-train-cpu-mirror` on DGX and rebuilding `bench_train`, the same 5K x 10ch x 3ep run produces a strictly decreasing loss (matches the local CPU bench's 99.2% reduction within run-to-run variance). This validates that the convergence fix is correct on real GPU.

- [ ] T4.3 Stress test cgroup memory cap  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T1.7, T2.2
  Acceptance: Submit a deliberately oversized config (e.g. 200K x 50 channels) and confirm the pod is killed by the cgroup OOM at 32 GB without affecting other host processes (sshd remains responsive, no reboot needed).

- [ ] T4.4 SSH session leak verification  Owner: TBD  Est: 0.25h  verifies: [infrastructure]
  Deps: T4.1, T4.2
  Acceptance: Run `ssh dgx 'who | wc -l'` before, during, and after a series of 10 bench submissions. Count must not increase across runs. This is the original incident's regression test.

- [ ] T4.5 Document the validation results in docs/devlog.md  Owner: TBD  Est: 0.25h  delivers: [devlog entry with bench-via-spark commissioning notes]
  Deps: T4.1, T4.2, T4.3, T4.4
  Acceptance: New /journal-formatted entry under "## 2026-04-07: Spark commissioned as bench runner" capturing the convergence test result, the cgroup OOM test, and the SSH session count measurements.

## Parallel Work

### Tracks

| Track | Tasks | Description |
|---|---|---|
| A: Host install | T1.1 → T1.7 | Sequential install chain on DGX. |
| B: Manifest + helper | T2.1, T2.2, T2.3, T2.4 | Local authoring, can begin once T1.6 returns a healthy spark; T2.1 can be drafted in parallel with E1 since the manifest schema is fixed by Spark v1.6.0 docs. |
| C: Docs + ADR | T3.1, T3.2, T3.3 | Local doc work, can run in parallel with E2 once E2 has shape. |
| D: Validation | T4.1 → T4.5 | Sequential, depends on E1 + E2 complete. |

### Sync Points
- T1.6 must complete before T2.3 (helper smoke test needs a live Spark API).
- T2.2 must complete before T3.1 (ADR references the helper script's behavior).
- T2.3 must complete before T4.1 (validation uses the helper).

### Waves

#### Wave 1: Bootstrap (3 agents)
- [ ] T1.1 Verify podman on DGX  delivers: [podman version]
- [ ] T1.3 Download Spark .deb (no host dep)  delivers: [.deb on Mac then on DGX /tmp]
- [ ] T2.1 Draft manifest  delivers: [manifest yaml file in repo]

#### Wave 2: Install + helper (3 agents)
- [ ] T1.2 Install NATS  (depends on T1.1)
- [ ] T1.4 dpkg -i Spark  (depends on T1.2, T1.3)
- [ ] T2.2 Author helper script  (can start since manifest exists)

#### Wave 3: Configure + smoke (4 agents)
- [ ] T1.5 Configure spark.env
- [ ] T1.6 Start spark.service
- [ ] T2.4 Lint helper + manifest
- [ ] T3.1 Draft ADR (can start in parallel with config)

#### Wave 4: Verify reachability + tests (3 agents)
- [ ] T1.7 Verify HTTP from Mac
- [ ] T2.3 Helper smoke test (depends on T1.7 + T2.2)
- [ ] T3.2 Root CLAUDE.md update
- [ ] T3.3 Zerfoo CLAUDE.md update (4 tasks; well under 10-agent cap)

#### Wave 5: Validation (4 agents)
- [ ] T4.1 Smoke main on Spark
- [ ] T4.2 Validate fix branch on Spark
- [ ] T4.3 Stress cgroup OOM
- [ ] T4.4 SSH session leak verification

#### Wave 6: Devlog (1 agent)
- [ ] T4.5 Document validation results

### Dependency-Minimization Notes
- T1.3 (download .deb) deliberately has no host dep — it can run on the Mac in parallel with T1.1/T1.2.
- T2.1 (manifest authoring) deliberately has no Spark-running dep — schema is from docs, not introspection.
- T3.1 (ADR drafting) is parallelized with E2 because the decision is already made (this plan IS the decision); the ADR is just the formal write-up.

## Timeline and Milestones

| ID | Milestone | Dependencies | Exit Criteria |
|---|---|---|---|
| M1 | DGX has Spark installed and reachable | E1 complete | `curl http://192.168.86.250:8080/healthz` returns 200 from the Mac. |
| M2 | Bench manifest + helper committed and linted | E2 complete | `shellcheck` clean, `kubeval` clean, helper smoke test passes against a 1K bench. |
| M3 | Documentation and ADR landed | E3 complete | New ADR file in `docs/adr/`; both CLAUDE.md files updated. |
| M4 | Validation complete; bench wave can resume | E4 complete | Convergence regression reproduced via Spark on `main`, fixed on the fix branch, no SSH sessions leaked, cgroup OOM verified. |
| M5 | Plan archived | M4 complete | This plan moved to "Completed" or removed; the bench wave from `docs/plan.md` (T50.5.2, T51.5.2, T54.4.1, T63.2.1, T61.3.2) can now run via the new pipeline. |

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Podman not installed on DGX (or wrong version) | Blocks E1 | Med | T1.1 verifies first; if missing, install via apt before T1.2. |
| R2 | NVIDIA container toolkit not installed → Spark cannot pass GPU into Podman | Blocks GPU benches | Med | Verify with `podman run --device nvidia.com/gpu=all docker.io/nvidia/cuda:12-base nvidia-smi` after T1.1. If missing, install `nvidia-container-toolkit` via apt and run `nvidia-ctk runtime configure --runtime=podman`. |
| R3 | CUDA libs on host not compatible with Spark v1.6.0's GPU plumbing | Bench segfaults inside the container | Low | T4.1 explicitly tests by running an actual bench. If it fails, fall back to mounting the libs from a container base image (`docker.io/nvidia/cuda:12-runtime`) instead of host volume mount. |
| R4 | Port 8080 blocked by firewall on DGX | T1.7 fails | Low | If `curl` from Mac times out, `sudo ufw allow from 192.168.86.0/24 to any port 8080` on DGX. |
| R5 | The bench binary (built on host) has dynamic deps not in the container's base image | Runtime linker errors | Med | Use `ldd /tmp/bench_train` on DGX to enumerate deps. If only standard libs, `ubuntu:24.04` base is enough. If CUDA-runtime is needed at link time (likely), switch to `nvidia/cuda:12-runtime` base. |
| R6 | Spark v1.6.0 has a regression we don't know about | Pipeline broken | Low | If smoke test fails, fall back to the previous stable spark version (`v1.5.x`) by downloading a different .deb. Document in devlog. |
| R7 | This plan delays fixing the actual zerfoo regressions further | Mission risk | Med | Scope is intentionally minimal: install + manifest + script + smoke. No feature work in Spark itself. M4 explicitly resumes the bench wave. |

## Operating Procedure

### Definition of Done (per task)
- Engineering tasks: code committed in small focused commits, lint + tests green, PR opened, CI green, merged via rebase, helper script smoke test passes if applicable.
- Operations tasks (DGX install): the relevant systemd service is `active` and survives a `systemctl restart`. The exact commit hash of the spark binary is recorded in the progress log.
- Documentation tasks: file committed and visible at the documented path; cross-references resolve.

### Review and QA
- One reviewer for the manifest and helper script (security: host volume mounts and HTTP API exposed on LAN).
- Stress test (T4.3) is mandatory before declaring M4 complete. Cgroup enforcement is the central value prop of this plan; if it does not enforce, the plan failed.

### Always Add Tests / Lint
- T2.3 is the unit-style smoke test for the helper.
- T2.4 is the lint task. `shellcheck` for bash, `kubeval` (or `kubectl --dry-run=client`) for yaml.
- No Go code is changed in this plan, so no `go test` / `golangci-lint` tasks beyond what is already in CI.

### Commit Discipline
- Manifest, script, ADR, and CLAUDE.md updates each go in separate commits.
- Use Conventional Commits: `feat(bench): add Spark manifest`, `feat(bench): add bench-spark.sh helper`, `docs(adr): add 083-spark-bench-runner`, `docs(claude): document Spark bench procedure`.
- No commit straddles directories (per project convention).

### Versioning
- This plan does not produce a versioned release of zerfoo. Spark's own versioning is handled in its repo.

## Progress Log

### Change Summary 2026-04-07
- Plan created in response to today's DGX outage caused by SSH session accumulation while running PatchTST GPU benchmarks.
- Decision to use Spark v1.6.0 (latest stable, 2026-03-21) as the bench runner going forward, consuming the prebuilt `linux_arm64.deb` package from `github.com/feza-ai/spark` releases.
- 6 epics, 22 tasks, 6 waves designed to maximize parallelism (Wave 1 has 3 truly independent tasks).
- ADR planned at `docs/adr/083-spark-bench-runner.md` (next sequential number; highest existing is 082).
- Plan stored at `docs/plans/spark-bench-runner.md` (alongside `docs/plans/cgo-removal.md` per convention).
- This plan does NOT trim the existing `docs/plan.md` (1963 lines, 21 epics) — that is out of scope for this focused operational task.
- Linked context for future readers: today's GPU convergence fix is on branch `fix/gpu-train-cpu-mirror` (commit `0750c440`); the 10-epoch hang investigation is unblocked by completing this plan.

## Hand-off Notes

- **What you need to know:** This plan exists because on 2026-04-07 we had to reboot the DGX (192.168.86.250) twice in a single session due to SSH session accumulation from interactive `ssh dgx 'bench_train ...'` calls. The fundamental fix is to stop using interactive SSH for benches. Spark is the in-house tool for that, already at v1.6.0 in `feza-ai/spark` repo, with a working `.deb` package. The work is mostly install + a thin client wrapper.
- **Where the spark code lives:** `/Users/dndungu/Code/feza-ai/spark`. Releases: `gh -R feza-ai/spark release list`.
- **Where the existing deploy artifacts live (in spark repo):** `deploy/setup-dgx.sh` (binary install — older path), `deploy/install.sh` (cross-build helper), `deploy/spark.service`, `deploy/spark.env`, `deploy/nfpm/postinstall.sh` (used by the .deb path).
- **DGX access:** `ssh ndungu@192.168.86.250`. Repo at `~/zerfoo`. Go at `/usr/local/go/bin/go`.
- **DO NOT** run more than one or two benches via interactive SSH from the Claude Code bash tool — it auto-backgrounds long SSH commands and leaks channels. Even one bench can put the host in a degraded state if the channel never closes. This is exactly what this plan exists to fix.
- **CUDA on DGX:** GB10, unified memory, no MIG, so VRAM cannot be cgroup-capped. Memory caps in this plan are RAM caps (Podman `memory.max`). VRAM contention is handled by `SPARK_GPU_MAX=1` (one GPU pod at a time on the host).
- **When you are done with this plan:** the bench wave from `docs/plan.md` (T50.5.2, T51.5.2, T54.4.1, T63.2.1, T61.3.2) is unblocked and should run via `scripts/bench-spark.sh`. Update `docs/plan.md` to point to the new procedure for those tasks.

## Appendix

### Useful Spark API endpoints (v1.6.0)
- `GET  /healthz` — health check
- `GET  /api/v1/node-info` — GPU model, device count, CPU, memory
- `POST /api/v1/pods` — submit a pod manifest
- `GET  /api/v1/pods` — list pods
- `GET  /api/v1/pods/{name}` — get pod status
- `GET  /api/v1/pods/{name}/logs` — fetch logs (supports SSE streaming)
- `GET  /api/v1/pods/{name}/events` — lifecycle events
- `DELETE /api/v1/pods/{name}` — kill and remove pod

### Reference release manifest (v1.6.0)
Verified 2026-04-07 from `gh -R feza-ai/spark release view v1.6.0`:
- `spark_1.6.0_linux_arm64.deb` (4715786 bytes) — sha256 `bd1aa380abe19b0ae4df8c5c492694f2553e69a1241df254834e2b87e8834dae`
- `spark_1.6.0_linux_amd64.deb` (5119674 bytes) — sha256 `f300a850a03360deb8fcb3d4da759be0746e311b0065eb963c746dc923326bd9`
- `spark_1.6.0_checksums.txt` for verification.

### Manifest sketch (concept; final form authored in T2.1)
```
apiVersion: v1
kind: Pod
metadata:
  name: bench-patchtst-${TIMESTAMP}
  labels:
    app: bench
    type: patchtst-train
spec:
  restartPolicy: Never
  containers:
    - name: bench
      image: docker.io/library/ubuntu:24.04
      command: ["/var/lib/zerfoo/bin/bench_train"]
      args: ["-samples", "${SAMPLES}", "-channels", "${CHANNELS}", "-epochs", "${EPOCHS}", "-out", "/var/lib/zerfoo/bench-out/${RUN_ID}.txt"]
      env:
        - name: LD_LIBRARY_PATH
          value: /usr/local/cuda/lib64
      resources:
        limits:
          memory: 32Gi
          cpu: "8"
          nvidia.com/gpu: 1
      volumeMounts:
        - { name: zerfoo-bin, mountPath: /var/lib/zerfoo/bin, readOnly: true }
        - { name: cuda, mountPath: /usr/local/cuda, readOnly: true }
        - { name: bench-out, mountPath: /var/lib/zerfoo/bench-out }
  volumes:
    - { name: zerfoo-bin, hostPath: { path: /var/lib/zerfoo/bin, type: Directory } }
    - { name: cuda, hostPath: { path: /usr/local/cuda, type: Directory } }
    - { name: bench-out, hostPath: { path: /var/lib/zerfoo/bench-out, type: DirectoryOrCreate } }
```
