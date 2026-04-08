# ADR 083: Spark as the bench runner for DGX GPU workloads

- **Status:** Accepted
- **Date:** 2026-04-07
- **Context tags:** infrastructure, benchmarking, dgx, operations

## Context

On 2026-04-07 we ran a sequence of PatchTST GPU training benchmarks on the DGX
Spark host (`192.168.86.250`, NVIDIA GB10, aarch64) to validate a convergence
fix on branch `fix/gpu-train-cpu-mirror`. Every bench was submitted via
interactive SSH from the Claude Code bash tool:

```
ssh ndungu@192.168.86.250 'bench_train -samples 25000 -channels 20 -epochs 10'
```

The bash tool auto-backgrounds long-running commands but does not release the
SSH channel while they run. Over the course of the session, channels
accumulated: 60+ user sessions, load average 13+, `sshd` starved, the host
eventually became unreachable and had to be rebooted. The bench process itself
was resource-clean; the failure mode was entirely on the client-side SSH
session accumulation.

Two problems needed addressing:

1. **No hard backstop on bench resource usage.** A runaway bench had no cgroup
   limits and could (in theory, even without the SSH issue) starve the host of
   RAM or CPU. GB10 has unified memory and no MIG support, so VRAM cannot be
   cgroup-capped, but RAM and CPU can.
2. **No non-SSH submission path.** Every tool we had for invoking benches
   went through an interactive SSH channel.

## Decision

Use [Spark](https://github.com/feza-ai/spark) (in-house single-node pod
orchestrator, currently `v1.6.0`) as the sole submission path for any DGX
benchmark that runs longer than ~10 seconds.

Concretely:

- Install `spark_1.6.0_linux_arm64.deb` on DGX as a systemd service, backed by
  a local NATS server. The `.deb` ships an nfpm postinstall hook that handles
  user/group, directories, and `systemctl daemon-reload`.
- Author a reusable Pod manifest at `docs/bench/manifests/patchtst-train.yaml`
  that enforces cgroup limits (`memory: 32Gi`, `cpu: "8"`, `nvidia.com/gpu: 1`)
  via Podman. The bench binary is mounted read-only from
  `/var/lib/zerfoo/bin/bench_train`; CUDA 13 runtime libs are mounted
  read-only from `/usr/local/cuda` (zerfoo loads them via purego/dlopen so the
  base image can be plain `ubuntu:24.04`).
- Author a thin client wrapper at `scripts/bench-spark.sh` (bash + curl) that
  substitutes `${RUN_ID}`, `${SAMPLES}`, `${CHANNELS}`, `${EPOCHS}` into the
  manifest, POSTs to `http://192.168.86.250:8080/api/v1/pods`, polls
  `/api/v1/pods/<name>` for a terminal phase, fetches logs, and exits with
  the pod's success/failure status. No persistent SSH session is held.
- Configure the Spark host via `/etc/spark/spark.env` with
  `SPARK_GPU_MAX=1` (serializes GPU pods because GB10 has no MIG),
  `SPARK_SYSTEM_RESERVE_CPU=4000` (4 cores reserved for `sshd`/system), and
  `SPARK_SYSTEM_RESERVE_MEMORY=8192` (8 GB RAM reserved).
- Mark interactive `ssh dgx 'bench_train ...'` as deprecated in both
  `CLAUDE.md` files and forbid any looped/long bash-tool SSH invocation against
  the DGX.

## Alternatives considered

| Option | SSH leak? | Resource caps? | GPU support? | Ops cost |
|---|---|---|---|---|
| `ssh dgx 'bench ...'` (status quo) | **Yes — the incident root cause** | None | Direct | Zero |
| `ssh dgx 'nohup bench &'` + poll log | No (channel closes <1s) | None | Direct | One-line wrapper |
| `systemd-run --user --scope` | Same as ssh, plus cgroup v2 | Yes | Direct | Slightly higher |
| **Spark + Podman manifest** (chosen) | No (HTTP submit) | Yes (Podman cgroups) | Yes (`nvidia.com/gpu`) | Install + manifest authoring, reusable |
| k3s/microk8s on DGX | No | Yes | Yes | Full k8s overhead on a single node |

The `nohup` wrapper would have prevented the specific incident but offered no
resource caps, no central queue, no kill API, and no pod history. k3s gives
the same guarantees as Spark with ~100x the operational surface. Spark is
in-house, already exists at `v1.6.0` with a shipped `.deb`, and speaks the
Kubernetes Pod schema so the manifest is portable if we ever outgrow it.

## Consequences

### Positive
- SSH channel accumulation on DGX is structurally impossible for benches — the
  client POSTs once and polls via short-lived HTTP calls.
- Bench RAM and CPU are cgroup-capped. A runaway bench OOM-kills inside its
  container and cannot take down the host.
- A runaway bench is killable via `curl -X DELETE
  http://192.168.86.250:8080/api/v1/pods/<name>` from any machine on the LAN
  without needing an ssh session.
- The Pod manifest is a portable, version-controlled artifact. Adding new
  bench types (e.g. `bench_tps`, `bench_batch`) is a copy-paste of the
  template with new args.
- Bench outputs persist in `/var/lib/zerfoo/bench-out` on the host across pod
  lifecycle, decoupling log inspection from the bench run.

### Negative / costs
- One more service to keep running on DGX (`spark.service` + `nats-server.service`).
- Debugging a stuck bench now requires `curl .../logs` or `curl .../events`
  instead of direct process inspection — minor learning curve.
- VRAM is still not cgroup-capped (GB10 limitation, no MIG). VRAM contention
  is serialized via `SPARK_GPU_MAX=1`, which means only one GPU pod runs at a
  time on the host. Parallel CPU-only pods remain unaffected.
- The bench binary is built on the host and mounted in, which couples the pod
  to a specific host layout (`/var/lib/zerfoo/bin/bench_train`). Acceptable:
  the alternative is a CUDA-base container image with the binary baked in,
  and the rebuild cycle per bench change would be worse.

### Follow-ups
- Add manifests for other zerfoo benches (`bench_tps`, `bench_batch`) when
  they next need to run on GPU.
- Consider a `scripts/bench-spark.sh` version that reads a manifest name as
  an argument instead of hard-coding `patchtst-train.yaml`.
- If we ever need multi-tenant benches, revisit VRAM isolation (possibly via
  time-slicing or a newer GPU generation with MIG).

## Minimum Spark version: v1.6.1

Commissioning on 2026-04-08 surfaced a Spark bug that made `v1.6.0` unusable
for this plan: the executor only injected `--device nvidia.com/gpu=all` when
`Limits.GPUMemoryMB > 0`, but the manifest parser populates `GPUCount` from
the k8s-standard `nvidia.com/gpu: "1"` key and never sets `GPUMemoryMB`.
Pods scheduled via the HTTP API with a standard GPU request silently ran
without a GPU device and fell back to CPU. Fixed upstream in
[feza-ai/spark#9](https://github.com/feza-ai/spark/pull/9) and released as
`v1.6.1`. **Do not use `v1.6.0` on DGX for this runner.**

Also: the manifest MUST mount `/opt/zerfoo/lib` in addition to
`/var/lib/zerfoo/bin` and `/usr/local/cuda`, because `bench_train` dlopens
`libkernels.so` from that host path.

## References
- `docs/plans/spark-bench-runner.md` — full deployment plan, task breakdown,
  and validation steps.
- `github.com/feza-ai/spark` — Spark source and releases (`v1.6.1` minimum).
- [feza-ai/spark#9](https://github.com/feza-ai/spark/pull/9) — upstream GPU
  passthrough fix required by this runner.
- ADR 017 — DGX hardware validation strategy (prior context).
- Incident writeup in `docs/devlog.md` once validation completes under Epic E4
  of the plan.
