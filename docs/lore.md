# Project Lore

Append-only register of gotchas, invariants, and landmines. Unlike
docs/devlog.md (per-session investigation records, time-ordered and
prunable), entries here describe rules that must ALWAYS hold or things
that must NEVER happen again. Entries are topic-ordered, never reordered,
and never pruned. Each has a stable `L-NNNN` ID so it can be cross-
referenced from commit messages, ADRs, and code comments.

Retrieval: grep by tag, e.g. `grep -n "#arena" docs/lore.md` or
`grep -n "#capture" docs/lore.md`. Every entry carries at least one
domain tag, exactly one severity tag (`#critical`, `#gotcha`, or
`#invariant`), and a **Source:** line pointing at the ADR, issue, or
dated devlog entry that established it. Add new entries at the end with
the next `L-NNNN` ID; never renumber. See `~/.claude/skills/lore/SKILL.md`
(the `/lore` skill) for the entry format.

---

## L-0001: Save forward intermediates via SaveForBackward or recompute; never cache them in fields

**Tags:** #arena #backward #gpu-training #invariant
**Date:** 2026-06-11
**Repo:** zerfoo/zerfoo

**Rule:** Any forward intermediate that Backward reads must be preserved through SaveForBackward (which pins its storage) or recomputed from still-live inputs — never stashed in a struct field that outlives the forward pass, because the arena will reuse that storage.
**Why:** Under arena free-list reuse, a tensor cached in a node/layer field can be handed to a later allocation before Backward runs, so Backward dereferences corrupted memory. Two merged fixes established the contract concretely: layernorm had to recompute mean/variance from the input in its f64 backward after GPU arena cache corruption (zerfoo#842), and AdamW had to zero its gradient in-place instead of via `engine.Fill` to avoid arena-realloc corruption (zerfoo#845). ztensor ADR-006 defines the SaveForBackward / pin lifetime contract; the GPU-training-hardening plan E2 (deliverables D5/D6) migrated backward impls to honor it and validated under arena poison mode.
**Trigger:** A `Backward` that reads a `s.someIntermediate` field set during `Forward`, instead of a SaveForBackward'd tensor or a recompute from live inputs. Any new op whose backward touches a cached intermediate.
**Source:** ztensor ADR-006; docs/plan-gpu-training-hardening.md E2 (D5/D6); zerfoo#842, zerfoo#845.

## L-0002: dst-form ops must write into dst's storage; callers must capture the return value

**Tags:** #dst #gpu #reshape #ztensor #critical
**Date:** 2026-04-09
**Repo:** zerfoo/zerfoo

**Rule:** An op that takes a `dst` parameter must write its result into `dst`'s storage; callers must capture and use the returned tensor and must never assume `dst` was filled in place.
**Why:** `GPUEngine.Reshape`'s zero-copy GPUStorage fast path (ztensor `compute/gpu_engine_memory.go`) returned a brand-new tensor aliasing the source storage and ignored `dst` entirely. PatchTST GPU backward discarded the Reshape return value and fed the stale pre-allocated `fc.dX` (all zeros) into `encoderBackward`, freezing training loss at the byte-identical value 0.268357 across every epoch on GB10. The fix (zerfoo commit 73d14342) captured the Reshape return value and passed it downstream. E85's conversion of GPU ops from short-lived local-variable results to preallocated `dst` slots is what surfaced the latent bug.
**Trigger:** `engine.Reshape(ctx, src, shape, dst)` (or any other dst-form op) called for side effect with the return value discarded, while downstream code reads `dst` expecting the result.
**Source:** devlog 2026-04-08 / 2026-04-09 (Wave 7 in-situ instrumentation; E85 preallocation).

## L-0003: Order host reads behind the producing stream on GB10 unified memory

**Tags:** #gb10 #unified-memory #stream-sync #gpu-training #critical
**Date:** 2026-06-11
**Repo:** zerfoo/zerfoo

**Rule:** A host read of a device-written buffer must be ordered behind the stream that produced it (via the host-access sync hooks); never `Data()`-read a gradient a kernel may still be writing.
**Why:** On GB10 cache-coherent unified memory the gradAccumulator host fallback round-tripped every device gradient through the host — `Data()` D2H, add, `TrySet` H2D — once per sample, and the host read raced the still-async kernel that was writing the gradient, producing a deterministic gradient NaN around batch 3-4 (ztensor#137). Unified memory hides this on small runs and exposes it at scale. Fixed upstream with per-device host-access sync hooks; zerfoo#855 additionally takes the round-trip off the hot path by deriving the graph's own engine for fully device-resident in-place f32 accumulation on the graph's stream.
**Trigger:** Any host-side `Data()` / add / `TrySet` on a tensor a GPU op just produced, without a stream sync between the kernel and the read.
**Source:** devlog 2026-06-11; ztensor#137.

## L-0004: Epoch-check arena frees; drop frees that arrive after a Reset

**Tags:** #arena #gc #free-list #gpu-training #critical
**Date:** 2026-06-11
**Repo:** zerfoo/zerfoo

**Rule:** Arena frees must be epoch-checked — a free targeting storage allocated before the last arena `Reset` must be dropped, never applied to the current free list.
**Why:** The training loop's first major GC freed thousands of dead pre-`Reset` storages whose stale `FreeArena` calls poisoned and double-issued free-list memory that live tensors now owned, corrupting gradients (ztensor#138). The symptom is corruption that appears only after the first big GC in a long run. Fixed upstream with arena reset-epochs (`FreeAtEpoch` drops cross-epoch frees); zerfoo#857 bumped to the fixed ztensor.
**Trigger:** A GC finalizer or deferred free that calls `FreeArena` on a storage allocated before the most recent arena `Reset`.
**Source:** devlog 2026-06-11; ztensor#138.

## L-0005: Capture-classify every new op up front, not when capture fails on it

**Tags:** #capture #cuda-graph #invariant
**Date:** 2026-04-16
**Repo:** zerfoo/zerfoo

**Rule:** Classify every new op for CUDA-graph capture compatibility when it is added; do not wait for `cudaStreamEndCapture` to fail on it one op at a time.
**Why:** Capture incompatibilities surface as whack-a-mole. `Gemma4PLECombinedProducer`'s D2H read plus CPU-resident intermediate tensor broke capture first (ADR-088); once that fix expanded the capture region to the full transformer body, `LMHead`'s Transpose broke `cudaStreamEndCapture` at instruction 568 (ADR-089). Each was a separate diagnosis because ops were never capture-classified in advance. The hazard ops are the ones that do a D2H copy (`.Data()`), build a CPU-resident (`CPUStorage`) tensor mid-graph, or branch on a device value.
**Trigger:** Adding an op that reads `.Data()`, wraps a CPU-resident buffer as a tensor mid-graph, or depends on a host-visible device value, without recording it in the capture-incompatible classification.
**Source:** ADR-088; ADR-089; devlog 2026-04-15 / 2026-04-16.

## L-0006: Everything a captured graph touches must be allocation-stable across replays

**Tags:** #capture #cuda-graph #replay #gpu-training #critical
**Date:** 2026-04-16
**Repo:** zerfoo/zerfoo

**Rule:** Every buffer a captured CUDA graph touches must be allocation-stable across replays; no per-call scratch with defer-frees inside the captured region.
**Why:** A captured graph bakes in device pointers, so scratch that is allocated and freed per call makes replays dereference freed or reassigned memory. `tryFlashDecode` launching on a private stream with stream-ordered scratch frees is the crash class in zerfoo#865, and `FusedSDPA`'s flash path crashes with an illegal memory access under graph replay in zerfoo#870. Worse, capture-replay TRAINING currently diverges gradients silently (zerfoo#878, corrected 1.0 seed): losses ascend and the model degenerates — so training under capture-replay is not yet safe.
**Trigger:** `defer engine.Free(scratch)` or any per-call allocation inside code that runs under CUDA-graph capture; enabling capture-replay for a training loop.
**Source:** zerfoo#865; zerfoo#870; zerfoo#878.

## L-0007: Verify tensor storage type empirically; GB10 unified memory lies about locality

**Tags:** #gb10 #unified-memory #storage #gotcha
**Date:** 2026-04-08
**Repo:** zerfoo/zerfoo

**Rule:** Never trust a tensor's name or a nearby comment about whether its storage is on device; log `GetStorage()` and confirm the storage type on the actual host before reasoning from it.
**Why:** On GB10 unified memory, tensors on the "GPU" path are frequently `CPUStorage`. The PatchTST convergence hunt burned multiple waves because a comment near ztensor's `Data()` claimed a fresh D→H memcpy while the "GPU" training path was `CPUStorage` throughout and `SetData` was merely a slice-header swap — both prior workarounds rested on that false premise. The gemma4e H-series ablations hit the same trap: ADR-088 found the PLE producer building CPU-resident tensors mid-graph despite running under CUDA.
**Trigger:** Reasoning about a bug from the storage kind implied by a variable's name, or from a `Data()`-semantics comment, instead of from a logged `tensor.GetStorage()` type on the host you are actually running on.
**Source:** devlog 2026-04-08 (PatchTST GPU convergence saga); devlog 2026-04-21 (gemma4e H-series ablations).

## L-0008: Run DGX GPU benchmarks through Spark only, never interactive SSH

**Tags:** #dgx #spark #benchmark #critical
**Date:** 2026-04-07
**Repo:** zerfoo/zerfoo

**Rule:** Never run DGX GPU benchmarks over interactive SSH; submit anything that loops for more than ~10s through the Spark HTTP API (`scripts/bench-spark.sh` / a Pod manifest).
**Why:** On 2026-04-07 a session ran PatchTST benches via `ssh ndungu@192.168.86.250 'bench_train ...'`. The bash tool auto-backgrounds long-running commands but does not release the SSH channel while they run, so channels accumulated (60+ sessions, load average 13+), `sshd` starved, and the host became unreachable and had to be rebooted (ADR-083). The Spark manifest also cgroup-caps RAM / CPU / GPU, so a runaway bench OOM-kills inside its container instead of taking down the host.
**Trigger:** `ssh ndungu@192.168.86.250 '<anything that loops>'`, `go test -bench`, `bench_train`, `go run ./cmd/bench_*`, or any `go test -tags cuda` that touches GPU kernels, run interactively instead of via Spark.
**Source:** CLAUDE.md (Hardware section); docs/adr/083-spark-bench-runner.md.

## L-0009: A sentinel that unrelated state can satisfy is worse than no sentinel

**Tags:** #sentinel #testing #gpu-training #invariant
**Date:** 2026-04-08
**Repo:** zerfoo/zerfoo

**Rule:** An assertion must fail when the thing it guards is actually absent; never let unrelated state satisfy it and never let a false-positive panic stand in for the metric it was supposed to reach.
**Why:** PR #365's "fix" for the frozen PatchTST loss was validated by a sentinel that compared `*TensorNumeric` wrapper identity — wrappers that were always aliased, so the check always passed while the gradient path stayed broken and loss stayed frozen at 0.268357. The strengthened replacement sentinel then panicked on a false positive (it compared ephemeral `Data()` base pointers that `GPUStorage.Slice()` materializes fresh on every call); that panic looked like the sentinel catching a bug but actually hid that the real convergence assertion was never reached.
**Trigger:** A guard that asserts on wrapper/struct identity, a fixed sentinel value, or a pointer recomputed on each call, rather than on the invariant's real subject (e.g. the backing storage the kernel writes into).
**Source:** devlog 2026-04-08 (CORRECTION — regression is NOT fixed); PR #365.

## L-0010: Reproduce the pre-commit state before blaming a recent commit

**Tags:** #bisect #debugging #gotcha
**Date:** 2026-04-08
**Repo:** zerfoo/zerfoo

**Rule:** Before attributing a regression to a specific commit, reproduce the failure and confirm the prior commit actually passed; write the minimal reproducer first.
**Why:** The PatchTST regression hunt ran eight waves partly because commit 168a938f (PR #365) was trusted as a working fix when it never was — a bisect marker had to be added warning not to trust it. The existing `TestPatchTST_TrainWindowed_EngineConvergence` would have isolated "the bug is in the GPU engine specifically" in a single run; writing that minimal reproducer first would have cut the investigation from about eight waves to two.
**Trigger:** Declaring "commit X broke it" or "commit Y fixed it" from code reading alone, without running the failing case on both sides of the commit.
**Source:** devlog 2026-04-08 (FINAL localization + CORRECTION entries; Lessons).

## L-0011: Don't re-quantize embedding-shaped Q4_K tensors through Q4_K→f32→Q4_0 in the GGUF loader

**Tags:** #gguf #quantization #gemma4e #critical
**Date:** 2026-04-21
**Repo:** zerfoo/zerfoo

**Rule:** The GGUF loader must keep native K-quant storage for embedding-shaped gather targets; do not round-trip Q4_K through f32 to Q4_0 for a tensor that is only ever gathered.
**Why:** `decodeQ4KTensor` (model/gguf/loader.go) re-quantized Q4_K to Q4_0 at load — doubly lossy (Q4_K 6-bit sub-scale noise, then independent Q4_0 per-32-block noise, the two errors stacking) for pure gather targets like gemma4e's `model.ple_embed_tokens.weight` (shape `[262144, 8960]`). Q4_0's block layout only speeds GEMV, never gather, so the extra loss buys nothing; this is the suspected root of degenerate gemma4e decode (H17 showed uniform Q4 gather noise). NOTE / discrepancy with the seed summary: the round-trip is no longer strictly *unconditional*. As of current main a native-Q4_K path exists but is opt-in — it engages only when `ZERFOO_GEMMA4_PLE_NATIVE_Q4K=1` AND `isEmbeddingShape(shape)` holds (`decodeQ5KTensor` / `decodeQ6KTensor` share the same gated guard). With the env var unset the lossy Q4_0 round-trip is still the DEFAULT, and issue #766 (the task to make native-Q4_K storage the standard behavior) remains OPEN.
**Trigger:** Adding or relying on a `decodeQ*KTensor` path for an embedding / gather table without the `isEmbeddingShape` native-storage guard, or running gemma4e decode with `ZERFOO_GEMMA4_PLE_NATIVE_Q4K` unset.
**Source:** zerfoo#766 (OPEN); devlog 2026-04-21 (T99.2.2.8 H21 reference diff, deviation D4).

## L-0012: One top-level directory per commit

**Tags:** #git #commit #workflow #invariant
**Date:** 2026-07-02
**Repo:** zerfoo/zerfoo

**Rule:** Each commit must touch only one top-level directory; never mix changes across top-level dirs (e.g. `docs/` and `training/`) in a single commit.
**Why:** The repo's commit convention — reflected in the `/apply`, `/journal`, and `/lore` workflows, which each stage a single path (`git add docs/lore.md`) — is one directory per commit, so history stays bisectable and PRs stay reviewable per subsystem. NOTE / discrepancy with the seed summary: this could not be verified as an *installed* pre-commit hook in this worktree — there is no `.git/hooks/pre-commit`, no `.githooks/`, no `core.hooksPath`, and the rule is not stated in the project `CLAUDE.md`. Treat it as a documented convention (and honor it) until the enforcing hook is located; if you add such a hook, update this entry with its path.
**Trigger:** A `git add` / commit spanning two top-level directories at once.
**Source:** CLAUDE.md / repo commit hooks (convention; installed hook not located as of 2026-07-02).

## L-0013: A null-pointer kernel launch poisons the whole CUDA context; graceful-degradation tests must skip when CUDA is available

**Tags:** #gb10 #kernel #cuda-context #purego #gotcha
**Date:** 2026-07-02
**Repo:** zerfoo/zerfoo

**Rule:** Never launch a real kernel with NULL device pointers on a live CUDA context. The `*GracefulWithoutCUDA` tests (which pass nil pointers to assert the wrappers error out) MUST guard with `if cuda.Available() { t.Skip("CUDA available, skipping graceful-failure test") }` -- the graceful path is only meaningful when CUDA is absent (klib() nil -> early error return before any launch).
**Why:** With CUDA available, `klib()` is non-nil and a wrapper like `AddFP16(nil,nil,nil,1,nil)` calls `cuda.Ccall(k.launchAddFP16, 0,0,0,1,0)`, launching the FP16 kernel with null device pointers. The on-device null dereference is an illegal memory access that leaves a STICKY error (cuda 700) on the context, so every subsequent test in the package fails at its first cudaMalloc/cudaStreamCreate -- a package-wide IMA cascade that looks like many broken kernels but is one poisoning test. The launch is async, so the wrapper's `checkKernel` sees launch-success and returns nil, which also silently fails the test's own "should return error" assertion. `TestFP16GracefulWithoutCUDA` was the sole graceful test missing the guard (zerfoo#922); its six siblings (counter, elementwise_parity, fp8_ops, gather, offset_memcpy, rope_select) all had it. Corollary: to find the first-faulting test when Spark truncates logs to the tail, run `-v -failfast` so the first failure lands at the tail where truncation cannot hide it.
**Trigger:** A new `*GracefulWithoutCUDA` (or any nil-device-pointer) test without the `cuda.Available()` skip guard; more generally, any code path that can reach a kernel launch with a null/unallocated device pointer on a live context.
**Source:** zerfoo#922; devlog 2026-07-02 (T135.1).
