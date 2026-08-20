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
**Why:** A captured graph bakes in device pointers, so scratch that is allocated and freed per call makes replays dereference freed or reassigned memory. `tryFlashDecode` launching on a private stream with stream-ordered scratch frees was the crash class in zerfoo#865, and `FusedSDPA`'s flash path crashed with an illegal memory access under graph replay in zerfoo#870. Capture-replay TRAINING also diverged gradients silently (zerfoo#878): losses ascended and the model degenerated because the one cached-across-steps loss seed was still arena-backed.
**Resolved (2026-07, T133.1-T133.4):** the whole cluster closed under the SAME contract fix -- move every buffer the captured region touches (or that a strategy/consumer caches across steps) OFF the per-call arena pool and onto allocation-stable storage that keeps a fixed device address for the runner's lifetime. Concretely: (1) #865 -- `tryFlashDecode`'s stream-ordered scratch frees replaced with lifetime-scoped scratch ownership (PR #928); (2) #870 -- `FusedSDPA`'s flash-path scratch became a persistent `gpuScratchBuffer` field on the node instead of a per-call alloc/free (PR #933); (3) #878 -- the cached d(loss)/d(loss)=1 seed (`training/grad_accum.go` `buildOnesSeed`) was re-homed from `engine.Fill`'s arena-pooled `GPUStorage` into a raw nil-pool `GPUStorage` (`NewGPUStorageFromSlice`), the same allocation-stability guarantee `newPersistentGradTensor` already used for gradient accumulators (PR #937). The general pattern going forward: captured-region operands and any state a strategy caches across `Step` calls must be nil-pool `GPUStorage` (or an equivalent non-arena persistent scratch buffer), never `pool.Alloc`-backed, because a later `ResetPool` will silently recycle the arena block to an unrelated intermediate and the graph replays against the wrong data with correct-looking counters. With the root cause fixed, the T129.2 `ZERFOO_UNSAFE_CAPTURE_TRAINING` containment gate on `NewCaptureReplayRunner` was removed (T133.4); `ZERFOO_DISABLE_CUDA_GRAPH` remains as a plain escape hatch.
**Trigger:** `defer engine.Free(scratch)` or any per-call allocation inside code that runs under CUDA-graph capture; caching cross-step training state (a seed, an accumulator, a scratch buffer) in arena-pooled storage instead of nil-pool/persistent storage.
**Source:** zerfoo#865 (PR #928); zerfoo#870 (PR #933); zerfoo#878 (PR #937); devlog 2026-07-02/2026-07-03 (T133.1-T133.3).

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

## L-0014: The GGUF loader validates tensor shape/offset in four near-identical duplicated loops -- fix all four or the guard is fake

**Tags:** #gguf #overflow #loader #critical
**Date:** 2026-08-09
**Repo:** zerfoo/zerfoo

**Rule:** Any bounds/overflow fix to GGUF tensor-descriptor validation (element-count, dimension count, offset, or nesting-depth checks) must be applied to all four load-path sites -- `model/gguf/loader.go`, `loader_mmap.go`, and the two duplicated loops in `split_file.go` -- not just the one path the reproducing test happened to exercise. Extracting the shared logic into a single helper (as F1's `computeNumElements` fix did) is preferred over patching four call sites by hand.
**Why:** deep-review 002 found the same element-count overflow bug (F1) copy-pasted across all four sites, and the pattern repeated: F2 (offset signed-conversion) had duplicate sites in `loader_mmap.go` and `split_file.go`; F3 (unbounded dimension count) and F4 (unbounded metadata array-nesting recursion, found later by the FuzzParse fuzzer, fixed in PR #978) both live in the shared `parser.go` path and are easy to assume "already covered" by the loader-level fixes when they are not. A fix that only patches the site a test happens to hit leaves the other three (or the parser-level path) exploitable by a crafted file that takes a different load method (mmap vs. split vs. non-mmap).
**Trigger:** Any new GGUF-parsing hardening PR that adds a bounds check to only one of `loader.go` / `loader_mmap.go` / `split_file.go` (x2 sites) / `parser.go`, or a regression test that only exercises one load path (e.g. only `LoadTensors`, not also `LoadTensorsMmap`, `LoadTensorsSplit`, `LoadTensorsMmapSplit`).
**Source:** docs/deep-reviews/002-full-codebase.md (F1/F2/F3, Remediation Status section); F4 fix PR #978; devlog 2026-08-08.

## L-0015: Security capabilities in serve/security/ and distributed/tlsconfig.go are correct but invisible until a CLI flag reaches them

**Tags:** #security #cli #adr-094 #invariant
**Date:** 2026-08-09
**Repo:** zerfoo/zerfoo

**Rule:** A security capability (rate limiter, keystore, mTLS config, incident responder, or any future addition to `serve/security/` or `distributed/tlsconfig.go`) is not "shipped" until it is reachable from a `cmd/cli` flag and exercised by a CLI-level test -- library correctness alone does not protect a deployment that never calls the constructor.
**Why:** deep-review 002 found the rate limiter, scoped keystore, and mTLS config all implemented correctly with passing unit tests, but the shipped `serve`/`worker` CLI never called any of their constructors -- an operator reading the source could reasonably believe protections were active that were not wired to anything. T142.3 wired the rate limiter's Start/Stop into the server lifecycle so it had something to call; T145.1 (PR #977) then added the actual `--rate-limit`/`--rate-limit-burst`/`--keystore` flags that construct and pass them in. `--tls-*` flags already existed. ADR-094 codifies this as the standing "ship the defense you write" rule.
**Trigger:** A new security-relevant type or function added under `serve/security/` or `distributed/` without a corresponding `cmd/cli` flag (or other operator-reachable entry point) in the same change, or a security PR whose test coverage stops at the package boundary instead of a CLI-level integration test.
**Source:** docs/deep-reviews/002-full-codebase.md (Remediation Status section); docs/adr/094-untrusted-boundary-security-hardening.md; T145.1 PR #977; T142.3.

## L-0016: Downloading large files from huggingface.co on the DGX host silently truncates unless curl is forced to IPv4 + HTTP/1.1 + --fail

**Tags:** #dgx #network #download #huggingface #gotcha
**Date:** 2026-08-10
**Repo:** zerfoo/zerfoo

**Rule:** On the DGX host (`aitopatom-bfc8`), fetch large files from huggingface.co with `curl -4 --http1.1 -sL --fail --retry 10 --retry-delay 3 --retry-all-errors -C -`, and always verify the downloaded file's size against the server's `Content-Length` (`curl -sIL` following redirects) before trusting a download completed. Never trust `curl -sL`'s exit code 0 alone as proof of a complete file.
**Why:** `zerfoo pull` (and plain `curl -sL`) repeatedly produced files far smaller than the server's advertised size while still exiting 0 or reporting a generic connection error -- one case downloaded to exactly 3.2% of the expected 1.07GB with exit code 0. Root causes, stacked: (1) this host's default dual-stack DNS/routing to huggingface.co over IPv6 fails outright (`curl` with no `-4` got connection failures; `-4`-forced succeeded) -- confirmed via `nvidia-smi`-verified real hardware, not a sandbox artifact. (2) Even over IPv4, huggingface.co's HTTP/2 connections to this host hit stream resets (`curl` exit 92, `CURLE_HTTP2_STREAM`) that `curl -sL` without `--fail` does not treat as a failure -- it silently accepts the truncated body. Forcing `--http1.1` eliminated the stream resets entirely; every download that failed over HTTP/2 succeeded on retry over HTTP/1.1 with `-C -` to resume. One `zerfoo pull` failure that looked like a checksum/data-integrity bug (a full-size download whose SHA-256 disagreed with HuggingFace's own `X-Linked-Etag` header) turned out to be genuine -- confirmed by an independent `curl` download reproducing the exact same hash at the exact expected `Content-Length` twice -- so don't assume every download anomaly is this same truncation bug; verify size first, then investigate mismatches that persist at full size separately.
**Trigger:** Any large (multi-GB) download from huggingface.co initiated from this host, via `zerfoo pull`, `curl`, or any other client, especially over HTTP/2. A file present at the expected path is not evidence it is complete.

## L-0017: Manually-backgrounded processes (nohup + disown) get silently reaped between agent tool-call boundaries; use the harness's own background-task tracking instead

**Tags:** #agent-harness #background-process #gotcha
**Date:** 2026-08-10
**Repo:** zerfoo/zerfoo

**Rule:** For any shell command that must keep running across multiple agent turns (e.g. a long download), use the harness's own background-task mechanism (the Bash tool's `run_in_background` option), not `nohup ... & disown`.
**Why:** Five large model downloads were started via `nohup curl ... & disown` in one turn. By the next check (a few turns later), four of the five processes had vanished with no error in their log files and no OOM-killer entry in `dmesg`/`journalctl` -- they were not killed by resource pressure, they simply stopped existing. The one download still running was the one issued as the current foreground command of that same call. Re-issuing the same downloads via the harness's sanctioned background-task parameter survived across turns reliably and delivered proper per-task completion notifications.
**Trigger:** Reaching for `nohup`/`disown`/`setsid` to detach a long-running command from an agent session instead of using the harness's built-in background-execution support.

## L-0018: Never resolve a parity/bench model from a DIRECTORY -- the flagship GGUFs are staged flat and directory resolution returns the first .gguf, making the whole matrix vacuous

**Tags:** #parity #gguf #models #vacuous-green #dgx #critical
**Date:** 2026-08-20
**Repo:** zerfoo/zerfoo

**Rule:** Any test, benchmark, or script that loads a specific model MUST name the exact GGUF **file**. Parity and bench code resolves it through `tests/parity/modelset` (checked-in row -> filename table in `model-matrix.json`), asserts the resolved path is the file the row declares, checks the GGUF header's `general.architecture`, and records the claim so two rows can never load one file. Never pass a directory to `inference.Load` in a parity/bench path, and never point several `*_MODEL_DIR` variables at one directory expecting each suite to find "its" model.
**Why:** The 9 flagship models are staged FLAT in `/var/lib/zerfoo/models` (no per-model subdirectories), and `inference.Load` resolves a directory via `findGGUF` (`inference/inference.go:317`), which returns the FIRST `.gguf` in `os.ReadDir` order. `scripts/dgx-validate-inpod.sh` used to grep every `ModelDirEnvVar` out of `tests/parity` and export them ALL to that one directory, so every parity suite -- Gemma, Llama, Mistral, Phi, Qwen -- loaded `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` and the matrix went green while proving nothing about eight of the nine models. Reproduced end to end in `TestDirectoryScanCollapsesDistinctRowsOntoOneFile`: two distinct model IDs pointed at one flat temp dir both surface the alphabetically-first stub's magic number (`0x41414141`) in the loader error, never the second stub's (`0x42424242`). Corollary trap: `general.architecture` alone cannot discriminate the staged set -- `llama` covers both Llama 3.2 and Mistral, `qwen2` covers both Qwen2-7B and the DeepSeek-R1 distill, and `gemma3` covers 1B and 4B -- so an architecture-only assertion would still have passed the vacuous matrix. The exact path is the identity check; architecture, `general.name` and file size are corroboration.
**Trigger:** A new parity suite whose `ModelParityConfig` omits `MatrixRow`; any `inference.Load` (as opposed to `inference.LoadFile`) in `tests/parity` or a bench harness; any script exporting more than one `*_MODEL_DIR` to the same path; a parity stage reported as `pass` whose log does not contain a per-row resolved absolute path.
**Source:** T136.6 / S136.6.1; `tests/parity/modelset/matrix.go`; `tests/parity/model_identity_test.go`; `scripts/dgx-validate-inpod.sh`; `docs/bench/manifests/validate-arm64.yaml`.
