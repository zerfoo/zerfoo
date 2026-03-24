# Miscellaneous Packages Quality Review

Date: 2026-03-23

## 1. generate/*.go

### debugOnnx cached var -- VERIFIED OK
`debugOnnx` is a package-level `var` initialized via `os.Getenv("ZERFOO_DEBUG_ONNX") == "1"` at load time (line 24 of `generator.go`). It is read-only after init. No race condition.

### stdlib log import -- ISSUE FOUND
`generate/generator.go` still imports `"log"` (line 6) and uses `log.Printf` in 6 call sites (lines 236, 240, 346, 379, 444, 569). Two of these are non-debug fallback warnings (lines 236, 240 for CompileTraced failures); four are gated behind `debugOnnx`. All should be migrated to `log/slog` for consistency with the rest of the codebase. The debug calls should use `slog.Debug` and the fallback warnings should use `slog.Warn`.

**Severity: Low.** Functional but inconsistent with project convention.

### slog usage -- NOT YET ADOPTED
The `generate/` package does not import `log/slog` anywhere. It is the last holdout among production packages.

## 2. meta/*.go

### MAML deterministic seeding -- VERIFIED OK
`NewMAML` correctly uses the `Seed` field: when non-nil, it creates `rand.New(rand.NewPCG(*config.Seed, 0))` (line 96), providing deterministic weight init and task sampling. When nil, it uses `rand.Uint64()` for both PCG seeds (line 98), giving non-deterministic behavior.

The `TestMAML_MetaConvergence` test uses `Seed: &seed` with `seed = 42`, confirming deterministic execution.

### Flaky test risk -- LOW RISK, ONE CONCERN
`TestMAML_FewShotAdaptation` does NOT use a seed (line 93: no Seed field set), so weight initialization is non-deterministic. The test asserts `mse < 2.0`, which is a generous threshold for linear tasks, so flakiness is unlikely but possible in theory. Consider adding a seed for full reproducibility.

## 3. internal/workerpool/pool.go

### sync.Once Close() -- VERIFIED OK
`Close()` uses `sync.Once` correctly (lines 49-54): `close(p.tasks)` is called exactly once, then `p.wg.Wait()` blocks until all workers exit. Multiple concurrent `Close()` calls are safe. The `Submit` method does not check if the pool is closed, but this is acceptable since sending to a closed channel would panic -- callers are expected to coordinate lifecycle.

**Minor concern:** If `Submit` is called after `Close`, it will panic on the channel send. There is no `closed` flag or check in `Submit`. This is a design choice documented by Go convention (like `sync.Pool`) but worth noting.

## 4. support/*.go

### JSON injection in api.go -- VERIFIED SAFE
All error responses use either:
- Static JSON string literals: `http.Error(w, '{"error":"..."}', status)` (safe)
- `json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})` on line 167 (safe -- `json.Encoder` escapes all special characters)

The `err.Error()` value on line 167 comes from `Ticket.Transition()`, which formats using `Status` constants (string type, but values are compile-time constants like `"open"`, `"closed"`). No user input flows into error messages unescaped.

### Other injection vectors -- NONE FOUND
- All request bodies are decoded via `json.NewDecoder` which rejects malformed JSON.
- Webhook URLs come from server-side `WebhookTarget` registration, not from user input.
- The `customerID` query parameter is used as a map lookup key, not interpolated into strings.

### ListByCustomer sort -- BUG FOUND
Lines 158-162 of `ticket.go` attempt to sort results "newest first" but use a single-pass pairwise swap from the ends inward. This is NOT a correct sort -- it only compares each pair `(i, j)` once, where `i` and `j` converge from opposite ends. For 3+ tickets, this produces incorrect ordering. Example: tickets created at times [1, 3, 2] would remain [1, 3, 2] (no swap occurs since 1 < 3) instead of the correct [3, 2, 1].

**Fix:** Use `slices.SortFunc` or `sort.Slice` with a proper comparator.

**Severity: Medium.** Incorrect sort for customers with 3+ tickets.

### Webhook dispatch error swallowed in api.go
Line 56 in `CreateTicket` calls `a.Webhooks.Dispatch(...)` but the returned `[]error` is silently discarded (fire-and-forget). Same pattern on lines 135-139 (AddComment) and 171-177 (CloseTicket). This is likely intentional (webhook failures should not fail the API request), but there is no logging of failed webhook deliveries.

**Severity: Low.** Should log webhook errors via slog.

## 5. timeseries/*.go -- normalizeWindows NaN handling

### NaN propagation -- GAP FOUND
`normalizeWindows` (dlinear.go:60-112) does not filter or detect NaN/Inf values in the input `windows` data. If any input value is NaN:
1. The mean computation (line 81) will become NaN via `means[c][t] += NaN`
2. The std computation will become NaN
3. All normalized outputs for that channel/timestep will be NaN
4. NaN will silently propagate through the entire training loop

The `isFinite` helper exists (line 53) but is never called inside `normalizeWindows`. The callers (`dlinear.go:295`, `nhits.go:296`, `patchtst.go:1168`, `cfc.go:236`) pass windows directly without pre-filtering.

**Severity: Medium.** Input data with NaN values will silently corrupt all training. Should either reject NaN inputs with an error, or replace NaN values with 0 (or channel mean of non-NaN values) before normalization.

### Early halt on NaN/Inf loss -- VERIFIED OK
All training loops (dlinear, nhits, patchtst, cfc) have early-halt checks for NaN/Inf loss after gradient steps. This catches downstream NaN but not upstream input NaN.

## 6. monitor/*.go, shutdown/*.go, recover/*.go

### monitor/drift.go -- CORRECT
Both `PageHinkley` and `ADWIN` implementations are correct:
- Thread-safe (mutex-protected `Observe` and `Reset`)
- `PageHinkley`: standard algorithm with running mean, cumulative sum, and min tracking
- `ADWIN`: correct Hoeffding bound calculation, proper window trimming on drift detection
- Default parameter handling is clean (zero-value defaults to sensible values)

No issues found.

### shutdown/coordinator.go -- CORRECT
- Idempotent shutdown via `done` bool flag (lines 51-54)
- Reverse-order close is correct (line 62)
- Lock is released before calling closers to avoid holding lock during potentially slow operations (lines 57-59)
- All errors are collected and returned

No issues found.

### recover/retrain.go -- CORRECT
- Pipeline phases are well-defined with proper error wrapping via `PipelineError`
- Optional callbacks (RollbackFn, ValidateFn, RedeployFn) are nil-checked before invocation
- `Run` loop correctly checks stream termination
- `RetrainFn` is validated as required in constructor

No issues found.

## 7. serve/disaggregated/*.go

### slog migration -- VERIFIED COMPLETE
No stdlib `"log"` import anywhere in the disaggregated package. The single `slog.Warn` call in `gateway.go:367` uses structured logging with key-value pairs. Migration is complete.

### gRPC security -- CONCERN FOUND
`NewGateway` defaults to `insecure.NewCredentials()` when no `DialOptions` are provided (gateway.go:99). While the `GatewayConfig.DialOptions` field allows callers to supply TLS credentials, the insecure default means that a misconfigured deployment will silently use unencrypted gRPC.

**Severity: Medium.** For production serving of ML inference over gRPC, the default should either:
1. Require explicit dial options (return error if `DialOptions` is empty), or
2. Log a warning when falling back to insecure credentials

### Health check state comparison -- FRAGILE
Line 356 compares gRPC connectivity state via string comparison: `w.conn.GetState().String() == "READY"`. This works but is fragile -- the `connectivity` package exports constants (`connectivity.Ready`, `connectivity.Idle`) that should be used instead for type safety.

### SSE error format -- MINOR ISSUE
Line 316 writes raw `err.Error()` into an SSE event: `fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())`. If the error message contains newlines, this breaks the SSE framing protocol. Should JSON-encode the error or replace newlines.

### Hardcoded EOS token ID
`decode_worker.go:107` hardcodes `nextToken == 2` as the EOS token. This should come from the `DecodeRequest` or model config rather than being hardcoded.

## Cross-Cutting Verification Results

### Remaining stdlib `log` in production code
Only `generate/generator.go` still imports `"log"`. All other production packages have migrated to `log/slog`.

### Remaining panics in layers/
One panic found: `layers/reducesum/reducesum.go:111` -- `panic("ReduceSum layer requires exactly 1 input for backward")`. This should return an error instead of panicking, since the `Backward` method signature already returns `error`.

### Swallowed errors (`_ =`)
Most `_ =` assignments are benign:
- **Intentional:** `_ = f.Close()` in defer (common Go pattern for read-only file handles)
- **Intentional:** `_, _ = fmt.Fprint(...)` (stdout writes rarely fail in CLIs)
- **Concern:** `cmd/cli/finetune_sentiment.go:100` -- `_ = tcfg` suppresses unused variable. The entire function is a stub that pretends to succeed without doing anything. The function prints "Model saved to..." without actually saving.
- **Concern:** `cmd/bench_tps/main.go:112` -- `_, _ = mdl.Generate(...)` discards the warmup generation result AND error. If warmup fails, the subsequent benchmark will produce misleading numbers.

## Summary of Issues

| # | Location | Severity | Issue |
|---|----------|----------|-------|
| 1 | `generate/generator.go` | Low | Still uses stdlib `log` (6 call sites), should migrate to `slog` |
| 2 | `support/ticket.go:158-162` | Medium | `ListByCustomer` sort is broken for 3+ tickets (not a real sort) |
| 3 | `timeseries/dlinear.go:60` | Medium | `normalizeWindows` does not detect/handle NaN input values |
| 4 | `serve/disaggregated/gateway.go:99` | Medium | Insecure gRPC default without warning |
| 5 | `serve/disaggregated/gateway.go:356` | Low | String comparison for gRPC state instead of typed constants |
| 6 | `serve/disaggregated/gateway.go:316` | Low | SSE error message not escaped for newlines |
| 7 | `serve/disaggregated/decode_worker.go:107` | Medium | Hardcoded EOS token ID = 2 |
| 8 | `layers/reducesum/reducesum.go:111` | Low | Panic in Backward instead of returning error |
| 9 | `support/api.go:56,135,172` | Low | Webhook dispatch errors silently discarded (no logging) |
| 10 | `cmd/cli/finetune_sentiment.go:100` | Low | Stub function that fakes success without doing work |
| 11 | `cmd/bench_tps/main.go:112` | Low | Warmup generation error swallowed |
| 12 | `meta/meta_test.go:93` | Low | `TestMAML_FewShotAdaptation` lacks deterministic seed |
