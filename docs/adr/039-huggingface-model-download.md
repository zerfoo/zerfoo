# ADR 039: HuggingFace Model Download via zerfoo pull

## Status
Accepted

## Date
2026-03-15

## Context
Currently, users must manually download GGUF model files and pass file paths to
Zerfoo. This creates friction in the getting-started experience and forces users
to know where to find GGUF quants on HuggingFace. Ollama solves this with
`ollama pull model-name` and a curated registry; llama.cpp relies on users
downloading files manually.

Zerfoo needs a `zerfoo pull` command that downloads GGUF models from HuggingFace
to a local cache, enabling the one-line inference API (`zerfoo.Load("google/gemma-3-4b")`).

## Decision
Implement HuggingFace model download using the HuggingFace HTTP API directly
(no Python dependencies, no external CLIs).

### Design

1. **Model resolution:** `zerfoo pull google/gemma-3-4b` resolves to the
   HuggingFace repo `google/gemma-3-4b-it-GGUF` (or similar). Resolution rules:
   - If the repo contains GGUF files directly, use them.
   - If not, search for a `-GGUF` variant of the repo.
   - If a specific quant is requested (`--quant Q4_K_M`), filter to matching files.
   - Default quant: Q4_K_M (best quality/size tradeoff for most users).

2. **Download:** Use the HuggingFace Hub HTTP API (`/api/models/{repo}` for
   metadata, direct file URLs for download). Support:
   - Range requests for resume-on-disconnect.
   - Progress bar on stderr.
   - SHA256 verification using HuggingFace's commit metadata.
   - Optional HF_TOKEN environment variable for gated models.

3. **Local cache:** Store downloaded files in `~/.cache/zerfoo/models/{org}/{repo}/{filename}`.
   - Symlink-free, flat structure.
   - Cache manifest (JSON) tracks: repo, filename, sha256, download date, file size.
   - `zerfoo list` shows cached models with sizes.
   - `zerfoo rm {model}` removes a cached model.

4. **Library integration:** `zerfoo.Load("google/gemma-3-4b")` checks the local
   cache first, then downloads if not found. `zerfoo.Load("/path/to/model.gguf")`
   continues to work for local files (path detection: starts with `/` or `.`).

### API

CLI:
```
zerfoo pull google/gemma-3-4b              # downloads default Q4_K_M quant
zerfoo pull google/gemma-3-4b --quant Q8_0 # specific quant
zerfoo pull --token $HF_TOKEN meta-llama/... # gated model
zerfoo list                                 # show cached models
zerfoo rm google/gemma-3-4b                # remove from cache
```

Library:
```go
model, err := zerfoo.Load("google/gemma-3-4b") // auto-downloads if not cached
model, err := zerfoo.Load("/path/to/model.gguf") // local file, no download
```

### Implementation Notes

- HTTP client: Go standard library `net/http`. No third-party dependencies.
- HuggingFace API: `GET https://huggingface.co/api/models/{repo}` for repo
  metadata, `GET https://huggingface.co/api/models/{repo}/tree/main` for file
  listing, direct CDN URLs for file download.
- File size awareness: GGUF files can be 1-100+ GB. Must stream to disk, not
  buffer in memory. Use `io.Copy` with a progress-reporting writer.
- Concurrency: single-file download (GGUF is one file per model). No need for
  multi-file parallel download.

## Consequences

**Positive:**
- Removes the biggest friction point in getting started with Zerfoo.
- Enables the one-line inference API (`zerfoo.Load("model-name")`).
- No external dependencies -- pure Go HTTP client.
- Compatible with HuggingFace ecosystem (gated models, token auth).
- Cache is inspectable and manageable via CLI.

**Negative:**
- Couples Zerfoo to HuggingFace's API and URL structure (mitigated: HF API is
  stable and widely used; can add other sources later).
- Large downloads may fail on unreliable connections (mitigated: range request
  resume support).
- No curated model registry like Ollama -- users must know repo names (mitigated:
  documentation with recommended models, and future `zerfoo search` command).
- Gated models require manual HF token setup (unavoidable; same as every tool).
