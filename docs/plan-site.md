# Zerfoo Documentation Site

## 1. Context

### Problem Statement

Zerfoo has a polished single-page landing site at zerfoo.feza.ai (GitHub Pages repo:
zerfoo/zerfoo.github.io) but no user documentation. Developers must read source code,
Go doc comments, or scattered Markdown files in the repo to learn the framework. This
blocks adoption -- users need quickstarts, API references, cookbooks, and architecture
guides to evaluate and use Zerfoo.

### Objectives

- Add comprehensive documentation at zerfoo.feza.ai/docs/ without breaking the landing page.
- Cover: getting started, API reference, cookbooks, architecture guides, zonnx conversion, ecosystem.
- Provide client-side search, sidebar navigation, dark/light theme, and mobile responsiveness.
- Automate build and deploy via GitHub Actions.

### Non-Goals

- Redesigning or replacing the existing landing page.
- Building a custom documentation framework.
- Server-side rendering or dynamic content.
- Paid documentation hosting (Readme.io, GitBook, etc.).
- API documentation auto-generation from Go source (pkg.go.dev handles this).

### Constraints and Assumptions

- The site repo (zerfoo/zerfoo.github.io) uses GitHub Pages with a CNAME to zerfoo.feza.ai.
- The existing landing page is a single index.html with inline CSS/JS (~35 KB).
- Hugo is the chosen static site generator (see docs/adr/064-docs-site-hugo.md).
- Content is written in Markdown by the development team.
- No Node.js or Python build dependencies.
- Apache 2.0 license.

### Success Metrics

| Metric | Target |
|--------|--------|
| Documentation pages | 25+ across all sections |
| All quickstart code examples | Tested and working |
| Client-side search | Functional across all pages |
| Mobile responsive | All pages render correctly on 375px viewport |
| Build time | Under 10 seconds |
| Lighthouse performance score | 90+ |

---

## 2. Discovery Summary

**Work type:** Engineering (Hugo setup, CI) + Content (documentation writing)

**Existing site inventory:**
- Landing page (index.html): hero, features, benchmarks, models, blog links, ecosystem, CTA
- Assets: logo.png, zerfoo.png, zerfoo.svg, CNAME
- No existing documentation pages
- Links to pkg.go.dev for API docs and GitHub for getting-started.md

**Content gaps identified:**
- No getting started guide on the site (links to GitHub blob)
- No API reference beyond pkg.go.dev
- No cookbooks or tutorials on the site
- No architecture documentation for end users
- No zonnx conversion guide
- No ecosystem overview with cross-linking

**Technology choice:** Hugo with Book theme (docs/adr/064-docs-site-hugo.md)

---

## 3. Scope and Deliverables

### In Scope

- Hugo project setup in zerfoo.github.io repo
- GitHub Actions workflow for automated build and deploy
- Landing page preserved as root (/)
- Documentation section at /docs/ with sidebar navigation and search
- 7 documentation sections with 25+ pages total
- Dark/light theme matching the landing page aesthetic
- Mobile responsive layout

### Out of Scope

- Blog system (existing blog cards link to GitHub -- keep as-is for now)
- Auto-generated API docs from Go source (pkg.go.dev serves this)
- Internationalization / translations
- User comments or feedback system
- Analytics (can be added later)
- Custom Hugo theme (use existing Book theme with CSS overrides)

### Deliverables Table

| ID | Description | Acceptance Criterion |
|----|-------------|---------------------|
| D1 | Hugo project with Book theme | `hugo build` produces working site with docs sidebar |
| D2 | GitHub Actions CI/CD | Push to main triggers build and deploy to GH Pages |
| D3 | Getting Started section (3 pages) | Install, quickstart, first inference documented |
| D4 | API Reference section (6 pages) | Engine, Trainer, layers, tokenizer, tabular, timeseries |
| D5 | Cookbooks section (5 pages) | Model loading, streaming, quantization, distributed, timeseries |
| D6 | Architecture section (4 pages) | Models, GGUF, GPU backends, computation graph |
| D7 | zonnx section (3 pages) | Overview, ONNX conversion, SafeTensors conversion |
| D8 | Ecosystem section (4 pages) | ztensor, ztoken, float16/float8 overview |
| D9 | Theme customization | Dark/light toggle, colors match landing page |

---

## 4. Checkable Work Breakdown

### E1: Hugo Infrastructure Setup [Foundation]

- [ ] T1.1 Initialize Hugo project in zerfoo.github.io repo  Owner: TBD  Est: 1h  delivers: [Hugo project skeleton]
  - Install Hugo Book theme as a git submodule
  - Configure hugo.yaml: baseURL, title, params, menu structure
  - Move existing index.html to static/ so it serves as the root page
  - Move logo.png, zerfoo.png, zerfoo.svg to static/
  - Preserve CNAME in static/
  - Acceptance: `hugo build` succeeds, landing page loads at /, /docs/ shows Book theme

- [ ] T1.2 Customize Hugo Book theme to match landing page  Owner: TBD  Est: 1h  delivers: [themed docs site]
  - Create assets/css/custom.css with color variables matching landing page
  - Match purple (#8B5CF6), blue (#3B82F6), cyan (#06B6D4) palette
  - Configure dark/light theme support
  - Add Zerfoo logo to docs sidebar header
  - Add "Back to Home" link in sidebar
  - Acceptance: docs pages visually consistent with landing page

- [ ] T1.3 Create GitHub Actions workflow for Hugo build and deploy  Owner: TBD  Est: 30m  delivers: [CI/CD pipeline]
  - Create .github/workflows/hugo.yml
  - Use actions/checkout, peaceiris/actions-hugo, peaceiris/actions-gh-pages
  - Trigger on push to main branch
  - Build with `hugo --minify`
  - Deploy to gh-pages branch
  - Acceptance: push to main auto-deploys, site loads at zerfoo.feza.ai

- [ ] T1.4 Add docs navigation structure  Owner: TBD  Est: 30m  delivers: [sidebar navigation]
  - Create content/docs/_index.md with docs landing page
  - Create section directories: getting-started/, api/, cookbooks/, architecture/, zonnx/, ecosystem/
  - Add _index.md to each section with title and weight for ordering
  - Acceptance: sidebar shows all sections in correct order

### E2: Getting Started Documentation [Content]

- [ ] T2.1 Write installation guide  Owner: TBD  Est: 1h  delivers: [installation page]
  - content/docs/getting-started/installation.md
  - Cover: go get, go install CLI, build from source, verify installation
  - Minimum Go version, CGO_ENABLED=0 note, platform support
  - Acceptance: complete install instructions for Linux, macOS, Windows

- [ ] T2.2 Write quickstart guide  Owner: TBD  Est: 1h  delivers: [quickstart page]
  - content/docs/getting-started/quickstart.md
  - Cover: load model, chat, stream, embed, structured output
  - Each example must be copy-paste runnable
  - Acceptance: all code examples compile and produce expected output

- [ ] T2.3 Write first inference tutorial  Owner: TBD  Est: 1h  delivers: [inference tutorial page]
  - content/docs/getting-started/first-inference.md
  - Step-by-step: create Go project, add dependency, write main.go, run
  - Cover CLI usage: zerfoo pull, zerfoo run, zerfoo serve
  - Show curl against the OpenAI-compatible API
  - Acceptance: reader can go from zero to working inference in under 5 minutes

### E3: API Reference Documentation [Content]

- [ ] T3.1 Write Engine[T] reference  Owner: TBD  Est: 1h  delivers: [Engine API page]
  - content/docs/api/engine.md
  - Cover: compute.Engine[T] interface, CPU/CUDA/ROCm/OpenCL backends
  - Operations: MatMul, Add, Mul, Softmax, RMSNorm, RoPE, etc.
  - Device management, memory allocation, tensor transfers
  - Link to pkg.go.dev for full method signatures

- [ ] T3.2 Write Trainer[T] reference  Owner: TBD  Est: 1h  delivers: [Trainer API page]
  - content/docs/api/trainer.md
  - Cover: training.Trainer, training.TrainConfig, training.TrainResult
  - Optimizers: AdamW, SGD, EMA, SWA
  - Loss functions, backward pass, gradient clipping
  - Distributed training overview

- [ ] T3.3 Write layers reference  Owner: TBD  Est: 1.5h  delivers: [layers API page]
  - content/docs/api/layers.md
  - Cover: 18 sub-packages, key layers (attention, linear, norm, activation, etc.)
  - Architecture-specific layers (RoPE, SwiGLU, GQA, MLA)
  - Usage patterns with code examples

- [ ] T3.4 Write tokenizer reference  Owner: TBD  Est: 45m  delivers: [tokenizer API page]
  - content/docs/api/tokenizer.md
  - Cover: ztoken.Tokenizer, BPE, HuggingFace compatibility
  - Encode, Decode, special tokens, chat templates
  - Loading from HuggingFace tokenizer.json

- [ ] T3.5 Write tabular ML reference  Owner: TBD  Est: 1h  delivers: [tabular API page]
  - content/docs/api/tabular.md
  - Cover: tabular.Ensemble, Train, Predict, Save/Load
  - LightGBM/XGBoost integration via metee
  - Feature importance, cross-validation

- [ ] T3.6 Write time-series reference  Owner: TBD  Est: 1h  delivers: [timeseries API page]
  - content/docs/api/timeseries.md
  - Cover: DLinear, NHiTS, PatchTST, CfC backends
  - TrainWindowed, PredictWindowed, TrainConfig, TrainResult
  - CreateWindows, ParseWindowSizes utilities
  - Model saving and loading

### E4: Cookbooks [Content]

- [ ] T4.1 Write model loading cookbook  Owner: TBD  Est: 45m  delivers: [model loading cookbook]
  - content/docs/cookbooks/model-loading.md
  - Cover: Load from HuggingFace, local GGUF, quantization variants
  - Model caching, custom model paths, memory mapping

- [ ] T4.2 Write streaming cookbook  Owner: TBD  Est: 45m  delivers: [streaming cookbook]
  - content/docs/cookbooks/streaming.md
  - Cover: ChatStream, SSE streaming, token-by-token processing
  - Integration with HTTP handlers, WebSocket forwarding

- [ ] T4.3 Write quantization cookbook  Owner: TBD  Est: 45m  delivers: [quantization cookbook]
  - content/docs/cookbooks/quantization.md
  - Cover: Q4_0, Q4_K_M, Q8_0, FP16, BF16, FP8 types
  - When to use each, quality vs speed tradeoffs
  - How to quantize with zonnx

- [ ] T4.4 Write distributed training cookbook  Owner: TBD  Est: 1h  delivers: [distributed training cookbook]
  - content/docs/cookbooks/distributed-training.md
  - Cover: gRPC coordinator/worker setup, NCCL, gradient exchange
  - Multi-GPU on single node, multi-node training
  - Configuration and monitoring

- [ ] T4.5 Write time-series forecasting cookbook  Owner: TBD  Est: 1h  delivers: [timeseries cookbook]
  - content/docs/cookbooks/timeseries-forecasting.md
  - Cover: end-to-end walkthrough with DLinear and PatchTST
  - Data preparation, windowing, training, evaluation, prediction
  - Comparing backends, hyperparameter tuning

### E5: Architecture Guides [Content]

- [ ] T5.1 Write supported models guide  Owner: TBD  Est: 1h  delivers: [models architecture guide]
  - content/docs/architecture/supported-models.md
  - Cover: Llama 3, Gemma 3, Mistral, Qwen 2, Phi 3/4, DeepSeek V3
  - Architecture-specific features (RoPE theta, GQA, MLA, MoE, sliding window)
  - How to add a new architecture

- [ ] T5.2 Write GGUF format guide  Owner: TBD  Est: 45m  delivers: [GGUF format guide]
  - content/docs/architecture/gguf-format.md
  - Cover: GGUF v3 structure, metadata, tensor layout
  - Why GGUF (ADR-037), compatibility with llama.cpp ecosystem
  - Memory mapping, loading pipeline

- [ ] T5.3 Write GPU backends guide  Owner: TBD  Est: 1h  delivers: [GPU backends guide]
  - content/docs/architecture/gpu-backends.md
  - Cover: CUDA (cuBLAS, custom kernels), ROCm (HIP, rocBLAS), OpenCL (CLBlast)
  - GRAL abstraction layer, runtime detection, purego/dlopen
  - CUDA graph capture, kernel fusion

- [ ] T5.4 Write computation graph guide  Owner: TBD  Est: 45m  delivers: [computation graph guide]
  - content/docs/architecture/computation-graph.md
  - Cover: graph construction, compilation passes, fusion
  - CUDA graph integration, megakernel codegen
  - Memory planning, device placement

### E6: zonnx Conversion Guides [Content]

- [ ] T6.1 Write zonnx overview  Owner: TBD  Est: 30m  delivers: [zonnx overview page]
  - content/docs/zonnx/overview.md
  - Cover: what zonnx does, supported input formats, output format
  - Installation, basic usage

- [ ] T6.2 Write ONNX to GGUF guide  Owner: TBD  Est: 45m  delivers: [ONNX conversion guide]
  - content/docs/zonnx/onnx-to-gguf.md
  - Cover: download from HuggingFace, convert with architecture flag
  - Quantization during conversion (Q4_0, Q8_0)
  - Supported architectures and tensor name mappings

- [ ] T6.3 Write SafeTensors to GGUF guide  Owner: TBD  Est: 45m  delivers: [SafeTensors conversion guide]
  - content/docs/zonnx/safetensors-to-gguf.md
  - Cover: BERT/RoBERTa models, directory structure requirement
  - config.json fields, metadata mapping
  - End-to-end FinBERT example

### E7: Ecosystem Documentation [Content]

- [ ] T7.1 Write ztensor overview  Owner: TBD  Est: 45m  delivers: [ztensor ecosystem page]
  - content/docs/ecosystem/ztensor.md
  - Cover: tensor creation, compute engine, GPU operations
  - Computation graph, device management
  - When to import ztensor directly vs use through zerfoo

- [ ] T7.2 Write ztoken overview  Owner: TBD  Est: 30m  delivers: [ztoken ecosystem page]
  - content/docs/ecosystem/ztoken.md
  - Cover: BPE tokenizer, HuggingFace compatibility
  - Loading tokenizers, encoding/decoding, special tokens

- [ ] T7.3 Write float16/float8 overview  Owner: TBD  Est: 30m  delivers: [numeric types page]
  - content/docs/ecosystem/numeric-types.md
  - Cover: Float16, BFloat16, Float8 E4M3FN
  - When to use each type, precision characteristics
  - Integration with ztensor

- [ ] T7.4 Write ecosystem overview  Owner: TBD  Est: 30m  delivers: [ecosystem landing page]
  - content/docs/ecosystem/_index.md (rich landing, not just a stub)
  - Dependency graph visualization (ASCII or embedded SVG)
  - Which module to import for which use case
  - Version compatibility matrix

---

## 5. Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A: Infrastructure | T1.1, T1.2, T1.3, T1.4 | Hugo setup, theme, CI, nav |
| B: Getting Started | T2.1, T2.2, T2.3 | Install, quickstart, tutorial |
| C: API Reference | T3.1, T3.2, T3.3, T3.4, T3.5, T3.6 | Engine, Trainer, layers, etc. |
| D: Cookbooks | T4.1, T4.2, T4.3, T4.4, T4.5 | Recipes and walkthroughs |
| E: Architecture | T5.1, T5.2, T5.3, T5.4 | Deep dives |
| F: zonnx | T6.1, T6.2, T6.3 | Conversion guides |
| G: Ecosystem | T7.1, T7.2, T7.3, T7.4 | Module overviews |

**Sync points:**
- Wave 1 (Track A) must complete before all other tracks start -- content authors need
  the Hugo project structure and section directories to write into.
- Tracks B-G are fully independent and can run in parallel after Wave 1.

### Waves

#### Wave 1: Infrastructure (4 agents)
- [ ] T1.1 Initialize Hugo project
- [ ] T1.2 Customize theme
- [ ] T1.3 GitHub Actions workflow
- [ ] T1.4 Navigation structure

Note: T1.1 is the foundation. T1.2-T1.4 can start after T1.1 creates the project skeleton.
In practice, Wave 1 runs sequentially as T1.1 first, then T1.2+T1.3+T1.4 in parallel.

#### Wave 2: All Content (10 agents)
All content tracks run in parallel. Each agent writes to its own section directory.

- [ ] T2.1 Installation guide
- [ ] T2.2 Quickstart guide
- [ ] T2.3 First inference tutorial
- [ ] T3.1 Engine[T] reference
- [ ] T3.2 Trainer[T] reference
- [ ] T3.3 Layers reference
- [ ] T3.4 Tokenizer reference
- [ ] T3.5 Tabular ML reference
- [ ] T3.6 Time-series reference
- [ ] T4.1 Model loading cookbook

#### Wave 3: Remaining Content (10 agents)

- [ ] T4.2 Streaming cookbook
- [ ] T4.3 Quantization cookbook
- [ ] T4.4 Distributed training cookbook
- [ ] T4.5 Time-series forecasting cookbook
- [ ] T5.1 Supported models guide
- [ ] T5.2 GGUF format guide
- [ ] T5.3 GPU backends guide
- [ ] T5.4 Computation graph guide
- [ ] T6.1 zonnx overview
- [ ] T6.2 ONNX to GGUF guide

#### Wave 4: Final Content + QA (4 agents)

- [ ] T6.3 SafeTensors to GGUF guide
- [ ] T7.1 ztensor overview
- [ ] T7.2 ztoken overview
- [ ] T7.3 Float16/float8 overview

#### Wave 5: Polish (2 agents)

- [ ] T7.4 Ecosystem overview (depends on T7.1-T7.3 for cross-references)
- [ ] T8.1 Final review and link verification  Owner: TBD  Est: 1h  delivers: [verified site]
  - Verify all internal links work
  - Verify all code examples compile
  - Test search functionality
  - Test mobile responsiveness
  - Run Lighthouse audit
  - Acceptance: all links resolve, Lighthouse 90+, search returns relevant results

---

## 6. Timeline and Milestones

| ID | Milestone | Exit Criteria | Depends On |
|----|-----------|---------------|------------|
| M1 | Hugo infrastructure live | Landing page + empty docs skeleton deployed | Wave 1 |
| M2 | Getting Started complete | 3 pages live, all examples tested | Wave 2 |
| M3 | API + Cookbooks complete | 11 pages live, cross-linked | Waves 2-3 |
| M4 | All content complete | 25+ pages, search working | Waves 3-4 |
| M5 | Launch ready | Link check, Lighthouse 90+, mobile tested | Wave 5 |

---

## 7. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Hugo Book theme conflicts with landing page CSS | Medium | Low | Landing page is a static file outside Hugo's template system |
| R2 | Code examples become stale as APIs evolve | High | Medium | Add CI step that extracts and compiles code blocks |
| R3 | GitHub Pages build fails on Hugo version mismatch | Low | Low | Pin Hugo version in GitHub Actions workflow |
| R4 | Search index grows too large for client-side | Low | Low | Hugo Book uses lunr.js which handles 100+ pages fine |
| R5 | Dark/light theme mismatch between landing and docs | Medium | Medium | Share CSS custom properties between both |

---

## 8. Operating Procedure

### Definition of Done (per page)
- Markdown file exists in correct content/ directory
- Page renders correctly in Hugo local server (`hugo server`)
- All code examples compile (verified manually or via CI)
- Internal links resolve (no broken links)
- Page has proper frontmatter (title, weight, bookToc)
- Dark and light themes render correctly

### Review Steps
- Author writes page in Markdown
- Run `hugo server` locally, verify rendering
- Test code examples by pasting into a Go file
- Check sidebar navigation ordering
- Push to main, verify deployed site

### Commit Discipline
- One commit per page or per logical change
- Conventional Commits: `docs(getting-started): add installation guide`
- Do not mix infrastructure changes with content changes

---

## 9. Progress Log

### 2026-03-21: Plan created
- Created plan-site.md with 7 epics, 30 tasks across 5 waves.
- Created ADR 064: docs-site-hugo.md documenting the Hugo + Book theme decision.
- Existing landing page at zerfoo.feza.ai will be preserved as the root page.
- Documentation will live at /docs/ using Hugo's output directory structure.

---

## 10. Hand-off Notes

### For new contributors
- The site repo is `zerfoo/zerfoo.github.io` (not in the zerfoo/zerfoo monorepo).
- The landing page is a standalone index.html -- do not modify it through Hugo templates.
- Documentation content lives in `content/docs/` as Markdown files.
- Run `hugo server` locally to preview changes before pushing.
- The GitHub Actions workflow auto-deploys on push to main.

### Key URLs
- Live site: https://zerfoo.feza.ai
- Docs (after deploy): https://zerfoo.feza.ai/docs/
- Site repo: https://github.com/zerfoo/zerfoo.github.io
- Hugo docs: https://gohugo.io/documentation/
- Hugo Book theme: https://github.com/alex-shpak/hugo-book

### Prerequisites
- Hugo extended version installed (`go install github.com/gohugoio/hugo@latest`)
- Git access to zerfoo/zerfoo.github.io repo
