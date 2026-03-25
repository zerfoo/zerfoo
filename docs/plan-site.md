# Documentation Cleanup and Website Migration

## 1. Context

### Problem Statement

Zerfoo's documentation has grown organically across 40+ Markdown files, 5 tutorials,
12 cookbook examples, 9 blog posts, and 3 API reference pages -- all sitting inside
the zerfoo/zerfoo repo under docs/. There is no live documentation site; the only web
presence is a landing page at zerfoo.feza.ai. A custom Go-based site generator exists
in docsite/ but was superseded by the Hugo decision (ADR-064). Several files overlap
or duplicate each other (two GPU guides, two enterprise deployment guides, three
getting-started variants, three benchmark files).

Developers must read source code or navigate GitHub blob URLs to learn the framework.
This blocks adoption.

### Objectives

- Audit and classify every docs/ file as user-facing (website) or internal (stays in repo).
- Consolidate overlapping files into single authoritative pages.
- Remove the deprecated custom docsite/ generator.
- Set up Hugo with Book theme in the zerfoo.github.io repo.
- Migrate all user-facing content to the Hugo site at zerfoo.feza.ai/docs/.
- Deploy via GitHub Actions. Preserve the existing landing page at /.
- Delete migrated Markdown files from the zerfoo/zerfoo repo to avoid dual maintenance.

### Non-Goals

- Redesigning or replacing the existing landing page.
- Auto-generating API docs from Go source (pkg.go.dev handles this).
- Server-side rendering or dynamic content.
- Paid documentation hosting.
- Internationalization or translations.
- Rewriting content from scratch -- adapt and consolidate existing material.

### Constraints and Assumptions

- Site repo: zerfoo/zerfoo.github.io, GitHub Pages, CNAME to zerfoo.feza.ai.
- Hugo is the static site generator (docs/adr/064-docs-site-hugo.md).
- No Node.js or Python build dependencies.
- Content is Markdown authored by the development team.
- Apache 2.0 license.
- Each repo in the ecosystem is independent -- commits stay within one repo.

### Success Metrics

| Metric | Target |
|--------|--------|
| Documentation pages live on site | 30+ |
| Duplicate/overlapping files in repo | 0 |
| User-facing docs remaining in repo | 0 (all on website) |
| Client-side search | Functional across all pages |
| Mobile responsive | All pages render on 375px viewport |
| Build time | Under 10 seconds |
| Lighthouse performance score | 90+ |
| docsite/ directory | Deleted |

---

## 2. Discovery Summary

**Work type:** Engineering (Hugo setup, CI, file cleanup) + Content (migration, consolidation)

**Content discovery reference:** .claude/scratch/content-discovery.md

**Existing content inventory:**
- 15 standalone user-facing docs (~5,600 lines)
- 5 tutorials (~1,040 lines)
- 12 cookbook examples with main.go (~2,400 lines)
- 9 blog posts (~1,650 lines)
- 3 API reference pages (~2,540 lines)
- Total: ~12,600 lines of user-facing content to migrate

**Duplicates identified (4 sets):**
1. getting-started.md + quickstart.md + tutorials/01-getting-started.md
2. gpu.md + gpu-setup.md
3. enterprise-deployment.md + enterprise-deployment-guide.md
4. benchmarks.md + benchmark-comparison.md + benchmarking-methodology.md

**Internal docs (stay in repo):** 14 files + adr/ (67 files) + conferences/ + distribution/ + enterprise/ + release-v1-config/

**Deprecated:** docsite/ (custom Go generator, replaced by Hugo per ADR-064)

---

## 3. Scope and Deliverables

### In Scope

- Classify and tag every file in docs/ as user-facing or internal.
- Consolidate 4 sets of duplicate/overlapping files into single authoritative pages.
- Delete docsite/ directory (custom Go generator).
- Initialize Hugo project in zerfoo.github.io with Book theme.
- GitHub Actions CI/CD for automated build and deploy.
- Migrate all user-facing content to Hugo site with proper frontmatter and navigation.
- Add blog section to Hugo site.
- Delete migrated files from zerfoo/zerfoo repo after site is live.
- Final link and quality verification.

### Out of Scope

- Blog comments or feedback system.
- Analytics (can be added later).
- Custom Hugo theme (use Book theme with CSS overrides).
- Rewriting content from scratch.
- Moving ADRs, design.md, devlog.md, or plans to the website.

### Deliverables Table

| ID | Description | Acceptance Criterion |
|----|-------------|---------------------|
| D1 | Hugo project with Book theme | `hugo build` produces working site with docs sidebar |
| D2 | GitHub Actions CI/CD | Push to main triggers build and deploy to GH Pages |
| D3 | Getting Started section (3 pages) | Install, quickstart, first inference -- consolidated from 3 sources |
| D4 | API Reference section (6 pages) | Engine, Trainer, layers, tokenizer, tabular, timeseries |
| D5 | Cookbooks section (12 pages) | All existing cookbook examples migrated |
| D6 | Architecture section (4 pages) | Models, GGUF, GPU backends, computation graph |
| D7 | Deployment section (2 pages) | Production + enterprise consolidated from 3 sources |
| D8 | Blog section (9 posts) | All existing blog posts with Hugo frontmatter |
| D9 | Ecosystem section (4 pages) | ztensor, ztoken, float16/float8 overview |
| D10 | zonnx section (3 pages) | Overview, ONNX conversion, SafeTensors conversion |
| D11 | Contributing section (2 pages) | Good first issues, contributor guide |
| D12 | Reference section (3 pages) | Extensions, API freeze, migration v1, benchmarks |
| D13 | Repo cleanup complete | No user-facing docs in zerfoo/zerfoo, docsite/ deleted |
| D14 | Theme customization | Dark/light toggle, colors match landing page |

---

## 4. Checkable Work Breakdown

### E0: Documentation Audit and Cleanup [Prerequisite]

- [ ] T0.1 Consolidate getting-started docs  Owner: TBD  Est: 45m  delivers: [single getting-started.md]
  - Merge getting-started.md, quickstart.md, and tutorials/01-getting-started.md
  - Keep the best content from each, remove duplication
  - Result: one consolidated file ready for Hugo migration
  - Acceptance: single file covers install, quickstart, and first-run with no repeated content

- [ ] T0.2 Consolidate GPU setup docs  Owner: TBD  Est: 30m  delivers: [single gpu-setup.md]
  - Merge gpu.md and gpu-setup.md into one authoritative GPU guide
  - Acceptance: single file covers CUDA, ROCm, OpenCL setup with no duplication

- [ ] T0.3 Consolidate enterprise deployment docs  Owner: TBD  Est: 45m  delivers: [single enterprise-deployment.md]
  - Merge enterprise-deployment.md and enterprise-deployment-guide.md
  - Acceptance: single file covers all enterprise deployment scenarios

- [ ] T0.4 Consolidate benchmark docs  Owner: TBD  Est: 30m  delivers: [single benchmarks.md]
  - Merge benchmarks.md, benchmark-comparison.md, and benchmarking-methodology.md
  - Acceptance: single file covers results, comparisons, and methodology

- [ ] T0.5 Delete docsite/ directory  Owner: TBD  Est: 5m  delivers: [removed deprecated code]
  - Remove docsite/ (custom Go site generator superseded by Hugo per ADR-064)
  - Acceptance: docsite/ no longer exists, `go build ./...` still passes

- [ ] T0.6 Verify internal docs are correctly classified  Owner: TBD  Est: 15m  delivers: [audit checklist]
  - Confirm plan.md, plan-site.md, plan-gguf-writer.md, design.md, devlog.md,
    QUALITY.md, VISION.md, adr/, conferences/, distribution/, enterprise/sla-tiers.md,
    release-v1-config/, roadmap-progress-*, updates-archive-* all stay in repo
  - Acceptance: classification matches .claude/scratch/content-discovery.md

### E1: Hugo Infrastructure Setup [Foundation]

Depends on: none (can run in parallel with E0)

- [ ] T1.1 Initialize Hugo project in zerfoo.github.io repo  Owner: TBD  Est: 1h  delivers: [Hugo project skeleton]
  - Install Hugo Book theme as a git submodule
  - Configure hugo.yaml: baseURL (zerfoo.feza.ai), title, params, menu structure
  - Move existing index.html to static/ so it serves as the root page
  - Move logo.png, zerfoo.png, zerfoo.svg to static/
  - Preserve CNAME in static/
  - Acceptance: `hugo build` succeeds, landing page loads at /, /docs/ shows Book theme

- [ ] T1.2 Customize Hugo Book theme to match landing page  Owner: TBD  Est: 1h  delivers: [themed docs site]
  - Depends on: T1.1
  - Create assets/css/custom.css with color variables matching landing page
  - Match purple (#8B5CF6), blue (#3B82F6), cyan (#06B6D4) palette
  - Configure dark/light theme support
  - Add Zerfoo logo to docs sidebar header
  - Add "Back to Home" link in sidebar
  - Acceptance: docs pages visually consistent with landing page

- [ ] T1.3 Create GitHub Actions workflow for Hugo build and deploy  Owner: TBD  Est: 30m  delivers: [CI/CD pipeline]
  - Depends on: T1.1
  - Create .github/workflows/hugo.yml
  - Use actions/checkout, peaceiris/actions-hugo, peaceiris/actions-gh-pages
  - Trigger on push to main branch
  - Build with `hugo --minify`
  - Deploy to gh-pages branch
  - Acceptance: push to main auto-deploys, site loads at zerfoo.feza.ai

- [ ] T1.4 Add docs navigation structure  Owner: TBD  Est: 30m  delivers: [sidebar navigation]
  - Depends on: T1.1
  - Create content/docs/_index.md with docs landing page
  - Create section directories: getting-started/, api/, cookbooks/, architecture/,
    deployment/, blog/, zonnx/, ecosystem/, contributing/, reference/
  - Add _index.md to each section with title and weight for ordering
  - Acceptance: sidebar shows all sections in correct order

### E2: Getting Started Migration [Content]

Depends on: T0.1 (consolidated source), T1.4 (nav structure)

- [ ] T2.1 Migrate installation guide  Owner: TBD  Est: 45m  delivers: [installation page]
  - content/docs/getting-started/installation.md
  - Source: consolidated getting-started.md (from T0.1)
  - Add Hugo frontmatter (title, weight, bookToc)
  - Cover: go get, go install CLI, build from source, platform support
  - Acceptance: complete install instructions for Linux, macOS, Windows

- [ ] T2.2 Migrate quickstart guide  Owner: TBD  Est: 45m  delivers: [quickstart page]
  - content/docs/getting-started/quickstart.md
  - Source: consolidated getting-started.md (from T0.1)
  - Add Hugo frontmatter
  - Cover: load model, chat, stream, embed, structured output
  - Acceptance: all code examples compile and produce expected output

- [ ] T2.3 Migrate first inference tutorial  Owner: TBD  Est: 45m  delivers: [inference tutorial page]
  - content/docs/getting-started/first-inference.md
  - Source: tutorials/01-getting-started.md (non-overlapping parts)
  - Add Hugo frontmatter
  - Cover: create Go project, add dependency, write main.go, run, CLI usage
  - Acceptance: reader can go from zero to working inference in under 5 minutes

### E3: Tutorials Migration [Content]

Depends on: T1.4 (nav structure)

- [ ] T3.1 Migrate model loading tutorial  Owner: TBD  Est: 30m  delivers: [model loading tutorial]
  - content/docs/tutorials/model-loading.md
  - Source: tutorials/02-model-loading.md
  - Add Hugo frontmatter, fix internal links to point to site URLs
  - Acceptance: renders correctly, all links resolve

- [ ] T3.2 Migrate text generation tutorial  Owner: TBD  Est: 30m  delivers: [text generation tutorial]
  - content/docs/tutorials/text-generation.md
  - Source: tutorials/03-text-generation.md
  - Add Hugo frontmatter, fix internal links
  - Acceptance: renders correctly, all links resolve

- [ ] T3.3 Migrate API server tutorial  Owner: TBD  Est: 30m  delivers: [API server tutorial]
  - content/docs/tutorials/api-server.md
  - Source: tutorials/04-api-server.md
  - Add Hugo frontmatter, fix internal links
  - Acceptance: renders correctly, all links resolve

- [ ] T3.4 Migrate tabular/timeseries tutorial  Owner: TBD  Est: 30m  delivers: [tabular tutorial]
  - content/docs/tutorials/tabular-timeseries.md
  - Source: tutorials/05-tabular-timeseries.md
  - Add Hugo frontmatter, fix internal links
  - Acceptance: renders correctly, all links resolve

### E4: API Reference Migration [Content]

Depends on: T1.4 (nav structure)

- [ ] T4.1 Migrate generate API reference  Owner: TBD  Est: 45m  delivers: [generate API page]
  - content/docs/api/generate.md
  - Source: api-reference/generate.md (963 lines)
  - Add Hugo frontmatter, fix code blocks and links
  - Acceptance: renders correctly with syntax highlighting

- [ ] T4.2 Migrate inference API reference  Owner: TBD  Est: 45m  delivers: [inference API page]
  - content/docs/api/inference.md
  - Source: api-reference/inference.md (941 lines)
  - Add Hugo frontmatter, fix code blocks and links
  - Acceptance: renders correctly with syntax highlighting

- [ ] T4.3 Migrate serve API reference  Owner: TBD  Est: 30m  delivers: [serve API page]
  - content/docs/api/serve.md
  - Source: api-reference/serve.md (633 lines)
  - Add Hugo frontmatter, fix code blocks and links
  - Acceptance: renders correctly with syntax highlighting

- [ ] T4.4 Write Engine[T] reference  Owner: TBD  Est: 1h  delivers: [Engine API page]
  - content/docs/api/engine.md
  - New content: compute.Engine[T] interface, CPU/CUDA/ROCm/OpenCL backends
  - Operations: MatMul, Add, Mul, Softmax, RMSNorm, RoPE, etc.
  - Link to pkg.go.dev for full method signatures
  - Acceptance: comprehensive Engine reference with working examples

- [ ] T4.5 Write Trainer[T] reference  Owner: TBD  Est: 1h  delivers: [Trainer API page]
  - content/docs/api/trainer.md
  - New content: training.Trainer, TrainConfig, TrainResult, optimizers, loss functions
  - Acceptance: comprehensive Trainer reference with working examples

- [ ] T4.6 Write tokenizer reference  Owner: TBD  Est: 45m  delivers: [tokenizer API page]
  - content/docs/api/tokenizer.md
  - New content: ztoken.Tokenizer, BPE, HuggingFace compatibility
  - Acceptance: comprehensive tokenizer reference

### E5: Cookbook Migration [Content]

Depends on: T1.4 (nav structure)

- [ ] T5.1 Migrate cookbooks 01-06  Owner: TBD  Est: 1.5h  delivers: [6 cookbook pages]
  - Migrate: basic-text-generation, streaming-chat, embedding-similarity,
    openai-server, custom-sampling, structured-json-output
  - For each: add Hugo frontmatter, inline the main.go code, add explanation
  - Source: docs/cookbook/01-06 directories
  - Acceptance: all 6 pages render, code blocks have syntax highlighting

- [ ] T5.2 Migrate cookbooks 07-12  Owner: TBD  Est: 1.5h  delivers: [6 cookbook pages]
  - Migrate: lora-fine-tuning, batch-inference, speculative-decoding,
    tool-calling, rag, vision-multimodal
  - For each: add Hugo frontmatter, inline the main.go code, add explanation
  - Source: docs/cookbook/07-12 directories
  - Acceptance: all 6 pages render, code blocks have syntax highlighting

### E6: Blog Migration [Content]

Depends on: T1.4 (nav structure)

- [ ] T6.1 Migrate blog posts 01-05  Owner: TBD  Est: 1h  delivers: [5 blog pages]
  - Migrate: introducing-zerfoo, benchmark-comparison, architecture-deep-dive,
    why-go-for-ml, migrating-from-ollama
  - For each: add Hugo frontmatter (title, date, author, tags, description)
  - Source: docs/blog/01-05*.md
  - Acceptance: all 5 posts render with proper metadata

- [ ] T6.2 Migrate blog posts 06-09  Owner: TBD  Est: 45m  delivers: [4 blog pages]
  - Migrate: gguf-industry-standard-format, how-we-beat-ollama-cuda-graph-capture,
    ml-inference-go-service-10-lines, zero-cgo-pure-go-ml-inference
  - For each: add Hugo frontmatter
  - Source: docs/blog/gguf-*.md, how-we-*.md, ml-inference-*.md, zero-cgo-*.md
  - Acceptance: all 4 posts render with proper metadata

### E7: Architecture and Deployment Migration [Content]

Depends on: T0.2, T0.3, T0.4 (consolidated sources), T1.4 (nav structure)

- [ ] T7.1 Migrate architecture tour  Owner: TBD  Est: 30m  delivers: [architecture tour page]
  - content/docs/architecture/overview.md
  - Source: architecture-tour.md (500 lines)
  - Add Hugo frontmatter, fix internal links
  - Acceptance: renders correctly, all links resolve

- [ ] T7.2 Migrate GPU setup guide  Owner: TBD  Est: 30m  delivers: [GPU setup page]
  - content/docs/architecture/gpu-setup.md
  - Source: consolidated gpu-setup.md (from T0.2)
  - Add Hugo frontmatter
  - Acceptance: covers CUDA, ROCm, OpenCL setup

- [ ] T7.3 Migrate production deployment guide  Owner: TBD  Est: 30m  delivers: [production deployment page]
  - content/docs/deployment/production.md
  - Source: production-deployment.md
  - Add Hugo frontmatter
  - Acceptance: covers systemd, health checks, TLS, monitoring

- [ ] T7.4 Migrate enterprise deployment guide  Owner: TBD  Est: 30m  delivers: [enterprise deployment page]
  - content/docs/deployment/enterprise.md
  - Source: consolidated enterprise-deployment.md (from T0.3)
  - Add Hugo frontmatter
  - Acceptance: covers HA, mTLS, multi-node, Kubernetes

- [ ] T7.5 Migrate benchmarks  Owner: TBD  Est: 30m  delivers: [benchmarks page]
  - content/docs/reference/benchmarks.md
  - Source: consolidated benchmarks.md (from T0.4)
  - Add Hugo frontmatter
  - Acceptance: covers results, comparisons, methodology

### E8: Reference and Ecosystem Pages [Content]

Depends on: T1.4 (nav structure)

- [ ] T8.1 Migrate extensions guide  Owner: TBD  Est: 20m  delivers: [extensions page]
  - content/docs/reference/extensions.md
  - Source: extensions.md (278 lines)
  - Acceptance: renders correctly

- [ ] T8.2 Migrate API stability and migration guides  Owner: TBD  Est: 30m  delivers: [2 reference pages]
  - content/docs/reference/api-stability.md (source: engine-api-freeze.md)
  - content/docs/reference/migration-v1.md (source: migration-v1.md)
  - Acceptance: both render correctly

- [ ] T8.3 Migrate contributing guide  Owner: TBD  Est: 30m  delivers: [contributing page]
  - content/docs/contributing/overview.md
  - Source: good-first-issues.md + relevant parts of CONTRIBUTING.md
  - Acceptance: renders correctly

- [ ] T8.4 Write ztensor ecosystem page  Owner: TBD  Est: 45m  delivers: [ztensor overview]
  - content/docs/ecosystem/ztensor.md
  - New content: tensor creation, compute engine, GPU operations
  - Acceptance: comprehensive overview with code examples

- [ ] T8.5 Write ztoken ecosystem page  Owner: TBD  Est: 30m  delivers: [ztoken overview]
  - content/docs/ecosystem/ztoken.md
  - New content: BPE tokenizer, HuggingFace compatibility
  - Acceptance: comprehensive overview

- [ ] T8.6 Write numeric types page  Owner: TBD  Est: 30m  delivers: [float16/float8 overview]
  - content/docs/ecosystem/numeric-types.md
  - New content: Float16, BFloat16, Float8 types
  - Acceptance: comprehensive overview

- [ ] T8.7 Write ecosystem landing page  Owner: TBD  Est: 30m  delivers: [ecosystem overview]
  - content/docs/ecosystem/_index.md
  - Dependency graph, which module for which use case, version compatibility
  - Acceptance: provides clear navigation to all ecosystem pages

### E9: zonnx Conversion Guides [Content]

Depends on: T1.4 (nav structure)

- [ ] T9.1 Write zonnx overview  Owner: TBD  Est: 30m  delivers: [zonnx overview page]
  - content/docs/zonnx/overview.md
  - New content: what zonnx does, supported formats, installation
  - Acceptance: clear overview with basic usage

- [ ] T9.2 Write ONNX to GGUF guide  Owner: TBD  Est: 45m  delivers: [ONNX conversion guide]
  - content/docs/zonnx/onnx-to-gguf.md
  - New content: download, convert, quantize, supported architectures
  - Acceptance: end-to-end conversion walkthrough

- [ ] T9.3 Write SafeTensors to GGUF guide  Owner: TBD  Est: 45m  delivers: [SafeTensors conversion guide]
  - content/docs/zonnx/safetensors-to-gguf.md
  - New content: BERT/RoBERTa models, FinBERT example
  - Acceptance: end-to-end conversion walkthrough

### E10: Repo Cleanup [Final]

Depends on: all migration epics (E2-E9) complete and site verified live

- [ ] T10.1 Delete migrated user-facing docs from zerfoo/zerfoo  Owner: TBD  Est: 30m  delivers: [clean repo]
  - Delete: getting-started.md, quickstart.md, gpu.md, gpu-setup.md,
    production-deployment.md, enterprise-deployment.md, enterprise-deployment-guide.md,
    architecture-tour.md, extensions.md, engine-api-freeze.md, migration-v1.md,
    benchmarks.md, benchmark-comparison.md, benchmarking-methodology.md,
    good-first-issues.md
  - Delete: tutorials/, cookbook/, blog/, api-reference/ directories
  - Keep: plan.md, plan-site.md, plan-gguf-writer.md, design.md, devlog.md,
    QUALITY.md, VISION.md, adr/, conferences/, distribution/, enterprise/,
    release-v1-config/, roadmap-progress-*, updates-archive-*
  - Acceptance: no user-facing docs remain in zerfoo/zerfoo, `go build ./...` passes

- [ ] T10.2 Update README.md links to point to website  Owner: TBD  Est: 20m  delivers: [updated README]
  - Change all docs/ links in README.md to https://zerfoo.feza.ai/docs/ URLs
  - Acceptance: all README links resolve to live site pages

- [ ] T10.3 Update CONTRIBUTING.md links  Owner: TBD  Est: 10m  delivers: [updated CONTRIBUTING.md]
  - Change docs/ references to website URLs where appropriate
  - Acceptance: all links resolve

### E11: Final Verification [QA]

Depends on: E10

- [ ] T11.1 Verify all site links resolve  Owner: TBD  Est: 30m  delivers: [link check report]
  - Run link checker against zerfoo.feza.ai
  - Fix any broken internal or external links
  - Acceptance: zero broken links

- [ ] T11.2 Verify code examples compile  Owner: TBD  Est: 45m  delivers: [code verification report]
  - Extract all Go code blocks from site pages
  - Verify they compile with `go build`
  - Acceptance: all code examples compile

- [ ] T11.3 Run Lighthouse audit  Owner: TBD  Est: 15m  delivers: [Lighthouse report]
  - Test 3 representative pages: docs landing, quickstart, API reference
  - Acceptance: performance score 90+ on all pages

- [ ] T11.4 Test mobile responsiveness  Owner: TBD  Est: 15m  delivers: [mobile QA report]
  - Test at 375px, 768px, 1024px viewpoints
  - Acceptance: all pages render correctly, sidebar collapses on mobile

- [ ] T11.5 Test search functionality  Owner: TBD  Est: 15m  delivers: [search QA report]
  - Search for: "quickstart", "Engine", "GGUF", "CUDA", "tabular"
  - Acceptance: all return relevant results within top 3

---

## 5. Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A: Audit + Cleanup | T0.1-T0.6 | Consolidate duplicates, delete deprecated code |
| B: Hugo Infrastructure | T1.1-T1.4 | Hugo setup, theme, CI, navigation |
| C: Getting Started | T2.1-T2.3 | Install, quickstart, first inference |
| D: Tutorials | T3.1-T3.4 | Model loading, text gen, API server, tabular |
| E: API Reference | T4.1-T4.6 | Existing + new API pages |
| F: Cookbooks | T5.1-T5.2 | 12 cookbook examples in 2 batches |
| G: Blog | T6.1-T6.2 | 9 blog posts in 2 batches |
| H: Architecture + Deploy | T7.1-T7.5 | Architecture, GPU, deployment, benchmarks |
| I: Reference + Ecosystem | T8.1-T8.7 | Extensions, contributing, ecosystem pages |
| J: zonnx | T9.1-T9.3 | Conversion guides |
| K: Repo Cleanup | T10.1-T10.3 | Delete migrated files, update links |
| L: QA | T11.1-T11.5 | Link check, code verify, Lighthouse, mobile, search |

**Sync points:**
- Tracks A and B run in parallel (Wave 1).
- Tracks C-J all depend on T1.4 (nav structure). Track C also depends on T0.1.
  Track H depends on T0.2, T0.3, T0.4.
- Track K depends on all of C-J being complete and site verified.
- Track L depends on K.

### Waves

#### Wave 1: Foundation (10 agents)

Tracks A and B run in parallel. All tasks within each track are independent
except T1.2-T1.4 depend on T1.1.

- [ ] T0.1 Consolidate getting-started docs
- [ ] T0.2 Consolidate GPU setup docs
- [ ] T0.3 Consolidate enterprise deployment docs
- [ ] T0.4 Consolidate benchmark docs
- [ ] T0.5 Delete docsite/ directory
- [ ] T0.6 Verify internal docs classification
- [ ] T1.1 Initialize Hugo project
- [ ] T1.2 Customize Hugo Book theme (after T1.1)
- [ ] T1.3 GitHub Actions workflow (after T1.1)
- [ ] T1.4 Navigation structure (after T1.1)

#### Wave 2: Content Migration Batch 1 (10 agents)

All content tracks that depend only on T1.4 and completed consolidation tasks.

- [ ] T2.1 Migrate installation guide
- [ ] T2.2 Migrate quickstart guide
- [ ] T2.3 Migrate first inference tutorial
- [ ] T3.1 Migrate model loading tutorial
- [ ] T3.2 Migrate text generation tutorial
- [ ] T3.3 Migrate API server tutorial
- [ ] T3.4 Migrate tabular/timeseries tutorial
- [ ] T4.1 Migrate generate API reference
- [ ] T4.2 Migrate inference API reference
- [ ] T4.3 Migrate serve API reference

#### Wave 3: Content Migration Batch 2 (10 agents)

Remaining migrations and new content.

- [ ] T4.4 Write Engine[T] reference
- [ ] T4.5 Write Trainer[T] reference
- [ ] T4.6 Write tokenizer reference
- [ ] T5.1 Migrate cookbooks 01-06
- [ ] T5.2 Migrate cookbooks 07-12
- [ ] T6.1 Migrate blog posts 01-05
- [ ] T6.2 Migrate blog posts 06-09
- [ ] T7.1 Migrate architecture tour
- [ ] T7.2 Migrate GPU setup guide
- [ ] T7.3 Migrate production deployment guide

#### Wave 4: Content Migration Batch 3 (10 agents)

Final content pages.

- [ ] T7.4 Migrate enterprise deployment guide
- [ ] T7.5 Migrate benchmarks
- [ ] T8.1 Migrate extensions guide
- [ ] T8.2 Migrate API stability and migration guides
- [ ] T8.3 Migrate contributing guide
- [ ] T8.4 Write ztensor ecosystem page
- [ ] T8.5 Write ztoken ecosystem page
- [ ] T8.6 Write numeric types page
- [ ] T8.7 Write ecosystem landing page
- [ ] T9.1 Write zonnx overview

#### Wave 5: Final Content + Cleanup (5 agents)

- [ ] T9.2 Write ONNX to GGUF guide
- [ ] T9.3 Write SafeTensors to GGUF guide
- [ ] T10.1 Delete migrated docs from repo
- [ ] T10.2 Update README.md links
- [ ] T10.3 Update CONTRIBUTING.md links

#### Wave 6: QA (5 agents)

- [ ] T11.1 Verify all site links
- [ ] T11.2 Verify code examples compile
- [ ] T11.3 Run Lighthouse audit
- [ ] T11.4 Test mobile responsiveness
- [ ] T11.5 Test search functionality

---

## 6. Timeline and Milestones

| ID | Milestone | Exit Criteria | Depends On |
|----|-----------|---------------|------------|
| M1 | Duplicates consolidated, docsite removed | 4 consolidated files, docsite/ deleted | Wave 1 |
| M2 | Hugo infrastructure live | Landing page + empty docs skeleton deployed | Wave 1 |
| M3 | Getting Started + Tutorials live | 7 pages live, all examples tested | Wave 2 |
| M4 | API + Cookbooks + Blog live | 30+ pages, search working | Waves 2-4 |
| M5 | All content migrated | 40+ pages, all sections populated | Wave 5 |
| M6 | Repo cleanup complete | No user-facing docs in repo, all links updated | Wave 5 |
| M7 | Launch ready | Link check, Lighthouse 90+, mobile tested | Wave 6 |

---

## 7. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Hugo Book theme conflicts with landing page CSS | Medium | Low | Landing page is a static file outside Hugo template system |
| R2 | Code examples become stale as APIs evolve | High | Medium | Add CI step that extracts and compiles code blocks |
| R3 | GitHub Pages build fails on Hugo version mismatch | Low | Low | Pin Hugo version in GitHub Actions workflow |
| R4 | Consolidated docs lose important content | Medium | Medium | Diff each source file before and after merge |
| R5 | Broken links after repo cleanup | High | Medium | Run link checker before and after cleanup |
| R6 | Dark/light theme mismatch between landing and docs | Medium | Medium | Share CSS custom properties between both |
| R7 | External sites linking to GitHub blob docs break | Medium | High | Cannot control external links; add redirects if possible |

---

## 8. Operating Procedure

### Definition of Done (per page)

- Markdown file exists in correct content/ directory in zerfoo.github.io
- Page renders correctly in Hugo local server (`hugo server`)
- All code examples compile (verified manually or via CI)
- Internal links resolve (no broken links)
- Page has proper Hugo frontmatter (title, weight, bookToc)
- Dark and light themes render correctly

### Definition of Done (cleanup)

- Source file deleted from zerfoo/zerfoo repo
- No duplicate content remains
- README and CONTRIBUTING links updated to website URLs
- `go build ./...` passes in zerfoo/zerfoo

### Review Steps

- Author writes/migrates page in Markdown
- Run `hugo server` locally, verify rendering
- Test code examples by pasting into a Go file
- Check sidebar navigation ordering
- Push to main, verify deployed site

### Commit Discipline

- One commit per page or per logical change
- Conventional Commits: `docs(getting-started): migrate installation guide to Hugo site`
- Do not mix infrastructure changes with content changes
- Do not mix zerfoo.github.io commits with zerfoo/zerfoo commits

---

## 9. Progress Log

### 2026-03-25: Plan updated -- added cleanup and migration scope

- Expanded plan from "write new content for Hugo site" to "clean up existing docs,
  consolidate duplicates, migrate user-facing content to Hugo site, and remove
  migrated files from repo."
- Added E0 (Documentation Audit and Cleanup) with 6 tasks for consolidating 4 sets
  of duplicate files and removing the deprecated docsite/ generator.
- Updated E2-E9 to reference existing source files instead of writing from scratch.
- Added E10 (Repo Cleanup) with 3 tasks for deleting migrated files and updating links.
- Added E11 (Final Verification) with 5 QA tasks.
- Reorganized waves from 5 to 6 to accommodate cleanup-first approach.
- Created .claude/scratch/content-discovery.md with full inventory.
- Total: 12 epics, 48 tasks across 6 waves.

### 2026-03-21: Plan created

- Created plan-site.md with 7 epics, 30 tasks across 5 waves.
- Created ADR 064: docs-site-hugo.md documenting the Hugo + Book theme decision.
- Existing landing page at zerfoo.feza.ai will be preserved as the root page.
- Documentation will live at /docs/ using Hugo's output directory structure.

---

## 10. Hand-off Notes

### For new contributors

- The site repo is `zerfoo/zerfoo.github.io` (separate from zerfoo/zerfoo).
- The landing page is a standalone index.html in static/ -- do not modify through Hugo templates.
- Documentation content lives in `content/docs/` as Markdown files.
- Most content already exists in the zerfoo/zerfoo repo -- this plan migrates it, not rewrites it.
- Run `hugo server` locally to preview changes before pushing.
- The GitHub Actions workflow auto-deploys on push to main.

### Content migration workflow

1. Read the source file from zerfoo/zerfoo/docs/
2. Add Hugo frontmatter (title, weight, bookToc: true)
3. Fix internal links to point to /docs/ URLs instead of GitHub blob paths
4. Write to content/docs/ in the zerfoo.github.io repo
5. Verify with `hugo server`
6. After all content is migrated and live, delete the source file from zerfoo/zerfoo

### Key URLs

- Live site: https://zerfoo.feza.ai
- Docs (after deploy): https://zerfoo.feza.ai/docs/
- Site repo: https://github.com/zerfoo/zerfoo.github.io
- Hugo docs: https://gohugo.io/documentation/
- Hugo Book theme: https://github.com/alex-shpak/hugo-book

### Prerequisites

- Hugo extended version installed (`go install github.com/gohugoio/hugo@latest`)
- Git access to zerfoo/zerfoo.github.io repo

---

## Appendix

### File Classification Reference

**User-facing (migrate to website):**
getting-started.md, quickstart.md, gpu.md, gpu-setup.md, production-deployment.md,
enterprise-deployment.md, enterprise-deployment-guide.md, architecture-tour.md,
extensions.md, engine-api-freeze.md, migration-v1.md, benchmarks.md,
benchmark-comparison.md, benchmarking-methodology.md, good-first-issues.md,
tutorials/ (5 files), cookbook/ (12 examples), blog/ (9 posts), api-reference/ (3 files)

**Internal (keep in repo):**
plan.md, plan-site.md, plan-gguf-writer.md, design.md, devlog.md, QUALITY.md,
VISION.md, adr/ (67 files), conferences/ (3 files), distribution/ (4 files),
enterprise/sla-tiers.md, release-v1-config/, roadmap-progress-2026-03-17.md,
updates-archive-2026-03-15.md, gophercon-2026-proposal.md

**Delete (deprecated):**
docsite/ (custom Go generator, replaced by Hugo)
