# Extracted to feza-ai/zerfoo-enterprise

This directory holds packages that were removed from the zerfoo OSS
repository per ADR-090 ("zerfoo OSS scope: cloud, marketplace,
compliance"). Each subdirectory is the historical source tree as of
the extraction commit, retained here for git-history continuity only.

## Status

These packages are **archived**. Do not edit them in this repository.
The canonical source has been moved to the private commercial repo
`feza-ai/zerfoo-enterprise`. New work, bug fixes, and feature
development happen there.

The leading underscore in `_extracted-to-enterprise/` is intentional:
the Go toolchain ignores directories whose names start with `_` or
`.`, so `go build ./...` and `go test ./...` will not pick up the
archived sources.

## Packages

- `cloud/` — multi-tenant control plane (TenantManager, bbolt backend,
  per-tenant rate limiting and model allow-lists).
- `marketplace/` — AWS / GCP / Azure marketplace metering integrations.
- `compliance/` — audit evidence collection, gap analysis, readiness
  reports.

## References

- ADR: `docs/adr/090-zerfoo-oss-scope-cloud-marketplace-compliance.md`
- Plan task: `T124.7.2` in `docs/plan.md`
- Extraction commit range: see `git log --follow` against any file
  under this tree; the extraction commit is the move that placed the
  files here.

## Why archived rather than deleted

Keeping the trees under `docs/archive/_extracted-to-enterprise/`:

1. Preserves `git log --follow` for anyone investigating the history
   of a given file (no tombstone references to a deleted path).
2. Lets reviewers diff against the OSS snapshot when porting fixes
   between OSS and enterprise.
3. Documents the boundary: what is OSS vs commercial is visible in
   the tree, not just in an ADR.

Once the enterprise repo is established and stable, this archive may
be deleted in a future cleanup task.
