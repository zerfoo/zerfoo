# Release-Please v1.0.0 Configuration

This directory contains the release-please configs needed to cut v1.0.0 across
all 6 active Zerfoo repos. Each subdirectory mirrors the repo root layout.

## What changed from pre-1.0 configs

1. **Manifest version set to `1.0.0`** — tells release-please the next release
   is v1.0.0.
2. **Removed `bump-minor-pre-major` and `bump-patch-for-minor-pre-major`** —
   these flags only matter for 0.x semver; post-1.0 they are no-ops.
3. **Removed `skip-github-release`** — all repos should create GitHub Releases
   at 1.0.
4. **Standardised `changelog-sections`** across all repos.

## How to apply

Copy each subdirectory's files into the corresponding repo root:

```bash
# From the zerfoo/zerfoo checkout
for repo in float16 float8 ztensor ztoken zonnx; do
  cp docs/release-v1-config/$repo/.release-please-manifest.json ../$repo/
  cp docs/release-v1-config/$repo/release-please-config.json ../$repo/
  cp -r docs/release-v1-config/$repo/.github ../$repo/
done
```

Then commit and push in each repo. The next merge to `main` will trigger a
release-please PR proposing v1.0.0.

## Repo-specific notes

| Repo | Current | Notes |
|------|---------|-------|
| **zerfoo** | 1.7.0 | Already past 1.0; config updated in-place to remove stale pre-major flags |
| **float16** | 0.2.1 | Keeps existing goreleaser job |
| **float8** | 0.3.1 | Adds basic workflow (had none) |
| **ztensor** | 0.1.0 | Adds basic workflow (had none) |
| **ztoken** | 0.1.0 | Adds basic workflow (had none) |
| **zonnx** | 0.6.0 | Adds goreleaser for CLI binary distribution |

## Release order

Respect the dependency graph when cutting releases:

1. `float16` and `float8` (no deps)
2. `ztoken` (no deps)
3. `ztensor` (depends on float16, float8)
4. `zerfoo` (depends on ztensor, ztoken)
5. `zonnx` (standalone)
