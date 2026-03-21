# ADR 064: Use Hugo for Documentation Site

## Status
Accepted

## Date
2026-03-21

## Context
Zerfoo needs a documentation site at zerfoo.feza.ai alongside the existing single-page
landing site (plain HTML). The options considered were:

1. **Plain HTML** -- extend index.html with manually written docs pages.
2. **MkDocs Material** -- Python-based, Markdown-driven, excellent search.
3. **Hugo** -- Go-based, fast, Markdown-driven, GitHub Pages compatible.
4. **Docusaurus** -- React-based, feature-rich, heavier runtime.

## Decision
Use Hugo with the Hugo Book theme for documentation at /docs/, keeping the existing
index.html as the landing page at /.

Rationale:
- Hugo is written in Go, matching the project ecosystem.
- Hugo Book theme provides sidebar navigation, search, and clean docs layout out of the box.
- Hugo generates static HTML that deploys directly to GitHub Pages via GitHub Actions.
- The existing index.html landing page is preserved as-is at the root.
- Hugo outputs to a /docs/ subdirectory, avoiding any conflict with the landing page.
- No runtime dependencies (no Node.js, no Python).
- Hugo is a single binary -- `go install github.com/gohugoio/hugo@latest`.

## Consequences
- Documentation authors write Markdown files in content/docs/.
- Hugo config, themes, and content live in the zerfoo.github.io repo.
- GitHub Actions workflow builds Hugo and deploys to GitHub Pages.
- The landing page (index.html) is copied to the Hugo output root as a static file.
- Search is provided by Hugo Book's built-in client-side search (lunr.js).
- Versioning is handled via git tags and a version selector in the docs nav.
