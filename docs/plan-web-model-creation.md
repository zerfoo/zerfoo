# Web model creation execution plan

**Execution entry point:** [Self-contained Spark takeover](handoff-web-model-creation-spark.md).
That document contains the current deployment state, exact working paths,
prescriptive remaining tasks, known defects and acceptance checks. Follow it
instead of interpreting the unchecked checklist below as a fresh start.

## Objective

Position Zerfoo as a toolkit for creating, training, evaluating and running AI
on user-controlled hardware. The website designs a portable project only.
No visitor training runs on Zerfoo infrastructure or the founder's DGX.

## Work and acceptance

- [x] Record architecture and replace inference-only website positioning.
- [x] Deliver accessible conversation and live project preview, with download
  and coding-agent handoff. Unknown tasks remain explicitly unsupported.
- [x] Add bounded server-side design API, reviewed research retrieval,
  deterministic recipe validation and portable project packaging.
- [x] Enforce request limits, per-session limits and a durable global usage
  ceiling before provider dispatch. No provider keys reach the browser.
- [x] Provide versioned local-agent instructions and Kazi acceptance workflow;
  verify downloaded classifier project trains and predicts on local hardware.
- [x] Add automated API/security/packaging tests and browser interaction checks.
- [x] Prepare zer.foo canonical domain and path-preserving legacy redirect;
  deploy and verify after checking available Cloudflare configuration.

The launch implementation is complete for the supported numeric-classification
recipe. Two deployment gates remain outside this repository: inventorying the
existing apex DNS record IDs for rollback, and replacing the legacy host's
client-side redirect with a server-side permanent redirect. Both require access
to the old DNS/hosting owner; no DNS records are guessed or deleted.

## Delivery boundaries

The current verified recipe is numeric classification. Research summaries are
not executable qualifications. Initial architecture selection uses a bounded
verified classifier recipe; other requests produce a design brief for local
engineering, never a falsely runnable model. Website inference has a separate
hard budget and fails closed if spending controls are unavailable. Existing
training and paper-library work is preserved.
