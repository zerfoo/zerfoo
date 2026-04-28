# ADR 090: OSS scope of cloud/, marketplace/, and compliance/ in zerfoo

## Status
Accepted

## Date
2026-04-27

## Context tags
infrastructure, open-core, governance, packaging

## Context

Zerfoo's mission (per `CLAUDE.md` and ADR-057) is to be a generic, embeddable,
Apache-2.0 ML inference and training framework for Go. Three top-level
packages currently live in the public `zerfoo/` repository whose fit with that
mission is unclear:

- `cloud/` (~14 files: `server.go`, `tenant.go`, `tenant_bbolt.go`,
  `billing.go`, `usage.go`, `audit.go`, `sso.go`, `resource_manager.go`,
  plus tests) -- multi-tenant SaaS serving layer with API-key tenant
  isolation, bbolt tenant store, token metering, SSO/SAML, audit logging,
  and GPU LRU resource management. This is the prototype referenced by
  ADR-056 and ADR-060 ("Zerfoo Cloud").
- `marketplace/` (top-level + `aws/`, `azure/`, `gcp/` subpackages with
  metering, entitlement, fulfillment, subscription, procurement, and
  CloudFormation/ARM/Deployment-Manager templates) -- a unified abstraction
  over AWS / GCP / Azure cloud-marketplace billing APIs. Pure SaaS
  go-to-market plumbing; no inference or training code paths touch it.
- `compliance/` (~6,549 lines: `controls.go`, `dashboard.go`, `evidence.go`,
  `policy.go`, plus `audit/` and `observation/` subpackages) -- SOC 2
  Trust Services Criteria control mapping, evidence collection, policy
  document generation, gap and readiness reports, and continuous
  observation tooling. Targets the organization-running-Zerfoo, not the
  framework itself.

E124's package-layout cleanup needs a verdict on each before T124.1.3 (top-level
package allowlist lint) can land, and before T124.7.2 can execute the move /
keep / archive operation.

Three relevant prior ADRs frame the decision:

- **ADR-057 (Open-Core Licensing Strategy, Accepted, 2026-03-18).** The
  boundary rule is explicit: "core features that make the framework better
  for individual developers stay Apache 2.0. Features that solve
  organizational/operational problems (team management, compliance, access
  control) are commercial." The expected home for commercial features is a
  separate `zerfoo-enterprise` repository starting Year 4 (2029).
- **ADR-056 (Zerfoo Cloud Product Architecture, Proposed).** Specifies
  multi-tenancy, billing, and GPU resource management under
  `serve/cloud/` -- not a top-level `cloud/`. Status is Proposed pending
  founder approval; no commitment to ship inside the OSS repo.
- **ADR-060 (Cloud Platform Architecture, Accepted).** Three-tier model:
  self-hosted Apache 2.0 core, marketplace SaaS (Years 4+), and
  enterprise self-managed (Years 4+). Marketplace SaaS and enterprise
  features are commercial.
- **ADR-084 (Extract crossasset to wolf, Accepted).** Precedent: a
  package whose audience is not "Go developers building generic ML
  applications" is moved out of zerfoo, even at the cost of a major
  version bump. The bar for staying in OSS is "generic ML building
  block."

Applying the ADR-057 / ADR-060 boundary tests:

- `cloud/` -- solves an organizational/operational problem
  (multi-tenant SaaS hosting). Tenant isolation, SSO, audit logging, and
  per-tenant billing are listed verbatim in ADR-057 as the commercial
  side of the line. Single-tenant inference HTTP serving is already
  handled by `serve/`; `cloud/` adds nothing a Go developer embedding
  Zerfoo as a library needs.
- `marketplace/` -- solves a SaaS go-to-market problem (cloud-provider
  billing reconciliation). Has no consumer outside a hosted SaaS control
  plane. ADR-060 places marketplace listings explicitly in the
  commercial tier.
- `compliance/` -- solves an organizational problem (SOC 2 audit
  preparation for the entity running Zerfoo). The framework itself does
  not need SOC 2 controls to run; the operator does. ADR-057 names
  "compliance" as a commercial feature.

None of the three are imported by `inference/`, `generate/`, `serve/` (the
single-tenant serving layer), `training/`, `layers/`, or any other core
path. They are leaf packages that ship in the OSS tarball today purely
because they were prototyped here.

## Decision

Extract all three packages from the public `zerfoo/` repository to a new
private `zerfoo-enterprise/` repository, consistent with ADR-057's
open-core direction and following the ADR-084 extraction precedent.

Per-package verdict:

| Package        | Verdict                                | Target repo               |
|----------------|----------------------------------------|---------------------------|
| `cloud/`       | Extract to `zerfoo-enterprise`         | `feza-ai/zerfoo-enterprise` |
| `marketplace/` | Extract to `zerfoo-enterprise`         | `feza-ai/zerfoo-enterprise` |
| `compliance/`  | Extract to `zerfoo-enterprise`         | `feza-ai/zerfoo-enterprise` |

Rationale per package:

- **`cloud/` -- Extract.** Multi-tenancy, API-key auth, SSO, audit
  logging, per-tenant billing, and GPU LRU eviction are textbook
  commercial-tier features under ADR-057. ADR-056 already directs the
  cloud product code to live under `serve/cloud/` (not a top-level
  `cloud/`) and is itself only Proposed. The single-tenant HTTP
  inference path (`serve/server.go`, OpenAI-compatible) stays in OSS;
  what leaves is the multi-tenant SaaS shell.
- **`marketplace/` -- Extract.** AWS / GCP / Azure marketplace billing
  integration is purely SaaS go-to-market plumbing for the hosted
  control plane described in ADR-060. No OSS user running Zerfoo as an
  embedded library or self-hosted server has any reason to import it.
  Keeping it in OSS pollutes the public package tree with code no
  community consumer will use, and burdens community contributors with
  reviewing cloud-provider billing API changes.
- **`compliance/` -- Extract.** SOC 2 controls, evidence collection,
  policy generation, and observation tooling are organizational
  features for the entity *running* Zerfoo, not the framework itself.
  ADR-057 names compliance explicitly as a commercial feature. The
  package is also the largest of the three (~6.5 KLOC) and would
  dominate the contributor surface area for a feature 99% of OSS users
  will never touch.

Sequencing follows ADR-084:

1. Bootstrap `feza-ai/zerfoo-enterprise` as a new private repo with its
   own `go.mod`. T124.7.2 will perform the actual `git mv` of each
   directory and delete from `zerfoo/`.
2. Cut a major version bump in zerfoo at the time of removal (this will
   be the v3.0.0 cut already anticipated by ADR-084's crossasset
   extraction; the two extractions can ride the same major).
3. Update the T124.1.3 top-level package allowlist to no longer permit
   `cloud/`, `marketplace/`, or `compliance/`.
4. Update `docs/design.md` (T124.8.1) to describe the post-cleanup
   layout and reaffirm "new top-level packages require an ADR."

Wolf and any internal Feza consumers of `cloud/`, `marketplace/`, or
`compliance/` will switch their imports to
`github.com/feza-ai/zerfoo-enterprise/...` at the same major bump.

Out of scope for this ADR:

- The licensing terms of `zerfoo-enterprise` (commercial license per
  ADR-057, exact contract to be drafted at Year 4 launch).
- Whether any individual sub-feature might later be re-introduced into
  OSS as a generic primitive (e.g. a generic rate-limit middleware
  decoupled from tenant identity). Such re-introductions follow the
  standard ADR process and the T124.1.3 allowlist.
- The `serve/cloud/` path mentioned in ADR-056. That path does not
  exist today; if the cloud product is built later, ADR-056 should be
  amended to place it in `zerfoo-enterprise` instead of inside the OSS
  repo, consistent with this ADR.

## Consequences

**Positive:**
- Aligns the public repo with ADR-057's stated open-core boundary.
- Removes ~7-10 KLOC of commercial-tier code from the OSS surface,
  shrinking the public API and review burden.
- Establishes `zerfoo-enterprise` as the canonical home for future
  commercial features (multi-tenancy, SSO, RBAC, audit, marketplace,
  compliance) without case-by-case debate.
- Reaffirms the precedent set by ADR-084: zerfoo OSS only contains
  generic ML infrastructure.
- Closes T124.7.1 with a concrete placement decision T124.7.2 can act
  on.

**Negative:**
- Breaking change: any external consumer of
  `github.com/zerfoo/zerfoo/cloud`, `.../marketplace`, or
  `.../compliance` must update imports. Mitigated by riding the same
  v3.0.0 major as ADR-084.
- Requires creating and maintaining a second repo earlier than the
  Year 4 timeline implied by ADR-057. Acceptable: the prototype code
  already exists, so the marginal cost is repo bootstrap, not new
  development.
- Wolf (the only known internal consumer) must update its imports.
  Coordination handled in T124.7.2.
- Splits CI: any GPU/integration tests under `cloud/` move to
  `zerfoo-enterprise` CI, which will need its own runner config.

**Neutral:**
- License of the moved code remains undecided here; ADR-057 governs.
  Until a commercial license is selected, the extracted packages can
  ship internally under a private repo with no public license.

## References

- ADR-056: Zerfoo Cloud Product Architecture (Proposed)
- ADR-057: Open-Core Licensing Strategy (Accepted)
- ADR-060: Zerfoo Cloud Platform Architecture (Accepted)
- ADR-084: Extract crossasset package from zerfoo to wolf (Accepted)
- docs/plan.md E124.7 (Open-core split decision)
