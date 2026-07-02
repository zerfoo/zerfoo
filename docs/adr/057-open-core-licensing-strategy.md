# ADR 057: Open-Core Licensing Strategy

## Status
Accepted

## Date
2026-03-18

## Context
Zerfoo needs a commercialization strategy that supports $0-$150M ARR over 10 years
without alienating the open-source community. Historical precedent shows that
restrictive license changes (Redis BSL, Elastic SSPL, HashiCorp BSL) trigger forks,
community exodus, and trust destruction. Apache 2.0 provides no cloud-provider
protection but maximizes adoption. AGPL deters cloud-provider forks but may reduce
enterprise adoption due to compliance concerns.

Alternatives considered:
1. Apache 2.0 everything (current) -- maximum adoption, zero cloud protection.
2. AGPL core -- deters cloud forks (Grafana model) but scares some enterprises.
3. BSL/SSPL -- non-OSI, triggers forks, community backlash.
4. Apache 2.0 core + commercial enterprise features (open-core) -- GitLab model.

## Decision
Maintain Apache 2.0 for the core framework (zerfoo, ztensor, ztoken, zonnx, float16,
float8). Enterprise features (SSO/SAML, RBAC, audit logging, multi-tenancy, advanced
monitoring dashboards) will be offered under a separate commercial license in a
`zerfoo-enterprise` repository starting Year 4 (2029).

The boundary rule: core features that make the framework better for individual
developers stay Apache 2.0. Features that solve organizational/operational problems
(team management, compliance, access control) are commercial.

If a cloud provider forks and offers managed Zerfoo, compete on innovation velocity,
ecosystem integrations, and enterprise trust -- not license restrictions.

## Consequences
Positive:
- Maximum community adoption and contributor trust.
- Clear, defensible boundary between free and paid.
- No risk of fork-triggering license change drama.

Negative:
- Cloud providers can freely offer managed Zerfoo.
- Must compete on execution speed and ecosystem, not legal moats.
- Revenue start delayed to Year 3-4 (no monetization in Years 1-2).
