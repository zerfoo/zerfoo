# ADR 060: Zerfoo Cloud Platform Architecture

## Status
Accepted

## Date
2026-03-18

## Context
Zerfoo Cloud is a managed inference and training platform targeting $2M-$150M ARR
(Years 4-10). Must support multi-tenancy, GPU sharing, token-based billing, and
deployment on AWS/GCP/Azure marketplaces. Existing prototype includes GKE Terraform,
multi-tenant namespace isolation, token metering, and GPU LRU eviction.

Alternatives considered:
1. Container marketplace listing (AMI/Helm) -- 20% AWS revenue share, simpler.
2. SaaS marketplace listing -- 3% revenue share, requires hosted control plane.
3. Self-hosted enterprise license -- no marketplace tax, longer sales cycle.

## Decision
Three-tier deployment model:

1. **Self-hosted (Apache 2.0)**: Users run Zerfoo on their own infrastructure.
   Free forever. This drives adoption.

2. **Cloud marketplace SaaS (Year 4+)**: Zerfoo Cloud listed as SaaS on AWS/GCP/Azure
   marketplaces. 3% revenue share (not 20% container rate). Customers consume against
   their cloud spend commitments (EDP/MACC). Control plane on GKE; data plane
   deployable in customer VPC.

3. **Enterprise self-managed (Year 4+)**: Commercial license for zerfoo-enterprise
   features. Customer runs on their infrastructure with enterprise features enabled.
   Annual contract, no marketplace tax.

Platform architecture:
- Model Repository pattern (Triton convention): models/{name}/{version}/model.gguf
- Tenant isolation via Kubernetes namespace + NetworkPolicy + ResourceQuota
- GPU sharing: MIG for large tenants, time-slicing for small/bursty tenants
- Billing: token-based (input + output tokens, priced separately), Stripe metered
- Kubernetes operator (ZerfooInferenceService CRD) for declarative model serving

## Consequences
Positive:
- SaaS marketplace listing minimizes revenue share (3% vs 20%).
- Marketplace consumption credits accelerate enterprise sales.
- Three-tier model captures value at every adoption stage.

Negative:
- SaaS requires hosted control plane (ops burden).
- Kubernetes operator is significant engineering investment.
- Multi-cloud deployment multiplies testing and ops surface.
