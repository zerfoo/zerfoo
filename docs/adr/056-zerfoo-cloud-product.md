# ADR 056: Zerfoo Cloud Product Architecture

## Status
Proposed

## Date
2026-03-17

## Context
By 2030, Zerfoo will have a mature, battle-tested ML inference and training
framework with 5 years of production use in Wolf's trading system. The Go-native,
zero-CGo, embeddable design makes it uniquely attractive for Go shops that want
ML inference without Python. Offering Zerfoo as a cloud inference API creates a
second revenue stream for Feza, Inc beyond Wolf's trading profits.

The cloud product hypothesis: Go developers want OpenAI-compatible inference APIs
backed by open-weight models, with pay-per-token pricing and no vendor lock-in.

## Decision
Implement multi-tenant cloud serving in serve/cloud/ with the following boundaries:

Multi-Tenancy (serve/cloud/tenant.go):
- Tenant isolation via namespace: separate model registries, rate limits, billing
  accounts per API key
- Resource quotas: max_concurrent_requests, max_tokens_per_minute per tenant tier
- Tenant config stored in a lightweight embedded database (bbolt or pebble)

Billing and Metering (serve/cloud/billing.go):
- Token counter middleware: counts input+output tokens per request
- Usage events published to billing queue (configurable: file, Kafka, HTTP webhook)
- Pricing tiers: per-token (pay-as-you-go) and reserved capacity (monthly flat rate)
- Zerfoo does not process payments; integrates with Stripe or similar via webhook

GPU Resource Management (serve/cloud/resource_manager.go):
- Model registry with LRU eviction: evict least-recently-used models from GPU when
  memory pressure exceeds threshold
- Model load time: target under 10 seconds for 7B models via mmap GGUF
- Shared GPU: multiple small models (1B-3B) can share a single GPU with time-slicing

Deployment Target:
- Initial: single DGX Spark node (internal Feza use + beta customers)
- Year 5 target: multi-node GKE cluster on GCP (Terraform + Cloud Run for API
  gateway, GKE for model workers)
- CDN for GGUF model artifacts: Cloud Storage + signed URLs

This ADR is Proposed (not Accepted) because the cloud product launch depends on
founder approval (per Feza governance: new revenue streams require founder sign-off).

## Consequences
Positive:
- Creates second revenue stream; validates Zerfoo's production quality externally
- Go developer community has no incumbent (no other Go-native inference cloud)
- Dogfoods Zerfoo infrastructure under real multi-tenant load

Negative:
- Requires founder approval before implementation
- Multi-tenancy security is critical; tenant data isolation must be cryptographically
  enforced (separate namespaces are not sufficient; need mTLS per tenant)
- Cloud operations adds significant overhead: on-call rotation, SLA commitments,
  customer support -- out of scope for the engineering roadmap alone
- Regulatory considerations: inference API may be subject to AI Act (EU), EO 14110
  (US) reporting requirements depending on model capabilities
