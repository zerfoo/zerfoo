# Zerfoo Enterprise Support SLA Tiers

This document defines the support tiers available for Zerfoo customers, including response times, resolution targets, escalation paths, and pricing.

## Quick Comparison

| | Community | Professional | Enterprise |
|---|---|---|---|
| **Price** | Free | $2,000/month | $10,000/month |
| **Support Channels** | GitHub Issues | Slack + GitHub Issues | Phone + Slack + Email + GitHub |
| **P1 Response Time** | Best effort | 24 business hours | 4 hours |
| **P2 Response Time** | Best effort | 48 business hours | 8 hours |
| **P3 Response Time** | Best effort | 5 business days | 24 hours |
| **Coverage** | Community-driven | Mon--Fri, 9am--6pm PT | 24/7 (P1), business hours (P2/P3) |
| **Dedicated Engineer** | No | Yes (1) | Yes (2+) |
| **Architecture Review** | No | Quarterly | Monthly |
| **SLA Credits** | No | Yes | Yes |

---

## Priority Definitions

All tiers use the same priority classification for incoming requests.

| Priority | Definition | Examples |
|---|---|---|
| **P1 -- Critical** | Production system is down or severely degraded. Inference serving is unavailable. No workaround exists. | API server crash, model loading failure on all requests, data corruption, CUDA kernel panic causing GPU hang |
| **P2 -- Major** | A major feature is broken or performing significantly below expectations. A workaround may exist. | 50%+ throughput regression, KV cache corruption on specific model, distributed training coordinator failure, incorrect inference output |
| **P3 -- Minor** | A minor issue, general question, or feature request. Production workloads are not materially affected. | Documentation gap, CLI usability issue, non-critical log warning, feature request, configuration question |

---

## Tier 1: Community (Free)

**Support channels:** GitHub Issues only

**Response time:** Best effort -- no guaranteed response time

**Coverage:** Community-driven

**Included:**
- Public documentation and API reference
- Tutorials and quickstart guides
- Community Discord access
- GitHub Issue tracking for bugs and feature requests

**Limitations:**
- No guaranteed response or resolution time
- No private or direct support
- No dedicated engineering resources
- Issues are triaged and addressed at the maintainers' discretion

---

## Tier 2: Professional ($2,000/month)

**Support channels:** Dedicated Slack channel + GitHub Issues

**Coverage:** Monday--Friday, 9:00 AM -- 6:00 PM Pacific Time (excluding US federal holidays)

### Response and Resolution Targets

| Priority | Initial Response | Resolution Target |
|---|---|---|
| P1 -- Critical | 24 business hours | 3 business days |
| P2 -- Major | 48 business hours | 5 business days |
| P3 -- Minor | 5 business days | 10 business days |

### Included

- Dedicated support engineer assigned to your account
- Quarterly architecture review (1 hour, remote)
- Priority bug fixes -- confirmed bugs in your workloads are prioritized in the backlog
- Access to private Slack channel for direct communication with your support engineer
- Onboarding assistance for initial deployment

### Escalation Path

1. Dedicated support engineer
2. Engineering lead
3. CTO

---

## Tier 3: Enterprise ($10,000/month)

**Support channels:** Phone + Slack + Email + GitHub Issues

**Coverage:**
- P1 incidents: 24/7/365
- P2 and P3 issues: Monday--Friday, 9:00 AM -- 6:00 PM Pacific Time

### Response and Resolution Targets

| Priority | Initial Response | Resolution Target |
|---|---|---|
| P1 -- Critical | 4 hours | 8 hours |
| P2 -- Major | 8 hours | 24 hours |
| P3 -- Minor | 24 hours | 3 business days |

### Included

- Dedicated support team (2+ engineers) assigned to your account
- Monthly architecture review (1 hour, remote)
- Custom SLA terms available upon request
- SOC 2 Type II report access
- Deployment assistance for production rollouts, upgrades, and GPU configuration
- Training sessions for your engineering team (up to 4 hours per quarter)
- Priority access to new features and pre-release builds
- Direct phone escalation for P1 incidents

### Escalation Path

1. Dedicated support team
2. VP Engineering
3. CTO (with page for P1 incidents)

---

## SLA Credits

SLA credits apply to Professional and Enterprise tiers when Feza, Inc. fails to meet the committed response or resolution targets.

| Condition | Credit |
|---|---|
| Missed initial response time | 10% of monthly fee |
| Missed resolution target | 10% of monthly fee |
| Multiple SLA misses in a single month | Credits are cumulative |
| **Maximum credit per month** | **30% of monthly fee** |

**Credit eligibility requirements:**
- Customer must report the SLA miss within 5 business days of the incident
- Credits are applied to the next billing cycle and are not redeemable for cash
- SLA tracking begins after the customer submits a support request through an approved channel
- Credits do not apply during scheduled maintenance windows communicated at least 48 hours in advance

---

## Exclusions

The following items are not covered under any support tier:

- **Custom development:** Building features, integrations, or model architectures specific to your use case
- **On-site support:** All support is delivered remotely
- **Non-Zerfoo issues:** Problems originating in third-party software, customer application code, or infrastructure not managed by Feza, Inc.
- **Unsupported configurations:** Deployments on hardware or software platforms not listed in the Zerfoo compatibility matrix
- **End-of-life versions:** Support is provided only for the current major release and one prior major release

---

## Add-Ons

The following services are available as add-ons to Professional and Enterprise tiers. Pricing is provided upon request.

| Add-On | Description | Available With |
|---|---|---|
| **GPU Optimization Consulting** | Performance profiling, kernel tuning, and memory optimization for your specific GPU hardware and model workloads | Professional, Enterprise |
| **Custom Model Integration** | Assistance adding support for model architectures not yet included in Zerfoo (GGUF-compatible) | Professional, Enterprise |
| **Compliance Assistance** | Guidance on deploying Zerfoo in regulated environments (HIPAA, FedRAMP, PCI-DSS) | Enterprise |
| **Dedicated Training Workshop** | Full-day remote training for your engineering team covering inference optimization, distributed training, and production deployment | Professional, Enterprise |
| **Extended Architecture Reviews** | Additional architecture review sessions beyond the quarterly/monthly allocation | Professional, Enterprise |

---

## Contract Terms

- **Commitment:** Annual contract, billed quarterly in advance
- **Payment terms:** Net 30 from invoice date
- **Auto-renewal:** Contracts renew automatically for successive one-year terms unless either party provides 60 days written notice of non-renewal
- **Tier upgrades:** Customers may upgrade tiers at any time; the price difference is prorated for the remainder of the current quarter
- **Tier downgrades:** Downgrades take effect at the start of the next annual term
- **Termination for cause:** Either party may terminate with 30 days written notice if the other party materially breaches the agreement and fails to cure within the notice period

---

## Contact

To discuss Professional or Enterprise support, contact [sales@feza.ai](mailto:sales@feza.ai).
