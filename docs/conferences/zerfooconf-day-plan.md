# ZerfooConf Day — Event Plan

## Overview

ZerfooConf Day is a single-day developer conference dedicated to Go-native machine
learning. It is the first event of its kind — bringing together Go developers, ML
engineers, and platform engineers to share production experience building and
deploying ML systems without leaving the Go ecosystem.

- **Format:** Single-day conference (8:00 AM – 6:00 PM) + evening reception
- **Target size:** 200–300 attendees
- **Cadence:** Annual (first edition targets 2031)
- **Organizer:** Feza, Inc

## Target Audience

| Segment | Why They Attend |
|---------|----------------|
| Go developers | Learn to add ML inference and training directly to Go services |
| ML engineers | Explore alternatives to Python-centric toolchains for production serving |
| Platform engineers | Understand Kubernetes-native ML deployment with single-binary simplicity |
| Open-source contributors | Connect with maintainers, find contribution opportunities |
| Technical leaders | Evaluate Go-native ML for their organization's stack |

**Attendee profile:** Primarily mid-to-senior engineers (3–15 years experience) who
build or operate production systems. Comfortable with Go or adjacent compiled
languages. ML depth varies from beginner to expert.

## Venue Requirements

### Must-Have

- Main hall: 300-person theater seating with stage, large projection, and broadcast audio
- Two breakout rooms: 80-person capacity each for parallel tracks
- Workshop room: 40 seats with power outlets and stable Wi-Fi for hands-on labs
- Sponsor/expo area: 2,000+ sq ft open space adjacent to main hall
- Speaker green room with A/V preview capability
- Strong, dedicated Wi-Fi (300+ concurrent devices)
- On-site catering space for breakfast, lunch, and breaks
- A/V: stage monitors, wireless lavalier + handheld mics, live stream capture

### Nice-to-Have

- Outdoor terrace or lounge for informal networking
- Recording studio or quiet room for post-talk interviews
- Proximity to hotels and public transit
- On-site parking for 50+ vehicles

### Location Criteria

Primary target: San Francisco Bay Area (proximity to Go and ML engineering talent).
Secondary options: Seattle, Austin, or New York. Evaluate colocating with GopherCon
or a major ML conference if scheduling permits — shared travel reduces attendee cost.

## Schedule

### 8:00 – 9:00 | Registration and Breakfast

Open registration desk. Continental breakfast in the expo/sponsor area.
Attendees pick up badges, swag bags, and sponsor materials.

### 9:00 – 9:15 | Opening Remarks

Welcome, code of conduct, logistics, Wi-Fi credentials, schedule overview.

### 9:15 – 10:00 | Keynote: The State of Go-Native ML

Delivered by the Zerfoo founder. Covers the past year's milestones, performance
benchmarks, community growth, and the roadmap ahead. Announces major new features
or partnerships.

### 10:00 – 10:15 | Break

### 10:15 – 11:45 | Morning Technical Talks (2 tracks)

**Track A — Inference and Performance**

| Time | Talk | Duration |
|------|------|----------|
| 10:15 | Building a CUDA Graph Pipeline in Pure Go | 30 min |
| 10:50 | Quantized Inference: From FP32 to Q4_K_M Without Losing Your Mind | 25 min |
| 11:20 | Lightning talks (3 x 5 min + Q&A) | 25 min |

**Track B — Platform and Production**

| Time | Talk | Duration |
|------|------|----------|
| 10:15 | Running Zerfoo on Kubernetes: The Operator Story | 30 min |
| 10:50 | Multi-Model Serving at Scale with LRU GPU Eviction | 25 min |
| 11:20 | Lightning talks (3 x 5 min + Q&A) | 25 min |

### 11:45 – 12:45 | Lunch

Catered lunch in the expo area. Sponsor booths open. Birds-of-a-feather
tables: "Getting Started", "GPU Kernels", "Production Deployment",
"Contributing to Zerfoo".

### 12:45 – 2:15 | Afternoon Technical Talks (2 tracks)

**Track A — Deep Dives**

| Time | Talk | Duration |
|------|------|----------|
| 12:45 | Inside the Computation Graph Compiler: Fusion, Capture, Megakernels | 30 min |
| 13:20 | ARM NEON SIMD in Plan 9 Assembly: A Practical Guide | 25 min |
| 13:50 | Lightning talks (3 x 5 min + Q&A) | 25 min |

**Track B — Ecosystem and Applications**

| Time | Talk | Duration |
|------|------|----------|
| 12:45 | From ONNX to GGUF: Model Conversion with zonnx | 30 min |
| 13:20 | Building a RAG Pipeline with Zerfoo and Go | 25 min |
| 13:50 | Lightning talks (3 x 5 min + Q&A) | 25 min |

### 2:15 – 2:30 | Break

### 2:30 – 4:00 | Hands-On Workshops (parallel)

| Workshop | Room | Capacity |
|----------|------|----------|
| Workshop A: Your First Zerfoo Inference Server in 60 Minutes | Breakout 1 | 40 |
| Workshop B: Writing Custom CUDA Kernels with purego | Breakout 2 | 40 |
| Workshop C: Fine-Tuning a Small Language Model with Zerfoo | Workshop Room | 40 |

Each workshop provides a pre-configured cloud dev environment (Codespaces or
similar) so attendees do not need GPU hardware on their laptops.

### 4:00 – 4:15 | Break

### 4:15 – 5:00 | Community Panel: The Future of ML in Go

Moderated panel with 4–5 speakers: Zerfoo maintainers, users running Zerfoo in
production, and Go community leaders. Audience Q&A for the second half.

### 5:00 – 5:30 | Closing Keynote and Awards

Recap of the day. Announce community awards: Best Lightning Talk, Outstanding
Contributor, Best Workshop Demo. Preview of the next year's roadmap. Call for
contributions.

### 5:30 – 6:00 | Networking Reception

Drinks and appetizers in the expo area. Informal conversations with speakers,
sponsors, and maintainers.

## Speaker Recruitment Plan

### Internal Speakers (Feza, Inc)

- Keynote and closing keynote: Zerfoo founder
- 2–3 deep-dive technical talks from core maintainers (graph compiler, CUDA
  kernels, serving infrastructure)

### External Speakers

- **Call for Proposals (CFP):** Open 4 months before the event. Promote via Go
  community channels (Gopher Slack, Go blog, Reddit r/golang, X/Twitter).
- **Invited speakers:** Reach out directly to 5–8 engineers from companies using
  Zerfoo in production or building complementary Go ML tools.
- **Lightning talks:** Open submission with lower barrier — 5-minute slots
  encourage first-time speakers.
- **Diversity:** Actively recruit speakers from underrepresented groups in the Go
  and ML communities. Offer travel grants for accepted speakers who need support.

### CFP Timeline

| Milestone | Timing |
|-----------|--------|
| CFP opens | T-4 months |
| CFP closes | T-2.5 months |
| Notifications sent | T-2 months |
| Schedule published | T-6 weeks |
| Speaker prep deadline (slides) | T-2 weeks |

### Speaker Support

- Travel and hotel covered for all accepted speakers
- Speaker dinner the evening before the event
- Dedicated A/V rehearsal slot the morning of the event
- Professional video recording of all talks for post-event publishing

## Sponsorship Tiers

| Tier | Price | Included | Limit |
|------|-------|----------|-------|
| **Platinum** | $25,000 | Logo on stage backdrop, 10-min keynote slot, large booth (10x10), 8 attendee passes, logo on all printed materials, social media feature, branded swag item in attendee bags | 2 sponsors |
| **Gold** | $15,000 | Medium booth (8x8), 5 attendee passes, logo on website and slides, social media mention, insert in attendee bags | 4 sponsors |
| **Silver** | $7,500 | Small booth (6x6), 3 attendee passes, logo on website | 6 sponsors |
| **Community** | $2,500 | Logo on website, 2 attendee passes, mention in opening remarks | Unlimited |
| **Workshop** | $10,000 | Named workshop sponsor, banner in workshop room, 4 attendee passes, logo on workshop materials | 3 sponsors |

### Target Sponsors

- Cloud providers (GCP, AWS, Azure) — GPU compute and Kubernetes
- GPU hardware vendors (NVIDIA, AMD, Intel)
- Developer tooling companies (JetBrains, GitHub, Sourcegraph)
- Go-focused companies (enterprise Go users, Go consultancies)
- AI/ML infrastructure companies (model registries, observability, vector databases)

### Sponsorship Revenue Target

Conservative: 1 Platinum + 2 Gold + 3 Silver + 3 Community + 1 Workshop =
$25K + $30K + $22.5K + $7.5K + $10K = **$95,000**

Stretch: 2 Platinum + 4 Gold + 6 Silver + 5 Community + 2 Workshop =
$50K + $60K + $45K + $12.5K + $20K = **$187,500**

## Marketing Timeline

| Timing | Activity |
|--------|----------|
| T-6 months | Announce ZerfooConf: blog post, social media, Gopher Slack. Open early-bird registration. Begin sponsor outreach. |
| T-5 months | Publish speaker lineup (invited speakers). Weekly social media posts featuring Zerfoo use cases. |
| T-4 months | Open CFP. Launch email campaign to Go meetup organizers. Partner with Go podcasts for promotion. |
| T-3 months | Close early-bird pricing. Announce first sponsors. Publish workshop descriptions. |
| T-2.5 months | Close CFP. Begin talk selection. |
| T-2 months | Publish full schedule. Send speaker notifications. Open standard registration. |
| T-6 weeks | Email campaign: schedule highlights, speaker spotlights, travel tips. |
| T-4 weeks | Last-chance registration push. Publish attendee guide (venue, logistics, schedule). |
| T-2 weeks | Final logistics emails. Speaker slides due. Social media countdown. |
| T-1 week | Final attendee count to venue/catering. Print badges and signage. |
| Event day | Live social media coverage. Photographer on-site. |
| T+1 week | Thank-you emails to attendees, speakers, sponsors. Post-event survey. |
| T+2 weeks | Publish talk recordings on YouTube. Share key metrics and highlights blog post. |
| T+4 weeks | Analyze survey results. Begin planning for next year. |

### Channels

- **Owned:** Zerfoo blog (feza.ai/blog), Zerfoo GitHub Discussions, email list
- **Community:** Gopher Slack (#zerfoo, #general), Reddit r/golang, Hacker News
- **Social:** X/Twitter (@zerfoo), LinkedIn, Mastodon
- **Partnerships:** Go meetup groups, GopherCon cross-promotion, Go podcasts
  (Go Time, Cup o' Go)
- **Paid:** Targeted LinkedIn ads to Go + ML engineers, Google Ads for "Go ML
  framework" keywords

## Budget Breakdown

### Expenses

| Category | Estimate | Notes |
|----------|----------|-------|
| **Venue rental** | $15,000–$25,000 | Full-day rental including breakout rooms, A/V infrastructure |
| **Catering** | $18,000–$25,000 | Breakfast, lunch, afternoon snacks, reception (est. $75–$90/person x 250) |
| **A/V and production** | $12,000–$18,000 | Sound, projection, live streaming, video recording, stage setup |
| **Video post-production** | $3,000–$5,000 | Editing and publishing 15–20 talk recordings |
| **Speaker travel and hotel** | $15,000–$25,000 | Flights and 2-night hotel for 10–15 external speakers |
| **Speaker dinner** | $2,000–$3,000 | Evening before event, 20–25 people |
| **Swag** | $5,000–$8,000 | T-shirts, stickers, badges, lanyards, tote bags (250 units) |
| **Marketing** | $5,000–$10,000 | Paid ads, design work, printed materials, signage |
| **Registration platform** | $1,000–$2,000 | Ti.to, Tito, or similar |
| **Insurance** | $1,500–$2,500 | Event liability insurance |
| **Photography** | $2,000–$3,000 | Professional event photographer |
| **Miscellaneous** | $3,000–$5,000 | Contingency, supplies, last-minute needs |
| **Total** | **$82,500–$131,500** | |

### Revenue

| Source | Conservative | Stretch |
|--------|-------------|---------|
| Sponsorship | $95,000 | $187,500 |
| Ticket sales (200 x $150 early-bird / $200 standard) | $35,000 | $52,500 |
| Workshop add-on (40 x $50) | $2,000 | $6,000 |
| **Total revenue** | **$132,000** | **$246,000** |

### Net

| Scenario | Revenue | Expenses | Net |
|----------|---------|----------|-----|
| Conservative | $132,000 | $131,500 | +$500 |
| Stretch | $246,000 | $131,500 | +$114,500 |

The first year targets break-even. Surplus funds roll into the following year's
event and community programs (travel grants, open-source bounties).

## Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Total attendees | 200+ | Registration system |
| Net Promoter Score (NPS) | > 60 | Post-event survey |
| Sponsor partners | 5+ | Signed sponsorship agreements |
| CFP submissions | 40+ | CFP platform |
| Talk video views (30 days) | 10,000+ | YouTube analytics |
| New GitHub stars (30 days post-event) | 500+ | GitHub insights |
| New contributors (60 days post-event) | 20+ | GitHub contributor graph |
| Social media impressions (event week) | 100,000+ | Platform analytics |
| Speaker diversity | 30%+ underrepresented groups | Speaker demographics |
| Post-event survey response rate | 50%+ | Survey platform |

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Low ticket sales | Medium | Aggressive early-bird pricing, free community tickets for contributors, partner promotions |
| Insufficient sponsorship | Medium | Start outreach 6 months early, offer flexible custom packages, leverage existing vendor relationships |
| Speaker cancellations | Low | Maintain a waitlist of backup speakers, have maintainers prepared with reserve talks |
| Venue/A/V failure | Low | Pre-event tech rehearsal, backup presentation laptop, offline copies of all slides |
| Low workshop attendance | Low | Cap registrations, require sign-up in advance, offer cloud dev environments to remove hardware barriers |

## Code of Conduct

ZerfooConf adopts the [Go Community Code of Conduct](https://go.dev/conduct).
All attendees, speakers, sponsors, and staff are expected to follow it. A
dedicated Code of Conduct team will be reachable on-site and by email throughout
the event. Violations will be addressed promptly and may result in removal from
the event without refund.

## Post-Event

- Publish all talk recordings within 2 weeks
- Share slide decks on the Zerfoo blog
- Write a recap blog post with photos, metrics, and highlights
- Send a detailed thank-you to sponsors with metrics and attendee feedback
- Conduct an internal retrospective for the organizing team
- Begin planning the next ZerfooConf within 4 weeks of the event
