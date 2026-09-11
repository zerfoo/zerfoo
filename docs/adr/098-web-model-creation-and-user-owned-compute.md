# ADR 098: Web model design and user-owned compute

Status: Accepted for implementation

Zerfoo's scope is AI creation, training, evaluation and inference. Its web
experience translates a user's objective into a portable project. A coding
agent operates Zerfoo locally and uses Kazi to verify lifecycle acceptance.

The hosted service may call a bounded design LLM and validate structured
proposals. It must never execute generated code, train visitor models, accept
arbitrary remote sources, or use the founder's DGX as visitor infrastructure.
Full datasets remain on user hardware. Model definitions are generated from
qualified recipes and validated before download. Unqualified requests remain
design briefs with explicit gaps. Distilled papers remain unreviewed until
their claims and component mappings receive semantic review.

Downloads preserve objective, dataset requirements, model definition, training
configuration, evidence and versioned agent instructions. Actual hardware and
data checks happen locally. A binary runner is not a trained model.

Website LLM usage requires server-held credentials, bounded messages and
outputs, abuse controls and a durable reservation before each dispatch against
a global ceiling. Provider failures do not refund reservations automatically.
The free local-agent handoff remains available without hosted LLM access.

zer.foo is the intended primary domain. Existing URLs redirect with paths
preserved once DNS, TLS and deployment are verified.
