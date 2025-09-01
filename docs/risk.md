# Risk Log

This document tracks potential risks to the project, their likelihood, impact, and mitigation plans.

| Risk ID | Description | Likelihood | Impact | Mitigation Plan | Status |
|---|---|---|---|---|---|
| R1 | Reference implementation for Gemma 3 is not available or is difficult to integrate. | Medium | High | Allocate time for research and reverse-engineering if necessary. Prepare a fallback plan with a different reference model. | Open |
| R2 | Numerai data format changes unexpectedly. | Low | Medium | The data contract and validation in the pipeline should catch this. Monitor Numerai announcements. | Open |
| R3 | Performance on GPU is significantly worse than expected, blocking the 30-day goal. | Low | High | The plan includes a performance pass. If major issues are found, we may need to descale the model or simplify the architecture for the MVP. | Open |
