# ADR 052: Online Learning Safety Framework

## Status
Accepted

## Date
2026-03-17

## Context
Wolf's trading models must continuously adapt to market regime changes to maintain
signal quality. Static models trained on historical data degrade as market dynamics
shift (volatility regimes, correlation breakdowns, macro events). Online learning
(continual fine-tuning from live trading data) addresses this but introduces severe
risks: catastrophic forgetting, data leakage through look-ahead bias, gradient
explosion from adversarial market data, and runaway model drift leading to trading
losses.

No major AI lab has shipped production-grade online learning as of 2025; this is
an unsolved research problem. The framework must be conservative.

## Decision
Implement a gated online learning pipeline in training/online/ with mandatory safety
controls:

Update Trigger (training/online/trigger.go):
- Minimum time between updates: 24 hours (prevents overfitting to single-day noise)
- Data requirement: minimum 500 new labeled examples before triggering update
- Update blocked if current model is in active trading positions (checked via Wolf API)

Incremental Fine-Tuning (training/online/incremental.go):
- LoRA-only updates (never full fine-tuning in production)
- Maximum 100 gradient steps per online update cycle
- Learning rate: 1/10 of initial fine-tuning LR (prevents catastrophic forgetting)
- Gradient clipping at norm 0.5 (aggressive clipping for stability)

Safety Validators (training/online/validator.go):
- Perplexity gate: new model perplexity on held-out validation set must be within
  5% of champion model perplexity before promotion
- Output distribution gate: KL divergence between champion and challenger output
  distributions must be under 0.1 on validation inputs
- Drawdown gate: Champion-challenger shadow run for 48 hours; challenger must not
  underperform champion by more than 2% on paper trades

Rollback (training/online/rollback.go):
- Every model version saved with full LoRA adapter weights + base model hash
- Automatic rollback if: (a) validator fails after promotion, (b) live Sharpe
  degrades more than 0.2 vs rolling 30-day baseline, (c) Wolf risk system raises
  alert
- Rollback completes in under 30 seconds (adapter swap, not full model reload)

Audit Log (training/online/audit.go):
- Every update cycle logged: trigger time, data window, gradient norm, validator
  scores, promotion decision
- Immutable append-only log in Wolf's data store

## Consequences
Positive:
- Enables Wolf's models to adapt to regime changes without manual intervention
- Conservative safety gates prevent catastrophic failures from bad gradient updates
- Fast rollback (30 seconds) limits maximum exposure to a drifted model

Negative:
- 24-hour minimum update interval means models cannot react to intraday regime shifts
  faster than once per day
- Shadow run requirement adds 48 hours to model promotion cycle; not suitable for
  rapid adaptation during acute market events
- Online learning from live P&L labels introduces survivorship bias if losing
  trades are censored
- Implementation complexity is very high; recommend extensive simulation testing
  before live deployment
