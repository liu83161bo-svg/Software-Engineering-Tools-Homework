# Canary / Shadow Rollout Plan

## Overview
We use a **shadow-first, then canary, then gradual ramp-up** strategy to minimize blast radius.

## Phase 1: Shadow Mode
- **Duration**: 1 hour
- **Traffic**: New model receives copy of all requests but responses are discarded.
- **Metrics**: Compare outputs (predicted age) between shadow and production model.
  - If agreement rate < 95% on a sample of 1000 requests, pause and investigate.
- **Rollback Trigger**: Shadow logs show unexpected output distribution (e.g., ages outside [0,100]).

## Phase 2: Canary Release (5% traffic)
- **Duration**: 30 minutes
- **Traffic Split**: 5% of live requests routed to new model, 95% to current model.
- **Observation**: Monitor latency P95 (<30ms), error rate (<1%), cost per request (<0.01$).
- **Rollback Trigger**: Any SLO breach, or cost spike >20% compared to baseline.

## Phase 3: Gradual Ramp-Up
| Step | Traffic % | Observation Window | Checkpoints                  |
|------|-----------|-------------------|------------------------------|
| 1    | 5%        | 30 min            | SLO, drift signals, cost     |
| 2    | 20%       | 30 min            | Same as above                |
| 3    | 50%       | 30 min            | Same + monitor golden set    |
| 4    | 100%      | 1 hour            | Full production monitoring   |

## Phase 4: Full Production
- After 1 hour at 100%, mark release as stable.
- Previous model version is kept for 72 hours for fast rollback.

## Rollback Triggers (any of the following)
1. Latency P95 > 30ms for 5 consecutive minutes.
2. Error rate > 2% for any 1-minute window.
3. Output distribution shift detected (PSI > 0.1 compared to golden set).
4. Daily cost exceeds budget by 30% (monitored via CloudWatch or custom metric).
5. Human feedback (flagged by operator or automated monitoring).