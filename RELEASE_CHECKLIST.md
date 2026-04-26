# Release Checklist — EEG Age Classifier

## 1. Model Validation
**Eval Gate Pass**: `make eval-gate` passes (accuracy ≥ 0.80, avg recall ≥ 0.70 on golden set).

**Regression Check**: No degradation > 2% on golden set compared to previous version.

**Slice Performance**: Per-age recall ≥ 0.60 for all age groups; if any slice fails, document and approve exception.

**Latency & Resource**: Inference latency P95 < 20ms on target hardware; memory < 100MB.

## 2. Data Validation

**Data Contract Pass**: `make data-check` passes (syntax, structure, statistical checks).

**Golden Set Integrity**: Golden set is versioned with DVC and matches expected distribution.

**No Data Leakage**: Train/val/test split respects subject boundaries (confirmed by CI).

## 3. Infrastructure & Serving

**Service Health**: Canary test with 5% traffic runs for 15 minutes without SLO breach (latency P95 < 30ms, error rate < 1%).

**Rollback Ready**: Previous model version is staged and can be deployed within 5 minutes.

**Feature Flags**: All behavioral toggles are versioned in `configs/` and reviewed.

## 4. Security & Compliance

**Secrets Management**: No hardcoded secrets in code or config; use CI secrets.

**Logging Policy**: No PII logged; logging level is set to `INFO` in production.

**Access Control**: Model artifact and golden set are readable only by authorized roles.

## 5. Documentation & Approval

**Release Notes**: Describe what changed (model weights, thresholds, preprocessing) and why.

**Incident Playbook**: All types (data, model, infra) are updated with current contacts.

**Owner Approval**: At least one team member (non-author) has reviewed and signed off.

## 6. Observation Window After Release

**Shadow Mode**: New model runs in shadow for 1 hour, logs compared with production.

**Gradual Ramp-Up**: Traffic increases: 5% → 20% → 50% → 100% with 30-minute observation each step.

**Kill Switch**: If any SLO is violated during ramp-up, revert immediately.