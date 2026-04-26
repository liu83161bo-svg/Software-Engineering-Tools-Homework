# Incident Playbook

## Incident Type 1: Data Incident (Schema Violation / Missing Features)
**Symptoms**:

CI data check fails (syntax/structural check).

Production logs show many “NaN” or “unknown” values.

Monitoring shows PSI > 0.1 on input distribution.

**Triage Steps**:
1. **Immediate** (within 5 min):
   
2. Check if upstream source changed (contact data team).
   
3. Pause any active canary; keep current model serving.
   
4. If production is affected, consider rolling back to previous data version (use DVC checkout).
2. **Investigate** (within 30 min):
   
3. Identify which field(s) caused violation.
   
4. Verify if it’s a transient issue or permanent schema change.
3. **Resolution**:
   
4. If temporary: wait for fix from upstream, re-run data check.
   
5. If permanent: update DataContract.md, re-validate golden set, release updated model.

**Rollback Procedure**:

`dvc checkout data/sample_eeg_data.csv.dvc` (previous version).

Re-run CI to verify; if passes, promote old data version.

**Postmortem**:

Update data contract tests to catch this scenario.

Add alert for upstream schema changes.

---

## Incident Type 2: Model Incident (Quality Proxy Degradation)
**Symptoms**:

Golden set accuracy drops below 0.80.

Monitoring shows prediction entropy > 0.8.

Canary shows agreement rate < 95% with production model.

**Triage Steps**:
1. **Immediate** (within 5 min):
   
2. If canary is active, revert to 100% of old model.
   
3. If full production, rollback to previous model version.
   
4. Notify ML team via incident channel.
2. **Investigate** (within 60 min):
   
3. Compare metrics between old and new model on golden set.
   
4. Evaluate per-age slice performance – look for specific regressions.
   
5. Check if evaluation data distribution differs from training.
3. **Resolution**:
   
4. If regression due to data shift: collect new samples, retrain.
   
5. If regression due to model change: revert to previous model, fix training pipeline.

**Rollback Procedure**:

`dvc checkout data/1000-bandpass8-100Hz.pth.dvc` (or use MLflow model registry to restore stage).

Redeploy with CI: `make eval-gate && make deploy`.

**Postmortem**:

Add slice-specific gate to CI.

Improve golden set coverage for degraded age groups.

---

## Incident Type 3: Infrastructure Incident (Latency SLO Breach)
**Symptoms**:

Monitoring shows P95 latency > 30ms for 5 consecutive minutes.

Error rate increases (>3%).

Service may be degraded.

**Triage Steps**:
1. **Immediate** (within 2 min):
   
2. Check if CPU/memory is saturated.
   
3. If horizontal scaling is enabled, check auto-scaler logs.
   
4. If not scaling, manually scale up or switch to fallback endpoint.
   
5. If latency persists, issue a rollback to previous stable model.
2. **Investigate** (within 30 min):
   
3. Profile inference code – is the model slower due to weight changes?
   
4. Check for memory leak or concurrent request spike.
3. **Resolution**:
   
4. If model is slower: optimize model (quantization, ONNX) or use smaller architecture.
   
5. If infra issue: adjust autoscaling config, add caching.

**Rollback Procedure**:

Deploy previous model version using CI pipeline: `make rollback` (which restores previous artifact and re-runs CI).

Validate after rollback: latency recovers.

**Postmortem**:

Add latency gate to CI for each model version.

Introduce load testing in staging environment.