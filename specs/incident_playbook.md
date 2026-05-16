# Incident Playbook – Safety Edition

## Incident Type 1: Data Exposure (EEG Signal Leak)
**Scenario**: A log file containing raw EEG signals is accidentally committed to a public GitHub repository.

**Roles**:

Lead: Security Owner

Engineering: Author of the commit

Communication: Product Owner

**Triage Steps**:
1. **Immediate** (within 5 min):
   
2. Force-push to remove the commit from history (use `git filter-branch` or BFG).
   
3. Rotate any access keys or tokens that were in the same commit.
   
4. Notify Security Owner and trigger privacy incident process.
2. **Investigate** (within 60 min):
   
3. Determine which file(s) were exposed and for how long.
   
4. Check if any external fork or clone exists.
   
5. Identify whether PII (e.g., subject hashes that could be re-identified) was leaked.
3. **Resolution**:
   
4. If no PII was involved: document and improve CI secret scanning (gitleaks pre-commit hook).
   
5. If PII was involved: follow legal/IRB notification requirements.

**Postmortem**:

Add `.gitignore` rules for data files.

Enable branch protection to prevent direct pushes to main.

Review access controls on the repository.

---

## Incident Type 2: Tool Misuse (Unauthorized Patient Data Retrieval)
**Scenario**: A user or external script calls `get_patient_metadata` with many requests, attempting to scrape patient data despite the tool requiring confirmation.

**Roles**:

Lead: Engineering Owner

Security: Security Owner

Monitoring: On-call Engineer

**Triage Steps**:
1. **Immediate** (within 5 min):
   
2. Block the offending IP or user token via API gateway.
   
3. Disable the tool temporarily if abuse is widespread.
   
4. Review confirmation logs to see if any unauthorized access succeeded.
2. **Investigate** (within 30 min):
   
3. Analyze tool call patterns (frequency, parameters, source).
   
4. Check if any data was actually returned without confirmation.
3. **Resolution**:
   
4. If no data leaked: prevent recurrence by adding rate limiting and anomaly detection.
   
5. If data leaked: notify affected patients per policy, review access control lists.

**Postmortem**:

Add rate limiting to all write-tools.

Implement anomaly detection for tool call frequency.

Update tool contract to require explicit user confirmation even for authenticated sessions.

---

## Incident Type 3: Model Safety Regression (Age Classification Degradation)
**Scenario**: After a model update, the accuracy on the golden set drops by 5% and the refusal rate for adversarial inputs falls below 90%.

**Roles**:

Lead: ML Engineer

Validation: Product Owner

Operations: Engineering Owner

**Triage Steps**:
1. **Immediate** (within 5 min):
   
2. Rollback to the previous model version using the CI rollback procedure.
   
3. Trigger a full evaluation gate on the new model to confirm regression.
   
4. Notify all dependent services of the rollback.
2. **Investigate** (within 60 min):
   
3. Compare performance on each age slice and each adversarial prompt.
   
4. Identify which change (model weights, preprocessing, thresholds) caused the regression.
3. **Resolution**:
   
4. Fix the regression (retrain, adjust thresholds, or revert the offending change).
   
5. Add a new safety test that covers the specific failure mode.

**Postmortem**:

Expand the golden set and red-team suite to cover the detected weakness.

Tighten the evaluation gate threshold to require ≥95% refusal rate on adversarial inputs.

Add a required human review step before promoting any model with changes to safety-related components.