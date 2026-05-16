# Safety & Privacy Policy — EEG Age Classifier

## 1. Scope & Boundaries
This system classifies EEG signals into discrete age categories (0–47 years). It is intended for research and educational purposes only.

**Permitted Use:**

Academic research on EEG age classification

Internal model development and testing

Educational demonstrations with synthetic or de-identified data

**Prohibited Use:**

Clinical diagnosis or medical decision-making

Processing of real patient data without explicit IRB approval

Deployment in environments where a failed prediction could cause physical harm

Any use involving minors without parental consent (when real data is involved)

## 2. Data Privacy

**Minimization**: Only signal data and age labels are stored. No names, IDs, or contact information are retained.

**Anonymization**: All real data must be anonymized before ingestion. Subject hashes replace identifiable information.

**Retention**: Processed data is kept for a maximum of 90 days. Raw uploads are deleted immediately after processing.

**Access Control**: Only authorized team members with approved roles can access training data and model artifacts.

**Logging**: Logs contain request IDs and error codes only. No raw signals or personal data are logged.

## 3. Model Safety

**Refusal Behavior**: The model must reject inputs that do not match expected format (e.g., missing signal, wrong length, NaN values). Rejection returns `{"error": "invalid_input", "details": "..."}`.

**Uncertainty Handling**: When prediction confidence is below 0.4, the system returns a fallback response ("age uncertain") rather than a false confident guess.

**Output Constraints**: Predicted age must be within [0, 100]. Any output outside this range is considered a failure and triggers alert.

## 4. Ownership

**Product Owner**: [Name] – responsible for risk tolerance and use-case approval.

**Security Owner**: [Name] – responsible for policy enforcement and threat model updates.

**Engineering Owner**: [Name] – responsible for implementation and test coverage.

## 5. Compliance

This policy is version-controlled and reviewed quarterly.

All changes require PR approval from at least two owners.

Audit logs of model promotions and access events are retained for 1 year.