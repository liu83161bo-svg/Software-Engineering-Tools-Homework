# Definition of Done (DoD)
This AI project is deemed **production-ready** only when all the following criteria are met, with every item enforced automatically through CI checks.

## 1. Specification Document Requirements
- All 6 specification documents in the /specs directory are fully completed and approved via PR review;
- Document versions are consistent with code and model versions, with no conflicts.

## 2. Code and Testing Requirements
- Code in the /src directory passes ruff/black/mypy linting checks (enforced by CI gate);
- Test coverage in the /tests directory is ≥80% (measured by pytest-cov, enforced by CI gate);
- All unit tests, contract tests, and pipeline tests are passed (enforced by pytest CI gate).

## 3. Data and Model Requirements
- Data passes validation checks from the Great Expectations suite (enforced by data CI gate);
- Model performance meets the thresholds defined in EvalPlan.md (enforced by eval_gate CI gate);
- The model is reproducible, with explicit associations to fixed dataset versions and environment dependencies.

## 4. Operations and Monitoring Requirements
- The /Monitoring.md document contains a complete monitoring plan, alert thresholds, and response procedures;
- Rollback steps in RiskSafety.md have been tested and assigned a clear owner;
- Model training and evaluation can be executed via single commands (make train/eval).

## 5. Full CI/CD Pipeline Requirements
- All CI checks (code, testing, data, model) pass with a green status;
- Generated test, data, and evaluation reports are reviewable directly without re-running the pipeline.