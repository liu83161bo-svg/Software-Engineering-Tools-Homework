# Software-Engineering-Tools-Homework

This repository is a dedicated project workspace for the Software-Engineering-Tools-Homework, focusing on the standardized engineering implementation of AI features—covering the full lifecycle from requirements specification to production deployment with rigorous engineering discipline.

## Core Directory Structure
Each directory has a well-defined, single responsibility to ensure clarity and maintainability for all engineers:
- `/specs`: All core specification documents (PRD, SRS, Data Spec, Eval Plan, etc.)—the single source of truth for project decisions and feature definitions.
- `/src`: AI feature core code, data preprocessing logic, and serving layer implementation—structured as an importable Python package with clear entry points.
- `/tests`: Automated test cases (unit, contract, data, and pipeline tests)—implemented with pytest and integrated into CI/CD pipelines.
- `/pipelines`: Model training and evaluation scripts—optimized for reproducibility with single-command execution (make train/eval).
- `/reports`: Auto-generated artifacts (test reports, data quality reports, model evaluation reports)—attached to every CI run for direct review.
- `/configs`: Project configuration files (thresholds, hyperparameters, environment settings)—**no hardcoding in source code** for all tunable parameters.

## Quick Start
### 1. Environment Setup
Clone the repository and install core dependencies (Python 3.10+ is required):
```bash
# Clone the repo
git clone <Your Repository HTTPS/SSH Link>
cd ai-system-engineering-project

# Install core dependencies
pip install -r requirements.txt

# Install test dependencies for CI/CD
pip install pytest pytest-cov ruff black mypy
```

### 2. Run the Project
Execute model training and evaluation with standardized pipeline scripts:
```bash
# Initialize project environment and preprocess data
python src/setup.py

# Train the model
python pipelines/train.py

# Evaluate model performance against spec thresholds
python pipelines/eval.py
```

### 3. Run Automated Tests
Validate all components with the test suite (enforced in CI on every push/PR):
```bash
# Run all tests and generate coverage report
pytest tests/ --cov=src/ --cov-report=xml --cov-report=term

# Run only unit tests
pytest tests/unit/ -v

# Run only data quality tests
pytest tests/data/ -v
```

### 4. Reproduce Experimental Results
Achieve deterministic results with versioned data, locked dependencies, and fixed random seeds:
```bash
# Follow the reproducibility guide
bash reproduce.sh
```
All steps for reproduction are documented in `reproduce.md`, including environment locking, data version references, and pipeline execution commands.

## Engineering Standards & Best Practices
This project adheres to production-grade AI engineering discipline—all changes must comply with the following rules:
1. **CI/CD Enforcement**: Every code commit and PR must pass all CI checks (linting, testing, data validation) with a **green status**.
2. **PR Review Requirement**: All modifications to specification documents and core code require peer review and approval before merging to the main branch.
3. **Version Control for All Artifacts**: Code (Git), data (DVC/lakeFS), models (MLflow), and environment dependencies (Poetry/uv/Docker) are fully versioned for reproducibility.
4. **Documentation as Code**: All specification documents are version-controlled in Markdown, with Mermaid diagrams for architecture/flow visualizations.
5. **Fail-Fast Principles**: Automated gates block pipeline execution and code merging for any violation of data quality, model performance, or coding standards.

## Tooling Stack
The project uses industry-standard tools for MLOps/LLMOps, aligned with the course curriculum:
- **Version Control**: Git (code), DVC (data), MLflow (models)
- **CI/CD**: GitHub Actions (automated testing, training, and validation)
- **Code Quality**: Ruff, Black, Mypy (linting and type checking)
- **Testing**: Pytest (unit/contract/pipeline tests), Great Expectations (data validation)
- **Monitoring & Observability**: Evidently AI (drift detection), Prometheus + Grafana (system/model monitoring)
- **Environment & Packaging**: Poetry/uv (dependency management), Docker (containerization)

## Project Ownership & Governance
- **Code Owners**: Defined in `.github/CODEOWNERS`—all PRs require approval from relevant owners.
- **Incident Response**: Documented in `specs/RiskSafety.md`—clear escalation paths and owners for production issues.
- **Change Management**: All specification and code changes are tracked via PRs with detailed commit messages and review comments.

For detailed feature requirements and operational guidelines, refer to the specification documents in the `/specs` directory.