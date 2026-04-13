# Evaluation Specification Document

## 1. Metric Selection

### Primary Metrics
- **Accuracy**: Overall classification correctness, main success metric
- **Average Recall**: Mean recall across all age categories
- **F1 Score**: Harmonic mean of precision and recall

### Secondary Metrics
- **Confusion Matrix**: Identify systematic error patterns
- **Inference Latency**: P95 < 20ms
- **Memory Usage**: < 100MB

### Business Metrics
- **Critical Age Detection Rate**: Recall for young pups (0-7 days)
- **Model Stability**: Variance across consecutive test runs

## 2. Data Slices & Segments

### Age-based Slices (Experimental Days)
1. **Neonatal Period (0-3 days)**: Critical developmental phase after birth
2. **Early Pup Stage (4-7 days)**: Rapid neural system development
3. **Mid Pup Stage (8-13 days)**: Behavioral pattern formation
4. **Late Pup Stage (14-19 days)**: Approaching adult characteristics
5. **Early Adult (20-26 days)**: Onset of sexual maturity
6. **Adult Stage (27-47 days)**: Full maturity and stability

### Experimental Condition Slices
7. **High SNR Group**: Signal quality > 0.8, clean recording environment
8. **Medium SNR Group**: Signal quality 0.5-0.8, typical experimental conditions
9. **Low SNR Group**: Signal quality < 0.5, challenging recordings

### Temporal Slices
10. **Early Recording (0-400ms)**: Initial stimulus response
11. **Mid Recording (400-800ms)**: Stable signal phase
12. **Late Recording (800-1000ms)**: Response termination phase

## 3. Release Gates

### Hard Gates (Must Pass)
| Metric | Threshold | Tolerance | Action on Fail |
|--------|-----------|-----------|----------------|
| Overall Accuracy | 80% | ±2% | Block deployment |
| Young Pup Recall | 70% | ±5% | Block deployment |
| Inference Latency | 20ms | +5ms | Optimize model |
| Memory Usage | 100MB | +20MB | Compress model |

### Soft Gates (Warning Level)
| Metric | Warning Threshold | Action |
|--------|-------------------|--------|
| Single Age Recall | < 60% | Investigate data balance |
| Model Size | > 20MB | Consider lightweight alternatives |
| Training Time | > 30 minutes | Optimize training pipeline |

## 4. Regression Testing

### Golden Dataset Requirements
- **Size**: 40 samples (20-50 range)
- **Distribution**: Minimum 2 samples per experimental day
- **Quality**: Manually verified, high signal quality
- **Versioning**: DVC managed, Git commit linked

### Regression Rules
1. **Accuracy Regression**: New model cannot be >2% worse than baseline
2. **Critical Age Protection**: Young pups (0-7 days) cannot degrade >5%
3. **Performance Preservation**: Inference latency cannot increase >10%

### Testing Frequency
- **Every Model Update**: Complete regression test
- **Weekly**: Random sampling from golden set
- **After Data Updates**: Re-evaluate all gates

## 5. Reporting Requirements

### Automated Report
```json
{
  "report_type": "Model Evaluation",
  "timestamp": "2026-01-01T10:30:00Z",
  "model_version": "v1.0.0",
  "dataset_version": "golden_v1",
  "metrics": {
    "accuracy": 0.85,
    "avg_recall": 0.78,
    "latency_p95": 15.2
  },
  "threshold_status": {
    "passed": 4,
    "failed": 0,
    "warnings": 1
  },
  "regression_status": "pass"
}
```

### Manual Report Template
```
Model Evaluation Report
=====================
Evaluation Date: [Date]
Model Version: [Version]
Dataset Version: [Dataset Version]

1. Performance Summary
   - Accuracy: X% (Target: Y%)
   - Average Recall: X% (Target: Y%)
   - Inference Latency: Xms (Target: Yms)

2. Key Findings
   - Strongest Age Groups: [List best performing ages]
   - Need Improvement Groups: [List worst performing ages]
   - Systematic Errors: [If identified]

3. Regression Status
   - Vs Baseline: [Improvement/Degradation] of X%
   - Gate Status: [Pass/Fail]

4. Recommendations
   - Deployment Recommendation: [Deploy/Optimize/Block]
   - Optimization Directions: [Specific suggestions]
```

### CI Integration Reports
- **Location**: `reports/evaluation_report.md`
- **Format**: Markdown + JSON metadata
- **Frequency**: Generated on each CI run
- **Storage**: Git versioned, MLflow logged

---
