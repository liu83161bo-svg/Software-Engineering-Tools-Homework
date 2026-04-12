# **Data Contract**

## **1. Schema Contract**

### **1.1 Raw Data (.mat files)**
```yaml
File Structure:
  Format: HDF5 (.mat)
  Required Keys:
    - lfpN: [n_trials × 1000] float32 array
    - par/Age: scalar integer
  Optional Keys:
    - par/Session: string
    - par/SubjectID: string

Schema Validation:
  - lfpN must have exactly 1000 columns (timepoints)
  - Age must be integer between 0 and 100
  - No NaN or infinite values allowed
  - File size must be ≤ 100MB
```

### **1.2 Processed Data (CSV/JSONL)**
```yaml
CSV Format:
  Required Columns:
    - trial_id: int64 (unique)
    - file_name: string (pattern: *.mat)
    - trial_index: int64 (≥0)
    - age: int64 (0-100)
  Optional Columns:
    - subject_hash: string (anonymized)
    - recording_date: YYYY-MM-DD

JSONL Format:
  Each line must contain:
    - trial_id: integer
    - signal: [1000] float array
    - age: integer
    - metadata: object (optional)
```

### **1.3 Data Types & Constraints**
| Field | Type | Constraints | Default Value |
|-------|------|-------------|---------------|
| trial_id | int64 | Unique, auto-increment | N/A |
| file_name | string | Must end with .mat | Required |
| trial_index | int64 | 0 ≤ value ≤ 999 | 0 |
| age | int64 | 0 ≤ value ≤ 100 | Required |
| signal | float32[1000] | -500 ≤ values ≤ 500 | Normalized |
| subject_hash | string | SHA256, length=64 | hash("unknown") |
| quality_score | float | 0.0 ≤ value ≤ 1.0 | 1.0 |

---

## **2. Quality Contract**

### **2.1 Syntax Quality Rules**
| Rule ID | Rule | Severity | Action |
|---------|------|----------|--------|
| SYN-01 | File must be valid .mat (HDF5) format | Critical | Reject file |
| SYN-02 | Required fields present (lfpN, Age) | Critical | Reject file |
| SYN-03 | No corrupted data blocks | Critical | Reject file |
| SYN-04 | No NaN or infinite values | High | Mark as poor quality |
| SYN-05 | Age field not empty/null | High | Use default age (0) |

### **2.2 Structural Quality Rules**
| Rule ID | Rule | Severity | Action |
|---------|------|----------|--------|
| STR-01 | Signal length = 1000 samples | Critical | Reject trial |
| STR-02 | Trial indices unique within file | Medium | Auto-correct duplicates |
| STR-03 | No data leakage (subject-level) | Critical | Re-partition |
| STR-04 | Age consistency within subject | High | Use majority age |
| STR-05 | File naming convention followed | Low | Warning only |

### **2.3 Statistical Quality Rules**
| Rule ID | Rule | Threshold | Action |
|---------|------|-----------|--------|
| STA-01 | Signal mean within range | -200 ≤ μ ≤ 200 μV | Mark as outlier |
| STA-02 | Signal variance within range | 0 < σ² ≤ 10000 μV² | Mark as outlier |
| STA-03 | Age distribution balance | No class > 25% total | Apply sampling |
| STA-04 | SNR minimum | SNR ≥ 20 dB | Filter/reject |
| STA-05 | Temporal stationarity | Mean diff ≤ 50 μV | Flag for review |

---

## **3. Freshness Contract**

### **3.1 Update Schedule**
```yaml
Real-time Processing:
  - New data: Available within 24 hours of collection
  - Corrections: Applied within 48 hours of identification
  - Backfills: Completed within 7 days of schema change

Batch Processing:
  - Daily: Quality metrics calculation
  - Weekly: Statistical summary updates
  - Monthly: Full data re-validation

Version Updates:
  - Minor updates: Monthly (1st of each month)
  - Major updates: Quarterly (Q1: Jan, Q2: Apr, etc.)
  - Emergency patches: Within 72 hours of critical issue
```

### **3.2 Data Retention**
| Data Type | Retention Period | Archival Policy |
|-----------|-----------------|-----------------|
| Raw .mat files | 10 years | Compressed, encrypted |
| Processed CSV/JSONL | 5 years | Daily backups |
| Model training data | 3 years | Versioned snapshots |
| Logs & metrics | 2 years | Rotated monthly |
| Temporary files | 30 days | Auto-delete |

### **3.3 Change Management**
```yaml
Breaking Changes:
  - Notice period: 30 days
  - Migration path: Provided
  - Support: 6 months overlap

Schema Evolution:
  - Additive changes only
  - Backward compatibility required
  - Deprecation period: 12 months

Data Corrections:
  - Tracked in CHANGELOG.md
  - Version incremented
  - Notification to users
```

---

## **4. Service Level Agreements**

### **4.1 Availability SLA**
```yaml
Uptime:
  - Data access API: 99.5% monthly uptime
  - Batch processing: 95% success rate
  - File storage: 99.9% availability

Maintenance Windows:
  - Scheduled: Sundays 02:00-04:00 UTC
  - Emergency: 4 hours notice when possible
  - Duration: ≤ 2 hours per window

Performance:
  - Data retrieval: P95 < 500ms
  - File upload: P95 < 5s (100MB file)
  - Query response: P95 < 2s
```

### **4.2 Support SLA**
| Support Level | Response Time | Resolution Time | Availability |
|---------------|---------------|-----------------|--------------|
| Critical (P0) | 1 hour | 4 hours | 24/7 |
| High (P1) | 4 hours | 24 hours | Business hours |
| Medium (P2) | 1 business day | 3 business days | Business hours |
| Low (P3) | 3 business days | 7 business days | Business hours |

### **4.3 Quality SLA**
| Metric | Target | Measurement | Consequence |
|--------|--------|-------------|-------------|
| Data completeness | 99.9% | Missing values count | Data backfill |
| Data accuracy | 98% | Manual validation sample | Correction batch |
| Schema compliance | 100% | Automated validation | Reject non-compliant |
| Processing latency | ≤ 24h | E2E processing time | Priority processing |
| Error rate | < 1% | Failed jobs / total jobs | Root cause analysis |

---

## **5. Monitoring & Validation**

### **5.1 Automated Checks**
```python
# Example: Contract validation test
def validate_data_contract(data_frame):
    violations = []
  
    # Schema validation
    if 'age' not in data_frame.columns:
        violations.append("Missing required column: age")
  
    # Quality validation
    age_range = data_frame['age'].between(0, 100)
    if not age_range.all():
        violations.append(f"Age out of range: {data_frame[~age_range].shape[0]} rows")
  
    # Statistical validation
    if data_frame['age'].isnull().any():
        violations.append("Null values in age column")
  
    return violations
```

### **5.2 Metrics Dashboard**
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Schema compliance | 99.8% | 100% |  Warning |
| Data freshness | 99.5% | 99.9% |  Healthy |
| Quality score | 98.2% | 99.0% |  Warning |
| Processing time | 22h | 24h |  Healthy |
| User satisfaction | 4.5/5 | 4.5/5 |  Healthy |

### **5.3 Alert Rules**
```yaml
Critical Alerts:
  - Schema violation rate > 5%
  - Data loss > 1% of total
  - System downtime > 30 minutes

Warning Alerts:
  - Quality score drop > 5%
  - Processing delay > 6 hours
  - Storage utilization > 80%

Informational:
  - New data available
  - Weekly quality report
  - Monthly performance summary
```

---

## **6. Breach Handling**

### **6.1 Breach Classification**
| Level | Definition | Example | Response Time |
|-------|------------|---------|---------------|
| Level 1 | Critical breach | Data corruption, privacy leak | Immediate |
| Level 2 | Major breach | SLA missed by > 10% | 4 hours |
| Level 3 | Minor breach | Quality target missed | 24 hours |
| Level 4 | Informational | Performance degradation | 3 days |

### **6.2 Remediation Process**
```
1. Detection → 2. Classification → 3. Containment → 4. Investigation
     ↓               ↓               ↓               ↓
  Monitoring       Severity        Isolate       Root cause
    tools          assessment     affected       analysis
                                  data
                     ↓
                5. Correction → 6. Validation → 7. Communication
                     ↓               ↓               ↓
                Fix issue,     Test fixes,      Notify users,
                restore data   verify quality   update status
```

### **6.3 Compensation**
| Breach Type | Compensation | Conditions |
|-------------|--------------|------------|
| Data loss | Data recovery + 1 month free | Loss > 1% of total |
| SLA violation | Service credit | Uptime < 99.5% for month |
| Privacy breach | Legal compliance + notification | Any confirmed breach |
| Extended outage | Pro-rated refund + credit | > 4 hours continuous |

---

**Contract Version**: 1.0.0
**Effective Date**: 2026-01-01
**Review Cycle**: Quarterly
**Signatories**: Data Provider ([Your Institution]), Data Consumer (Research Team)
**Term**: 2 years (auto-renews unless terminated)

---
