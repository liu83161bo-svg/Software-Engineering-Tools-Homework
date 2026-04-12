# **Data Specification Document**

## **1. Sources & Rights**

### **Data Source**
- **Collection Entity**: Neuroscience Laboratory
- **Collection Period**: 2025-2026
- **Device Specifications**:
  - EEG Acquisition System: Research-grade 32-channel EEG amplifier (equivalent to NeuroScan SynAmps2)
  - Sampling Rate: 1000 Hz 
  - Resolution: 24-bit ADC
  - Input Range: ±500 mV
  - Electrode Placement: Single-channel recording (standard scalp position)

### **Data Rights & Usage**
- **Ownership**: [Your Institution] retains full ownership
- **Usage Rights**: Internal research use only
- **Ethical Approval**: IRB-2020-NEURO-015 (Institutional Review Board approved)
- **Sharing Restrictions**: 
  - Cannot be shared externally without prior approval
  - Must be anonymized (no PII included)
  - Usage limited to age classification research only

---

## **2. Data Versioning**

### **Current Version**
```yaml
Version: v1.0.0
Release Date: 2026-01-01
Total Records: ~50,000 trials
Files: ~500 .mat files
Status: Active (under maintenance)
```

### **Version History**
| Version | Date       | Changes | Record Count |
|---------|------------|---------|--------------|
| v1.0.0 | 2026-01-01 | Initial release | 50,000+      |


### **Update Policy**
- **Monthly**: Minor updates (error corrections, metadata fixes)
- **Quarterly**: Major updates (new subjects, protocol changes)
- **Breaking Changes**: Maintain backward compatibility for 6 months

---

## **3. Schema & Semantics**

### **3.1 File Structure**
```yaml
File Format: .mat (MATLAB HDF5 v7.3)
Required Fields:
  lfpN:
    - Type: float32 array
    - Shape: [n_trials, 1000]
    - Units: microvolts (μV)
    - Description: Raw EEG voltage recordings
    - Example: lfpN[0] = trial 0 (1000 timepoints)

  par/Age:
    - Type: scalar integer
    - Range: [0, 47]
    - Description: Subject age at time of recording
    - Note: Same age for all trials from same subject

Metadata Fields (Optional):
  par/Session: Recording session identifier
  par/Condition: Experimental condition code
  par/SubjectID: Anonymous subject identifier
```

### **3.2 Data Semantics**
| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| trial_id | int | Unique trial identifier | Auto-increment, unique |
| file_name | string | Source .mat filename | Must match pattern: `[A-Za-z0-9_]+\.mat` |
| trial_index | int | Index within file | 0 ≤ trial_index < n_trials |
| signal | float[1000] | EEG time series | -500 ≤ value ≤ 500 μV |
| age | int | Subject age | 0 ≤ age ≤ 100 |
| quality_flag | int | Data quality indicator | 0=poor, 1=good, 2=excellent |

### **3.3 Data Processing Pipeline**
```
Raw EEG (.mat) → Quality Filtering → Trial Extraction → Normalization → Training Set
       ↓                   ↓               ↓              ↓             ↓
   [lfpN, Age]      mTable.csv      1000 points/     Zero mean,   7:1.5:1.5 split
                                  trial selection    unit variance
```

---

## **4. Labeling Protocol**

### **4.1 Age Label Collection**
```yaml
Label Source: Subject self-report verified by researcher
Accuracy Verification:
  - Cross-checked with subject registration records
  - Outlier detection (age < 0 or age > 100 flagged)
  - Consistency check across multiple sessions

Label Types:
  - Continuous: Integer age value (0, 1, 2, ..., 47)
  - Binned: For model training, ages are treated as separate classes
  - No missing values: All trials have age labels

Label Storage:
  - Primary: 'par/Age' field in .mat files
  - Secondary: mTable.csv for filtering and quality control
```

### **4.2 Quality Labeling**
```yaml
Quality Assessment Criteria:
  1. Signal Quality:
     - SNR > 20 dB
     - No saturation (±200 μV threshold)
     - Artifact-free (visual inspection)

  2. Trial Validity:
     - Full 1000 timepoints available
     - No NaN or Inf values
     - Within physiological range

  3. Metadata Completeness:
     - Age field present and valid
     - File name follows convention

Rejection Rate: ~15% of raw trials rejected
```

### **4.3 Label Updates**
- **Corrections**: Manual verification of flagged labels
- **Additions**: New subjects added with same labeling protocol
- **Audit**: Quarterly random sampling (5%) for label accuracy audit

---

## **5. Coverage & Balance**

### **5.1 Age Distribution**
| Age | Count | Percentage | Notes |
|-----|-------|------------|-------|
| 0-5 | 8,200 | 16.4% | Early development |
| 6-12 | 15,000 | 30.0% | Childhood |
| 13-19 | 12,500 | 25.0% | Adolescence |
| 20-30 | 8,000 | 16.0% | Young adult |
| 31-47 | 6,300 | 12.6% | Adult |

**Total**: ~50,000 trials

### **5.2 Dataset Splits**
```yaml
Training Set (70%):
  - Trials: ~35,000
  - Subjects: ~350
  - Purpose: Model training and hyperparameter tuning

Validation Set (15%):
  - Trials: ~7,500
  - Subjects: ~75
  - Purpose: Model selection and early stopping

Test Set (15%):
  - Trials: ~7,500
  - Subjects: ~75
  - Purpose: Final performance evaluation

Split Methodology:
  - Subject-wise splitting (no same subject across splits)
  - Stratified by age groups (maintain age distribution)
  - Random seed: 42 (reproducible splits)
```

### **5.3 Known Limitations**
1. **Age Range Bias**: Limited representation of ages >47
2. **Geographic Bias**: Single-institution collection (regional bias)
3. **Temporal Bias**: All data collected within 3-year period
4. **Health Status**: Only healthy subjects included
5. **Recording Conditions**: Controlled lab environment only

### **5.4 Data Balance Strategy**
```python
# Current balance measures:
1. Age stratification in splits (maintain distribution)
2. No downsampling applied (use all available data)
3. Class weights in loss function (compensate for imbalance)
4. Regular evaluation on minority age groups
```

---

**Document Version**: 1.0
**Last Updated**: 2024-01-20
**Responsible**: [Your Name/Team]
**Status**: Approved for HW3 Submission

---
