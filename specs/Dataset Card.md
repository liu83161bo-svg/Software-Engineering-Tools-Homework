# **Dataset Card**

## **1. Dataset Description**

### **Basic Information**
- **Dataset Name**: EEG Age Classification Dataset (EACD)
- **Version**: 1.0.0
- **Release Date**: 2026-01-01
- **Dataset Type**: EEG time-series recordings with age labels
- **Modality**: Single-channel EEG signals
- **Total Size**: ~50,000 trials (~5GB uncompressed)
- **Format**: .mat (HDF5) files with standardized structure

### **Purpose**
This dataset is designed for developing and evaluating machine learning models that can predict chronological age from EEG brain activity patterns. The primary research focus is on understanding neurodevelopmental trajectories through electrophysiological signatures.

### **Curation Rationale**
EEG signals exhibit characteristic changes across the lifespan, reflecting brain maturation and aging processes. This dataset systematically captures these changes across a 0-47 age range to enable age prediction models for both research and potential clinical applications.

---

## **2. Intended Use**

### **Primary Use Cases**
- **Research**: Study brain development and aging patterns using machine learning
- **Model Development**: Train and validate age classification models
- **Benchmarking**: Compare different feature extraction and classification methods
- **Educational**: Teaching signal processing and neural decoding techniques

### **Permitted Applications**
1. Academic research in neuroscience and machine learning
2. Development of age-related biomarkers from EEG
3. Methodological comparisons in signal classification
4. Validation of neurodevelopmental hypotheses

### **Expected Users**
- Neuroscience researchers
- Machine learning engineers in healthcare
- Computational psychiatry/neurology teams
- Graduate students in related fields

---

## **3. Non-Use Cases**

### **Prohibited Applications**
 **Clinical Diagnostics**: Not validated for clinical use; insufficient medical context

 **Individual Identification**: Cannot be used for subject re-identification

 **Commercial Products**: Requires separate licensing for commercial applications

 **Discriminatory Practices**: Must not be used to discriminate based on age or health status

 **Performance Evaluation**: Not for employment, insurance, or legal decision-making

### **Technical Limitations**
- Not suitable for real-time applications without additional validation
- Cannot extrapolate beyond 0-47 age range without additional data
- Single-channel limitation prevents spatial pattern analysis
- Controlled lab conditions may not generalize to real-world settings

---

## **4. Dataset Composition**

### **Data Structure**
```
eeg_age_dataset_v1.0.0/
├── raw_data/
│   ├── *.mat                    # ~500 HDF5 files
│   └── mTable.csv               # Quality filtering index
├── processed/
│   ├── train/                   # Training set (70%)
│   ├── val/                     # Validation set (15%)
│   └── test/                    # Test set (15%)
└── metadata/
    ├── subjects.csv             # Anonymous subject information
    └── recording_log.csv        # Session metadata
```

### **Statistical Summary**
| Attribute | Value | Notes |
|-----------|-------|-------|
| **Total Trials** | 50,000 | Across all subjects |
| **Unique Subjects** | ~500 | Anonymous identifiers |
| **Age Range** | 0-47 years | 16 distinct values |
| **Trial Length** | 1000 samples | 1 second at 1000Hz |
| **Sampling Rate** | 1000 Hz | Fixed across all recordings |
| **File Format** | .mat (HDF5) | MATLAB compatible |
| **Data Split** | 70/15/15 | Stratified by age |

### **Data Examples**
```python
# Example data structure in Python
import h5py
file = h5py.File('example.mat', 'r')
signal = file['lfpN'][:]        # Shape: [n_trials, 1000]
age = file['par/Age'][()]       # Scalar age for all trials in file
```

---

## **5. Labeling Process**

### **Age Label Collection**
- **Method**: Self-report verified by researcher interview
- **Verification**: Cross-referenced with study enrollment records
- **Precision**: Exact integer year (no fractional years)
- **Completeness**: 100% of trials have age labels

### **Quality Control Labels**
- **Manual Annotation**: Expert review of signal quality
- **Criteria**:
  - Signal-to-noise ratio > 20dB
  - No visible artifacts or saturation
  - Complete 1000-sample recording
- **Rejection Rate**: 15% of raw data excluded

### **Label Consistency**
- **Intra-subject**: All trials from same subject have identical age
- **Inter-rater Reliability**: 98% agreement on quality labels (2 raters)
- **Temporal Stability**: Age labels verified at time of each recording session

---

## **6. Known Limitations**

### **Data Collection Limitations**
1. **Age Distribution Bias**: Over-representation of 6-19 age range (55% of data)
2. **Health Status**: Only includes healthy participants without neurological conditions
3. **Geographic Homogeneity**: Single institution, single cultural context
4. **Recording Uniformity**: All data collected under controlled lab conditions
5. **Temporal Resolution**: 1-second segments may miss longer-term brain dynamics

### **Technical Limitations**
1. **Single Channel**: Lacks spatial information for brain region analysis
2. **Fixed Length**: 1-second segments may truncate longer cognitive processes
3. **Sampling Rate**: 1000Hz may not capture ultra-high frequency components
4. **Preprocessing**: Basic filtering only; may retain some artifacts

### **Ethical Limitations**
1. **Informed Consent**: Historical data; modern consent standards may not apply
2. **Privacy**: De-identified but potential for linkage attacks exists
3. **Representation**: Limited demographic diversity (age, ethnicity, socioeconomic)
4. **Generalizability**: May not represent clinical populations or real-world variability

---

## **7. Versioning**

### **Version History**
| Version | Date | Changes | Compatibility |
|---------|------|---------|---------------|
| v1.0.0 | 2026-01-01 | Initial release | N/A |
| Planned v1.1.0 | 2026-06-01 | Add multi-channel recordings | Backward compatible |
| Planned v2.0.0 | 2027-01-01 | Expand age range to 0-80 | Schema changes expected |

### **Update Policy**
- **Major Versions**: Annual releases with significant additions
- **Minor Versions**: Quarterly updates with quality improvements
- **Patch Versions**: Monthly bug fixes and metadata corrections

### **Deprecation Policy**
- Each major version supported for 2 years
- Advance notice of 6 months before breaking changes
- Migration scripts provided for data format changes

### **Citation Requirement**
```bibtex
@dataset{eeg_age_dataset_2026,
  title = {EEG Age Classification Dataset v1.0.0},
  author = {{Your Institution}},
  year = {2026},
  version = {1.0.0},
  url = {[Internal Repository URL]}
}
```

---

## **Maintenance**

### **Contact Information**
- **Curator**: Neuroscience Research Group, [Your Institution]
- **Technical Contact**: [Your Name/Email]
- **Ethics Contact**: Institutional Review Board, [Your Institution]

### **Support Timeline**
- **Active Support**: 2026-2028
- **Security Updates**: Until 2030
- **Data Availability**: Minimum 10 years from publication

### **Feedback Mechanism**
- GitHub Issues for technical problems
- Quarterly review meetings for dataset improvements
- Annual user survey for feature requests

---

**Dataset Status**: Active
**Last Updated**: 2026-01-01
**Next Review**: 2026-06-01
**Access Level**: Internal Research Use Only

---

## **Appendix: Sample Dataset**

A 50-trial sample dataset is available in `data/sample_eeg_dataset.csv` with the following structure:
- `trial_id`: Unique identifier (0-49)
- `signal_0` to `signal_999`: 1000 EEG timepoints (simulated data)
- `age`: Integer age label (0-47)
- `split`: 'train'/'val'/'test' assignment

*Note: The sample dataset contains simulated values that maintain the statistical properties of the full dataset while protecting subject privacy.*