# Evaluation Plan - EEG Age Classifier

## 1. Evaluation Metrics
### 1.1 Primary Metrics
- **Accuracy**: Overall classification accuracy
- **Macro F1**: Average F1 across all age categories
- **Weighted F1**: F1 weighted by class frequency

### 1.2 Secondary Metrics
- **Per-class precision/recall**: For each age category
- **Confusion matrix analysis**: Identify systematic errors
- **Inference latency**: P95 < 20ms
- **Memory usage**: <100MB during inference

## 2. Data Slices (至少6个)
### 2.1 Age-based Slices
1. **Young children (0-5 years)**: Critical for developmental studies
2. **School age (6-12 years)**: Main population segment
3. **Adolescents (13-18 years)**: Transition period
4. **Young adults (19-30 years)**: Baseline performance
5. **Middle age (31-47 years)**: Upper limit of current data

### 2.2 Signal Quality Slices
6. **High SNR (>20dB)**: Clean signal performance
7. **Medium SNR (10-20dB)**: Typical real-world conditions
8. **Low SNR (<10dB)**: Challenging conditions

### 2.3 Temporal Slices
9. **First 500ms**: Early signal response
10. **Last 500ms**: Sustained signal patterns

## 3. Thresholds & Gates
### 3.1 Absolute Thresholds
| Metric | Threshold | Tolerance | Action on Fail |
|--------|-----------|-----------|----------------|
| Overall Accuracy | 0.85 | ±0.02 | Block deployment |
| Per-class Recall | 0.70 | ±0.05 | Warning, investigate |
| Inference Latency | 20ms | +5ms | Optimize model |
| Memory Usage | 100MB | +20MB | Model compression |

### 3.2 Regression Rules
- **Golden set accuracy**: No degradation > 2%
- **Critical slices**: No degradation > 5% on young children
- **Latency**: P95 cannot increase > 10%

## 4. Golden Set Requirements
### 4.1 Composition
- **Size**: 40 samples (20-50 range)
- **Distribution**: 2-3 samples per age category
- **Quality**: Manually verified, high SNR signals
- **Storage**: `data/golden_set.jsonl` with DVC versioning

### 4.2 Usage
- **Regression testing**: Before every model update
- **CI integration**: Automated evaluation gate
- **Manual review**: Monthly quality check

## 5. Evaluation Frequency
- **Pre-deployment**: Full evaluation on test set
- **Post-deployment**: Weekly monitoring on golden set
- **Quarterly**: Comprehensive re-evaluation
- **Trigger-based**: After data collection changes
