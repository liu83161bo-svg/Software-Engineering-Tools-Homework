# **Acceptance_Criteria_Table.md**

## **Acceptance Criteria**

| ID | Scenario Type | Description | Metric | Threshold ± Tolerance | Measurement Method | Action on Fail |
|----|---------------|-------------|--------|----------------------|-------------------|----------------|
| **A. Normal Scenarios** |
| AC-01 | Normal-Standard | Standard quality EEG, clear age features | Accuracy | >85% ±2% | 15% hold-out test set | Adjust hyperparameters |
| AC-02 | Normal-Batch | Batch processing (64 trials) | Throughput | >100 samples/sec ±10% | Time per batch | Optimize data loading |
| AC-03 | Normal-Real-time | Single trial inference | End-to-end latency | <100ms ±20ms | Full pipeline timing | Simplify model, GPU optimization |
| **B. Edge Scenarios** |
| AC-04 | Edge-Age | Age near class boundaries (e.g., 12-13) | Boundary accuracy | >70% ±5% | Boundary sample subset | Augment training with boundary cases |
| AC-05 | Edge-Signal | Signal amplitude near saturation (±180μV) | Classification stability | CV <15% | Multiple inference runs | Enhance normalization |
| AC-06 | Edge-Duration | Slight length variation (950-1050 points) | Processing success rate | >95% | Auto-pad/truncate test | Improve preprocessing robustness |
| **C. Negative Scenarios** |
| AC-07 | Negative-Format | Non-.mat file input | Error detection rate | 100% | Provide .txt/.csv files | Clear error message, reject input |
| AC-08 | Negative-Missing | Missing required fields (lfpN or Age) | Graceful handling | 100% | Remove fields from test files | Structured error codes |
| AC-09 | Negative-Noise | High noise (SNR<10dB) | Performance degradation | Accuracy drop <25% | Add Gaussian noise (σ=50μV) | Enhance filtering, data augmentation |
| AC-10 | Negative-Memory | Large file processing (>1GB) | Memory safety | No crash, graceful exit | Generate large test files | Implement chunking, memory monitoring |
| **D. Statistical Scenarios** |
| AC-11 | Statistical-Imbalance | Class-imbalanced data | Minority class F1-score | >75% ±5% | Stratified evaluation | Use class-weighted loss |
| AC-12 | Statistical-Generalization | Leave-one-subject-out CV | Average accuracy | >80% ±3% | Subject-wise splits | Add data augmentation, regularization |
| AC-13 | Statistical-Consistency | Multiple runs with same input | Output consistency | 100% identical | Fixed seed, 10 repetitions | Identify and fix randomness sources |
| **E. System Scenarios** |
| AC-14 | System-Recovery | Training interruption recovery | Recovery success rate | >95% | Simulate interruption | Improve checkpoint mechanism |
| AC-15 | System-Extension | New age class addition (e.g., 48) | Incremental learning | Old class performance drop <5% | Freeze layers, fine-tune | Implement elastic architecture |
| AC-16 | System-Monitoring | Resource usage monitoring | Monitoring coverage | 100% key metrics | Check logs and monitoring | Expand monitoring metrics and alerts |

---

## **Measurement Details**

### **Accuracy (AC-01)**
```python
# Implementation
test_set = stratified_split(data, test_size=0.15, random_state=42)
predictions = model.predict(test_set)
accuracy = np.mean(predictions == true_labels)
assert accuracy > 0.85, "Accuracy below threshold"
```

### **Latency (AC-03)**
```python
import time
start = time.perf_counter()
for _ in range(100):  # Average over 100 inferences
    prediction = model.predict(single_trial)
end = time.perf_counter()
avg_latency = (end - start) / 100
assert avg_latency < 0.1, "Latency exceeds threshold"
```

### **Boundary Cases (AC-04)**
```python
# Boundary: predicted age within ±2 years of true age
boundary_samples = []
for sample in test_set:
    predicted_age = model.predict(sample)
    true_age = sample.label
    if abs(predicted_age - true_age) <= 2:
        boundary_samples.append(sample)
boundary_accuracy = calculate_accuracy(boundary_samples)
```

---

## **Failure Handling Protocol**

```
Test Failure → Log Details → Classify Failure → Corrective Action → Retest
     ↓              ↓              ↓              ↓              ↓
  Generate    Scenario ID,    Performance/     Adjust model/   Verify fix
   report     actual values,  Functional/      Modify code
              expected values System/Data
```
