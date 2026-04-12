# **Software Requirements Specification (Lite Version)**

## **1. Input/Output Schema**

### **1.1 Input Specification**
```yaml
File Format: .mat (HDF5, MATLAB v7.3+)
Required Fields:
  - lfpN: EEG signal data
    * Type: float32 array
    * Shape: [n_trials, 1000]
    * Units: μV
    * Sampling: 1000Hz (1 second per trial)

  - par/Age: Age label
    * Type: scalar integer
    * Range: [0, 47] (16 discrete values)

Optional Fields:
  - Other par/* metadata
  - channel_names: list of electrode names

File Naming Convention:
  - Recommended: {subject_id}_{session}_{condition}.mat
  - Example: sub001_ses1_rest.mat
```

### **1.2 Output Specification**
```yaml
Training Phase:
  - Model file: .pth (PyTorch weights)
  - Logs: training metrics, visualizations
  - Checkpoints: model snapshots

Inference Phase:
  - Basic:
    * predicted_class: integer [0-15]
    * age_label: integer (0,1,2,3,4,5,6,7,10,11,12,13,16,19,26,47)
    * confidence: float [0.0-1.0]

  - Extended (optional):
    * class_probabilities: probability distribution
    * attention_weights: temporal attention
    * feature_embeddings: deep features
```

---

## **2. API Specification**

### **2.1 Training API**
```python
def train_model(
    data_dir: str,
    filter_csv: Optional[str] = None,
    filter_type: str = "bandpass_individual",
    low_cutoff: float = 8.0,
    high_cutoff: float = 100.0,
    num_epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001
) -> Dict[str, Any]:
    """
    Train age classification model on EEG data.
  
    Returns:
        Dictionary containing model path, metrics, and visualizations
    """
```

### **2.2 Inference API**
```python
def predict_age(
    mat_file: Union[str, np.ndarray],
    model_path: str = "./data/1000-bandpass8-100Hz.pth",
    return_confidence: bool = True,
    return_attention: bool = False
) -> Dict[str, Any]:
    """
    Predict age from EEG signal.
  
    Returns:
        {
            'age_class': int,
            'age_label': int,
            'confidence': float,
            'probabilities': np.ndarray,  # optional
            'attention_map': np.ndarray   # optional
        }
    """
```

### **2.3 Data Processing API**
```python
def preprocess_eeg(
    signals: np.ndarray,
    filter_type: str = "bandpass_individual",
    sampling_rate: int = 1000,
    low_cutoff: float = 8.0,
    high_cutoff: float = 100.0
) -> np.ndarray:
    """
    Apply filtering and normalization to EEG signals.
    """

def load_mat_file(
    file_path: str,
    require_fields: List[str] = ["lfpN", "par/Age"]
) -> Dict[str, Any]:
    """
    Load and validate .mat file with required fields.
    """
```

---

## **3. Degradation Rules**

### **3.1 Timeout Rules**
| Scenario | Threshold | Action |
|----------|-----------|--------|
| Single trial inference | 500ms | Log warning, continue |
| Batch inference (64 trials) | 5s | Abort batch, log error |
| Full training (200 epochs) | 2h | Save checkpoint, resume later |
| Data loading (100 files) | 30s | Skip problematic files |

### **3.2 Uncertainty Zones**
1. **Low Confidence**: Max probability < 0.6 → Flag as "low_confidence"
2. **Boundary Cases**: Adjacent class probability difference < 0.3 → Flag as "boundary_case"
3. **Poor Quality**: SNR < 15dB or amplitude > ±200μV → Flag as "poor_signal"

### **3.3 Fallback Strategy**
```yaml
Level 1 (Mild):
  - Trigger: Single inference timeout / low confidence
  - Action: Use lightweight model (no Attention)
  - Performance drop: <10% accuracy

Level 2 (Moderate):
  - Trigger: Level 1 fails or memory overflow
  - Action: Return statistical features (band power, mean, variance)
  - Performance drop: <30% accuracy

Level 3 (Severe):
  - Trigger: System error or severe data corruption
  - Action: Return error code with remediation steps
  - Example: "ERROR_1001: Unsupported input format"
```

---

## **4. Non-Functional Requirements**

### **4.1 Performance Requirements**
| Metric | Target | Priority | Measurement Method |
|--------|--------|----------|-------------------|
| Accuracy | >85% (test set) | P0 | Stratified 15% hold-out |
| Inference Latency | P95 < 100ms/trial | P1 | time.perf_counter() |
| Training Time | <2h (200 epochs) | P1 | End-to-end timing |
| Memory Usage | <8GB GPU RAM | P1 | GPU monitoring |
| Model Size | <100MB | P2 | File size check |

### **4.2 Availability Requirements**
- **Reproducibility**: Fixed random seed (42), deterministic outputs
- **Maintainability**: Modular code, >70% test coverage, comprehensive docs
- **Recoverability**: Training checkpoint every 10 epochs

### **4.3 Cost Constraints**
| Resource | Constraint | Monitoring |
|----------|------------|------------|
| Compute Cost | <$3 per training (cloud equivalent) | GPU hours estimation |
| Storage | <10GB for models + data | Periodic cleanup |
| Energy | <2kWh per training | Power monitoring |

### **4.4 Data Ethics & Privacy**
- All EEG data must be anonymized
- Local processing only (no cloud upload)
- Research use only, with ethical approval

---

## **5. Acceptance Checklist**

### **5.1 Functional Requirements**
- [ ] **FR-1**: Correctly loads .mat files with lfpN and par/Age fields
- [ ] **FR-2**: Applies specified filter (bandpass 8-100Hz or highpass 2Hz)
- [ ] **FR-3**: Normalizes each trial independently (zero mean, unit variance)
- [ ] **FR-4**: Splits data with stratification (train/val/test = 70/15/15)
- [ ] **FR-5**: Trains CNN+Attention model for age classification
- [ ] **FR-6**: Implements early stopping (patience=20)
- [ ] **FR-7**: Generates Grad-CAM visualizations
- [ ] **FR-8**: Saves model checkpoints and training logs

### **5.2 Performance Requirements**
- [ ] **PR-1**: Test accuracy >85% (baseline: 87%)
- [ ] **PR-2**: Single trial inference <500ms on RTX 3090
- [ ] **PR-3**: Batch throughput >100 samples/second
- [ ] **PR-4**: Memory usage stable (<8GB peak)

### **5.3 Quality Requirements**
- [ ] **QR-1**: Code passes all automated tests (pytest)
- [ ] **QR-2**: No crashes on malformed inputs
- [ ] **QR-3**: Clear error messages for common failures
- [ ] **QR-4**: Complete documentation of API usage
- [ ] **QR-5**: Reproducible results with fixed seed

### **5.4 Data Requirements**
- [ ] **DR-1**: Validates input data format and structure
- [ ] **DR-2**: Checks for data leakage across splits
- [ ] **DR-3**: Handles missing/invalid age labels gracefully
- [ ] **DR-4**: Maintains data provenance (file names, trial indices)

### **5.5 Deployment Requirements**
- [ ] **DEP-1**: Clean environment setup (requirements.txt)
- [ ] **DEP-2**: GPU and CPU compatibility
- [ ] **DEP-3**: Model versioning and storage
- [ ] **DEP-4**: Logging and monitoring integration

