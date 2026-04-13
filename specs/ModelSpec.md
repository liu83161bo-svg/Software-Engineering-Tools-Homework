# Model Specification - EEG Age Classifier

## 1. Baseline Model
### 1.1 Current Baseline
- **Model**: CNN with Attention Pooling (AgeClassifier)
- **Architecture**: 
  - 4 convolutional layers (64-128-256-512 channels)
  - Attention pooling over time dimension
  - 3 fully connected layers (512-256-128)
- **Input**: 1000Hz EEG signal, 1000 time points
- **Output**: Age classification (16 age categories)
- **Performance**: ~85% accuracy on validation set

### 1.2 Simple Baselines (For Comparison)
- **Rule-based baseline**: Majority class prediction
- **Statistical baseline**: Linear regression on signal features
- **Simple ML baseline**: Random Forest with extracted features

## 2. Applicability Limits
### 2.1 Domain Constraints
- **Input signal**: Must be 1000Hz EEG, bandpass filtered (8-100Hz)
- **Signal length**: Exactly 1000 time points
- **Age range**: 0-47 years (16 discrete categories)
- **Data source**: Compatible with .mat format from specified recording setup

### 2.2 Out-of-Distribution Detection
- Signals with SNR < 10dB should be flagged
- Age predictions beyond 0-100 years considered invalid
- Unusual signal patterns (epileptic spikes) should trigger fallback

## 3. Resource Envelope
### 3.1 Computational Requirements
- **Training**: ~30 minutes on GPU (NVIDIA RTX 3060)
- **Inference**: <10ms per sample on CPU, <2ms on GPU
- **Memory**: Model size ~15MB, inference requires <100MB RAM

### 3.2 Deployment Constraints
- **Minimum hardware**: CPU with AVX2 support
- **Maximum latency**: 50ms for real-time applications
- **Throughput**: >100 samples/second on single CPU core

## 4. Update Policy
### 4.1 Update Triggers
- **Performance degradation**: >5% accuracy drop on golden set
- **Data drift**: Statistical shift in input distribution
- **New age categories**: Adding age groups outside current range
- **Hardware changes**: Migration to new inference hardware

### 4.2 Update Process
1. **Staging**: Deploy to 10% of traffic for 24h
2. **Monitoring**: Track accuracy, latency, error distribution
3. **Rollback criteria**: Any degradation on critical slices
4. **Full deployment**: After 7 days of stable performance

## 5. Failure Modes & Fallbacks
### 5.1 Detected Failures
- High prediction uncertainty (>0.8 entropy)
- Out-of-distribution signals
- Hardware failure (GPU unavailable)

### 5.2 Fallback Strategies
1. **Primary**: Return to previous model version
2. **Secondary**: Rule-based age estimation
3. **Tertiary**: Flag for human review
