import pandas as pd
from scipy import signal
import numpy as np
import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
import sys

"""
main()
├── load_data_from_files()
│   └── apply_bandpass_filter() or apply_highpass_filter()
├── preprocess_data()
├── LFPDataset()
├── save_detailed_test_info()
├── DataLoader()
├── AgeClassifier()
├── train_model()
│   ├── model.train()
│   ├── model.eval()
│   └── torch.save()
├── model.load_state_dict()
├── test_model_with_gradcam()
│   ├── GradCAM() 
│   ├── model.eval()
│   ├── grad_cam()
│   └── save to CSV
└── visualizations
    ├── plt.figure()
    ├── plt.plot()
    ├── plt.savefig()
    └── plt.show()
"""

# Filter parameters
USE_FILTER = True
FILTER_TYPE = 'bandpass_individual'  # Options: "highpass", "bandpass_concat", "bandpass_individual"
SAMPLING_RATE = 1000  # Hz
LOW_CUTOFF = 8  # Hz
HIGH_CUTOFF = 100.0  # Hz
HIGHPASS_CUTOFF = 2.0
FILTER_ORDER = 4

# Training parameters
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 20

# Model save path
MODEL_SAVE_PATH = './data/1000-bandpass8-100Hz.pth'

# Device configurationll
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class LFPDataset(Dataset):
    """
    Dataset class for loading LFP (Local Field Potential) signals
    """

    def __init__(self, signals, labels, file_names=None, trial_indices=None, transform=None):
        self.signals = signals
        self.labels = labels
        self.file_names = file_names
        self.trial_indices = trial_indices
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        if self.transform:
            signal = self.transform(signal)
        signal = signal.reshape(1, -1)

        if self.file_names is not None and self.trial_indices is not None:
            return torch.FloatTensor(signal), torch.LongTensor([label]), self.file_names[idx], self.trial_indices[idx]
        elif self.file_names is not None:
            return torch.FloatTensor(signal), torch.LongTensor([label]), self.file_names[idx]
        else:
            return torch.FloatTensor(signal), torch.LongTensor([label])


class AttnPool1d(nn.Module):
    """
    Attention pooling over time dimension
    Replaces global average pooling to let model learn where to look
    """

    def __init__(self, channels, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, kernel_size=1)
        )

    def forward(self, x):  # x: [B, C, T]
        logits = self.score(x)  # [B, 1, T]
        w = torch.softmax(logits, dim=-1)  # attention weights over time
        pooled = (x * w).sum(dim=-1)  # [B, C] - weighted sum
        return pooled


class AgeClassifier(nn.Module):
    """
    With Attention, CNN-based age classification model for EEG signals
    """

    def __init__(self, num_classes):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.4),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.4),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.4),

            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.AdaptiveAvgPool1d(1),    # Removed - replaced with attention pooling
            nn.Dropout(0.4)
        )

        # Add attention time pooling module
        self.pool = AttnPool1d(512)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)  # [B, 512] - attention-weighted pooling
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for model interpretability
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
            self.activations = output
            return output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return grad_input

        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, target_class=None):
        self.model.eval()

        with torch.set_grad_enabled(True):
            x = x.clone().requires_grad_(True)

            output = self.model(x)

            if target_class is None:
                target_class = output.argmax(dim=1)
            elif isinstance(target_class, int):
                target_class = torch.tensor([target_class], device=x.device)

            one_hot = torch.zeros(output.size(), device=x.device)
            one_hot[0, target_class] = 1.0

            self.model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)

            if self.gradients is None:
                raise RuntimeError("No gradients captured")

            weights = self.gradients.mean(dim=2, keepdim=True)

            cam = (weights * self.activations).sum(dim=1)
            cam = torch.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            return cam.detach().squeeze().cpu().numpy()

    def __del__(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


class Logger:
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'console_output.log.txt.txt')
        self.terminal = sys.stdout

        self.log = open(self.log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def apply_bandpass_filter_concat(signals, sampling_rate=SAMPLING_RATE, lowcut=LOW_CUTOFF, highcut=HIGH_CUTOFF,
                                 order=FILTER_ORDER):
    """
    Apply bandpass filter to EEG signals (ButterWorth filter)
    Original version with concatenation and potential edge effects
    """
    original_shape = signals.shape
    flattened_signals = signals.flatten()

    nyquist = sampling_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design bandpass filter
    b, a = signal.butter(order, [low, high], btype='band', analog=False)

    filtered_signals = signal.filtfilt(b, a, flattened_signals)

    filtered_signals = filtered_signals.reshape(original_shape)

    return filtered_signals


def apply_bandpass_filter_individual(signals, sampling_rate=SAMPLING_RATE, lowcut=LOW_CUTOFF, highcut=HIGH_CUTOFF,
                                     order=FILTER_ORDER):
    """
    Apply bandpass filter to EEG signals (ButterWorth filter)
    Each trial filtered independently to avoid edge effects between trials
    """
    original_shape = signals.shape

    print(f"Applying bandpass filter ({lowcut}-{highcut}Hz) to {signals.shape[0]} trials...")

    # Design bandpass filter (same parameters)
    nyquist = sampling_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band', analog=False)

    # Check if signals is 1D or 2D
    if len(original_shape) == 1:
        # Single signal
        filtered_signals = signal.filtfilt(b, a, signals)
    else:
        # Multiple trials: filter each independently
        n_trials = original_shape[0]
        filtered_signals = np.zeros_like(signals)

        for i in range(n_trials):
            # Filter each trial separately to avoid edge effects between trials
            filtered_signals[i] = signal.filtfilt(b, a, signals[i])

            # Optional: Show progress for large datasets
            if i > 0 and i % 100 == 0:
                print(f"  Filtered {i}/{n_trials} trials...")

    print(f"Filtering complete. Shape maintained: {filtered_signals.shape}")
    return filtered_signals


def apply_highpass_filter(signals, sampling_rate=1000, cutoff=8.0, order=4):
    original_shape = signals.shape
    # flattened_signals = signals.flatten()

    nyquist = sampling_rate / 2
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

    if len(original_shape) == 1:
        # Single signal
        filtered_signals = signal.filtfilt(b, a, signals)
    else:
        # Multiple trials: filter each independently
        n_trials = original_shape[0]
        filtered_signals = np.zeros_like(signals)

        for i in range(n_trials):
            # Filter each trial separately to avoid edge effects between trials
            filtered_signals[i] = signal.filtfilt(b, a, signals[i])

            # Optional: Show progress for large datasets
            if i > 0 and i % 100 == 0:
                print(f"  Filtered {i}/{n_trials} trials...")

    print(f"Filtering complete. Shape maintained: {filtered_signals.shape}")
    return filtered_signals

# def apply_highpass_filter(signals, sampling_rate=1000, cutoff=2.0, order=4):
#     original_shape = signals.shape
#     flattened_signals = signals.flatten()
#     print(original_shape)
#     nyquist = sampling_rate / 2
#     normal_cutoff = cutoff / nyquist
#     b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
#
#     filtered_signals = signal.filtfilt(b, a, flattened_signals)
#
#     filtered_signals = filtered_signals.reshape(original_shape)
#
#     return filtered_signals


def load_data_from_files(data_dir, filter_csv=None):
    """
    Load and preprocess EEG data from .mat files
    """
    filter_dict = {}
    if filter_csv and os.path.exists(filter_csv):
        df_filter = pd.read_csv(filter_csv)
        print(f"First few rows from CSV:")
        print(df_filter.head())

        for _, row in df_filter.iterrows():
            file_name = row['FileName']
            if '_age' in file_name:
                file_name = file_name.split('_age')[0]
            elif not file_name.endswith('.mat'):
                file_name = file_name + '.mat'
            filter_dict[file_name] = row['Label']
        print(f"Loaded filter with {len(filter_dict)} files")

        print("Sample files from filter:")
        for i, fname in enumerate(list(filter_dict.keys())[:5]):
            print(f"  {fname}: {filter_dict[fname]}")

    file_list = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    print(f"Found {len(file_list)} .mat files in directory")
    print("Sample .mat files in directory:")
    for i, fname in enumerate(file_list[:5]):
        print(f"  {fname}")

    matched_files = set(file_list) & set(filter_dict.keys())
    print(f"Files that match between directory and filter: {len(matched_files)}")

    test_samples_info = []
    all_signals_list = []
    all_labels_list = []
    all_filenames_list = []
    all_trial_indices = []

    files_processed = 0
    for fname in file_list:
        if filter_dict:
            if fname not in filter_dict:
                continue
            if filter_dict[fname] == 0:
                continue

        fpath = os.path.join(data_dir, fname)
        try:
            with h5py.File(fpath, 'r') as f:
                if 'lfpN' not in f:
                    print(f"  {fname}: No 'lfpN' dataset found")
                    continue

                lfp = np.array(f['lfpN'][:])
                print(f"  {fname}: lfp shape {lfp.shape}")

                if 'par' in f and 'Age' in f['par']:
                    try:
                        age_ref = f['par']['Age'][()]
                        if isinstance(age_ref, h5py.Reference):
                            age_val = f[age_ref][()][0][0]
                        else:
                            age_val = age_ref[0][0] if isinstance(age_ref, np.ndarray) else age_ref
                        age = int(age_val)
                    except:
                        age = 0
                        print(f"  {fname}: Error reading age, using 0")
                else:
                    age = 0
                    print(f"  {fname}: No age found, using 0")

                trials_added = 0
                for i, trial in enumerate(lfp):
                    if trial.shape[0] == 1000:
                        all_signals_list.append(trial)
                        all_labels_list.append(age)
                        all_filenames_list.append(fname)
                        all_trial_indices.append(i)
                        test_samples_info.append({
                            'file_name': fname,
                            'trial_index': i,
                            'age': age
                        })
                        trials_added += 1

                print(f"  {fname}: Added {trials_added} trials")
                files_processed += 1

        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

    print(f"Files processed: {files_processed}")
    print(f"Total trials collected: {len(all_signals_list)}")

    if all_signals_list:
        all_signals = np.array(all_signals_list)
        print(f"Loaded {all_signals.shape[0]} samples before filtering")

        # Apply bandpass filter
        # filtered_signals = apply_bandpass_filter(all_signals, sampling_rate=SAMPLING_RATE)
        if USE_FILTER:
            print(f"Applying filter:{FILTER_TYPE}")

            if FILTER_TYPE == "highpass":
                filtered_signals = apply_highpass_filter(all_signals, sampling_rate=SAMPLING_RATE,
                                                         cutoff=HIGHPASS_CUTOFF, order=FILTER_ORDER)
                print(f"Applied highpass filter with cutoff {HIGHPASS_CUTOFF}Hz")

            elif FILTER_TYPE == "bandpass_concat":
                # Apply bandpass filter (concatenated version with edge effects)
                filtered_signals = apply_bandpass_filter_concat(all_signals, sampling_rate=SAMPLING_RATE,
                                                                lowcut=LOW_CUTOFF, highcut=HIGH_CUTOFF,
                                                                order=FILTER_ORDER)
                print(f"Applied bandpass filter ({LOW_CUTOFF}-{HIGH_CUTOFF}Hz) - concatenated version")

            elif FILTER_TYPE == "bandpass_individual":
                # Apply bandpass filter (individual version without edge effects)
                filtered_signals = apply_bandpass_filter_individual(all_signals, sampling_rate=SAMPLING_RATE,
                                                                    lowcut=LOW_CUTOFF, highcut=HIGH_CUTOFF,
                                                                    order=FILTER_ORDER)
                print(f"Applied bandpass filter ({LOW_CUTOFF}-{HIGH_CUTOFF}Hz) - individual trials version")

            else:
                print(f"ERROR: Unknown filter type '{FILTER_TYPE}'. Using raw signals.")
                filtered_signals = all_signals

        else:
            # No filtering - use raw signals
            print("No filter applied - using raw signals")
            filtered_signals = all_signals

        for i in range(min(3, len(filtered_signals))):
            print(f"Sample {i}: mean before filtering: {np.mean(all_signals[i]):.6f}, "
                  f"mean after filtering: {np.mean(filtered_signals[i]):.6f}")

        signals = filtered_signals
        labels = np.array(all_labels_list)
        file_names = all_filenames_list
        trial_indices = all_trial_indices

    else:
        signals = np.array([])
        labels = np.array([])
        file_names = []
        trial_indices = []
        print("WARNING: No signals were loaded!")

    if test_samples_info:
        test_df = pd.DataFrame(test_samples_info)
        test_df.to_csv('test_samples_info_bandpass8-100Hz.csv', index=False)
        print(f"Saved test samples info to test_samples_info_bandpass8-100Hz.csv ({len(test_df)} samples)")

    return signals, labels, file_names, trial_indices


def preprocess_data(signals, labels):
    """
    Preprocess signals with standardization and encode labels
    """
    signals_normalized = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        signals_normalized[i] = (signals[i] - np.mean(signals[i])) / (np.std(signals[i]) + 1e-8)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    return signals_normalized, labels_encoded, le


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    """
    Train the model with early stopping
    """
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []  # NEW: Add list to store training accuracies

    best_acc = 0.0
    patience = EARLY_STOPPING_PATIENCE
    counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0  # NEW: Counter for correct predictions during training
        train_total = 0  # NEW: Counter for total samples during training

        for batch in train_loader:
            if len(batch) == 4:
                signals, labels, _, _ = batch
            elif len(batch) == 3:
                signals, labels, _ = batch
            else:
                signals, labels = batch

            signals = signals.to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # NEW: Calculate training accuracy for this batch
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Calculate training accuracy for this epoch
        train_acc = 100 * train_correct / train_total
        train_accuracies.append(train_acc)  # NEW: Store training accuracy

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    signals, labels, _, _ = batch
                elif len(batch) == 3:
                    signals, labels, _ = batch
                else:
                    signals, labels = batch

                signals = signals.to(device)
                labels = labels.squeeze().to(device)

                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # NEW: Update print statement to include training accuracy
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    return train_losses, val_losses, val_accuracies, train_accuracies  # NEW: Return train_accuracies


def save_detailed_test_info(test_dataset, filename='test_samples_detailed_bandpass8-100Hz.csv'):
    """
    Save detailed information about test samples
    """
    if test_dataset.file_names is not None and test_dataset.trial_indices is not None:
        detailed_info = []

        for i in range(len(test_dataset)):
            detailed_info.append({
                'dataset_index': i,
                'file_name': test_dataset.file_names[i],
                'trial_index': test_dataset.trial_indices[i],
                'age': test_dataset.labels[i] if i < len(test_dataset.labels) else -1
            })

        detailed_df = pd.DataFrame(detailed_info)
        detailed_df.to_csv(filename, index=False)
        print(f"Saved detailed test samples info to {filename} ({len(detailed_df)} samples)")
        return detailed_df
    else:
        print("No file names or trial indices available in test dataset")
        return None


def test_model_with_gradcam(model, test_loader, label_encoder, device):
    """
    Test model and generate Grad-CAM visualizations
    """
    model.eval()

    # Initialize Grad-CAM
    target_layer = model.features[-4]  # Last convolutional layer
    grad_cam = GradCAM(model, target_layer)

    all_cams = []
    all_signals = []
    all_true_ages = []
    all_predicted_ages = []
    all_file_names = []
    all_trial_indices = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                signals, labels, file_names, trial_indices = batch
                all_file_names.extend(file_names)
                all_trial_indices.extend(trial_indices)
            elif len(batch) == 3:
                signals, labels, file_names = batch
                all_file_names.extend(file_names)
            else:
                signals, labels = batch

            signals = signals.to(device)
            labels = labels.squeeze().to(device)

            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)

            for j in range(signals.size(0)):
                signal_tensor = signals[j].unsqueeze(0)
                true_age = labels[j].item()
                pred_age = predicted[j].item()

                try:
                    cam = grad_cam(signal_tensor, pred_age)
                    all_cams.append(cam)
                    all_signals.append(signals[j].cpu().numpy().squeeze())
                    all_true_ages.append(label_encoder.inverse_transform([true_age])[0])
                    all_predicted_ages.append(label_encoder.inverse_transform([pred_age])[0])
                except Exception as e:
                    print(f"Error generating Grad-CAM: {e}")
                    continue

    # Convert to numpy arrays
    all_cams = np.array(all_cams)
    all_signals = np.array(all_signals)
    all_true_ages = np.array(all_true_ages)
    all_predicted_ages = np.array(all_predicted_ages)

    print(f"Generated Grad-CAM for {len(all_cams)} samples")

    # Calculate accuracy
    accuracy = np.mean(all_true_ages == all_predicted_ages)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save detailed results
    results_df = pd.DataFrame({
        'file_name': all_file_names[:len(all_true_ages)],
        'trial_index': all_trial_indices[:len(all_true_ages)] if all_trial_indices else [-1] * len(all_true_ages),
        'true_age': all_true_ages,
        'predicted_age': all_predicted_ages,
        'correct': (all_true_ages == all_predicted_ages).astype(int)
    })
    results_df.to_csv('test_results_detailed_bandpass8-100Hz.csv', index=False)
    print("Saved detailed results to 'test_results_detailed_bandpass8-100Hz.csv'")

    return accuracy, all_cams, all_signals, all_true_ages, all_predicted_ages


def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, filter_info=None):
    """
    Plot learning curves with correct epoch numbering and additional info
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 添加滤波器信息作为图标题
    if filter_info:
        fig.suptitle(f"Learning Curves - Filter: {filter_info}", fontsize=14, fontweight='bold')

    # 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, label='Val Loss', color='red', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 在损失图中标记最佳epoch
    best_val_epoch = np.argmin(val_losses) + 1 if val_losses else 0
    if best_val_epoch > 0:
        ax1.axvline(x=best_val_epoch, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.text(best_val_epoch, ax1.get_ylim()[0], f'Best Epoch: {best_val_epoch}',
                 rotation=90, verticalalignment='bottom')

    # 准确率曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_accuracies, label='Train Acc', color='blue', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_accuracies, label='Val Acc', color='red', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 在准确率图中标记最佳epoch
    best_acc_epoch = np.argmax(val_accuracies) + 1 if val_accuracies else 0
    if best_acc_epoch > 0:
        ax2.axvline(x=best_acc_epoch, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax2.text(best_acc_epoch, ax2.get_ylim()[0], f'Best Acc: {best_acc_epoch}',
                 rotation=90, verticalalignment='bottom')

    # 准确率差距
    ax3 = axes[1, 0]
    acc_gap = [val_accuracies[i] - train_accuracies[i] for i in range(len(train_accuracies))]
    ax3.plot(epochs, acc_gap, color='green', linewidth=2, marker='^', markersize=4)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Val Acc - Train Acc (%)')
    ax3.set_title('Accuracy Gap (Positive = Val better)')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 损失比值
    ax4 = axes[1, 1]
    loss_ratio = [val_losses[i] / train_losses[i] for i in range(len(train_losses))]
    ax4.plot(epochs, loss_ratio, color='purple', linewidth=2, marker='d', markersize=4)
    ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Val Loss / Train Loss')
    ax4.set_title('Loss Ratio (<1.0 = Val better)')
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 添加统计信息
    info_text = f"Final Train Loss: {train_losses[-1]:.4f}\nFinal Val Loss: {val_losses[-1]:.4f}\n"
    info_text += f"Final Train Acc: {train_accuracies[-1]:.2f}%\nFinal Val Acc: {val_accuracies[-1]:.2f}%\n"
    info_text += f"Best Val Acc: {max(val_accuracies):.2f}% at epoch {best_acc_epoch}"

    fig.text(0.02, 0.02, info_text, transform=fig.transFigure, fontsize=10,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部文本留出空间
    plt.savefig('learning_analysis.png', dpi=300)
    plt.show()



def main():
    """
    Main execution function
    """
    LOG_SAVE_DIR = './logs'
    logger = Logger(LOG_SAVE_DIR)
    sys.stdout = logger

    print(f"Logs will be saved to: {logger.log_file}")
    print("=" * 50)
    print("EEG Age Classification Training Started")
    print("=" * 50)

    data_dir = "./data/1000"
    filter_csv = "./data/mTable.csv"

    print("Loading data...")
    signals, labels, file_names, trial_indices = load_data_from_files(data_dir, filter_csv)
    print(f"Loaded {len(signals)} samples")

    if len(signals) == 0:
        print("ERROR: No data loaded! Exiting.")
        return

    signals_processed, labels_processed, label_encoder = preprocess_data(signals, labels)

    # Split data into train/val/test sets
    X_temp, X_test, y_temp, y_test, files_temp, files_test, trials_temp, trials_test = train_test_split(
        signals_processed, labels_processed, file_names, trial_indices, test_size=0.15,
        stratify=labels_processed, random_state=42
    )
    X_train, X_val, y_train, y_val, files_train, files_val, trials_train, trials_val = train_test_split(
        X_temp, y_temp, files_temp, trials_temp, test_size=0.15, stratify=y_temp, random_state=42
    )

    # Create datasets
    train_dataset = LFPDataset(X_train, y_train, files_train, trials_train)
    val_dataset = LFPDataset(X_val, y_val, files_val, trials_val)
    test_dataset = LFPDataset(X_test, y_test, files_test, trials_test)

    # Save test sample information
    save_detailed_test_info(test_dataset, 'test_samples_detailed_bandpass8-100Hz.csv')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    num_classes = len(np.unique(labels_processed))
    model = AgeClassifier(num_classes=num_classes).to(device)

    # Training
    print("Training new model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # NEW: Receive train_accuracies from train_model function
    train_losses, val_losses, val_accuracies, train_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS
    )

    # 在main()中调
    plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies,
                         f"{FILTER_TYPE} ({LOW_CUTOFF}-{HIGH_CUTOFF}Hz)")

    # Load best model
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Testing with Grad-CAM
    print("Running comprehensive testing...")
    accuracy, all_cams, all_signals, all_true_ages, all_predicted_ages = test_model_with_gradcam(
        model, test_loader, label_encoder, device
    )

    # Generate visualizations
    if len(all_cams) > 0:

        original_length = all_signals.shape[1]
        cam_length = all_cams.shape[1]

        # Interpolate Grad-CAM if necessary
        if cam_length != original_length:
            print(f"Interpolating Grad-CAM from {cam_length} to {original_length} points")
            resized_cams = []
            for cam in all_cams:
                resized_cam = np.interp(
                    np.linspace(0, cam_length - 1, original_length),
                    np.arange(cam_length),
                    cam
                )
                resized_cams.append(resized_cam)
            all_cams = np.array(resized_cams)

        # Overall average visualization
        mean_cam = np.mean(all_cams, axis=0)
        mean_signal = np.mean(all_signals, axis=0)
        time_axis = np.arange(len(mean_signal))

        plt.figure(figsize=(15, 10))
        plt.plot(time_axis, mean_signal, 'b-', label='Average Signal', linewidth=2, alpha=0.8)
        plt.plot(time_axis, mean_cam, 'r-', label='Average Grad-CAM Importance', linewidth=2, alpha=0.8)
        plt.fill_between(time_axis, mean_cam, alpha=0.3, color='red')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude / Importance')
        plt.title(f'Average Signal and Grad-CAM Importance\nTest Accuracy: {accuracy:.2%}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('combined_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Age-specific analysis
        unique_ages = np.unique(all_true_ages)
        print(f"Unique ages in test set: {unique_ages}")

        if len(unique_ages) > 0:
            fig, axes = plt.subplots(len(unique_ages), 1, figsize=(15, 5 * len(unique_ages)))
            if len(unique_ages) == 1:
                axes = [axes]

            for i, age in enumerate(unique_ages):
                age_mask = all_true_ages == age
                if np.sum(age_mask) > 0:
                    age_cams = all_cams[age_mask]
                    age_signals = all_signals[age_mask]

                    mean_age_cam = np.mean(age_cams, axis=0)
                    mean_age_signal = np.mean(age_signals, axis=0)
                    time_axis = np.arange(len(mean_age_signal))

                    axes[i].plot(time_axis, mean_age_signal, 'b-', label=f'Signal', linewidth=2)
                    axes[i].plot(time_axis, mean_age_cam, 'r-', label=f'Importance', linewidth=2)
                    axes[i].set_title(f'Age {age} (n={len(age_cams)} samples)')
                    axes[i].set_ylabel('Amplitude / Importance')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)

                    if i == len(unique_ages) - 1:
                        axes[i].set_xlabel('Time Points')

            plt.tight_layout()
            plt.savefig('age_specific_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            logger.close()
            sys.stdout = logger.terminal



if __name__ == "__main__":
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)

    # Check required packages
    try:
        import torch, pandas, numpy, scipy, h5py, sklearn, matplotlib

        print("All required packages are available")
    except ImportError as e:
        print(f"Missing package: {e}")
        exit(1)

    # Check CUDA availability
    import torch

    if torch.cuda.is_available():
        print(f"CUDA is available ({torch.cuda.device_count()} GPU(s))")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(" CUDA is not available - using CPU")

    print("=" * 50)
    print()
    main()