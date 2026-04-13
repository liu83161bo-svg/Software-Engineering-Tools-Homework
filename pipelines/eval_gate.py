"""
Evaluation Gate for HW4 - Model Performance Check
Compares model performance against thresholds and fails CI if below thresholds
"""

import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader


class EvalGate:
    def __init__(self, config_path="./configs/thresholds.yaml"):
        """Initialize evaluation gate with thresholds"""
        self.config_path = config_path
        self.load_thresholds()

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_thresholds(self):
        """Load thresholds from YAML config"""
        with open(self.config_path, 'r') as f:
            self.thresholds = yaml.safe_load(f)

        print(f"Loaded thresholds from {self.config_path}")
        print(json.dumps(self.thresholds, indent=2))

    def load_model(self, model_path):
        """Load trained model"""
        # Load golden set to get number of classes
        golden_path = "./data/golden_set/golden_samples.jsonl"
        with open(golden_path, 'r') as f:
            ages = [json.loads(line)['age'] for line in f]

        num_classes = len(set(ages))

        # Initialize and load model
        model = AgeClassifier(num_classes=num_classes)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"WARNING: Model not found at {model_path}, using random weights")

        model.to(self.device)
        model.eval()
        return model, num_classes

    def load_golden_set(self):
        """Load golden set for evaluation"""
        golden_path = "./data/golden_set/golden_samples.jsonl"

        signals = []
        labels = []

        with open(golden_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                signals.append(np.array(sample['signal']))
                labels.append(sample['age'])

        signals = np.array(signals)
        labels = np.array(labels)

        # Preprocess
        signals_proc, labels_proc, label_encoder = preprocess_data(signals, labels)

        # Create dataset
        dataset = LFPDataset(signals_proc, labels_proc)

        print(f"Loaded golden set: {len(dataset)} samples")
        return dataset, label_encoder

    def evaluate_model(self, model, dataset, label_encoder):
        """Evaluate model on golden set"""
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) >= 2:
                    signals, labels = batch[0], batch[1]
                else:
                    continue

                signals = signals.to(self.device)
                labels = labels.squeeze().to(self.device)

                outputs = model(signals)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = np.mean(all_preds == all_labels)

        # Per-class metrics
        unique_labels = np.unique(all_labels)
        per_class_recall = []

        for label in unique_labels:
            mask = all_labels == label
            if mask.sum() > 0:
                recall = np.mean(all_preds[mask] == label)
                per_class_recall.append(recall)

        avg_recall = np.mean(per_class_recall)

        # Create results dictionary
        results = {
            'accuracy': float(accuracy),
            'avg_recall': float(avg_recall),
            'num_samples': len(all_labels),
            'per_class_recall': {int(label): float(recall) for label, recall in zip(unique_labels, per_class_recall)}
        }

        print(f"Evaluation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Samples: {len(all_labels)}")

        return results

    def check_thresholds(self, results):
        """Check results against thresholds"""
        print("\n" + "=" * 60)
        print("Threshold Check")
        print("=" * 60)

        failures = []

        # Check accuracy threshold
        acc_threshold = self.thresholds['accuracy']['threshold']
        acc_tolerance = self.thresholds['accuracy']['tolerance']

        if results['accuracy'] < acc_threshold - acc_tolerance:
            failures.append(
                f"Accuracy below threshold: {results['accuracy']:.4f} < {acc_threshold - acc_tolerance:.4f}")
            print(f"✗ Accuracy FAIL: {results['accuracy']:.4f} < {acc_threshold:.4f}")
        else:
            print(f"✓ Accuracy PASS: {results['accuracy']:.4f} >= {acc_threshold:.4f}")

        # Check recall threshold
        recall_threshold = self.thresholds['avg_recall']['threshold']
        recall_tolerance = self.thresholds['avg_recall']['tolerance']

        if results['avg_recall'] < recall_threshold - recall_tolerance:
            failures.append(
                f"Average recall below threshold: {results['avg_recall']:.4f} < {recall_threshold - recall_tolerance:.4f}")
            print(f"✗ Average Recall FAIL: {results['avg_recall']:.4f} < {recall_threshold:.4f}")
        else:
            print(f"✓ Average Recall PASS: {results['avg_recall']:.4f} >= {recall_threshold:.4f}")

        # Check per-class recall for critical ages
        critical_ages = self.thresholds.get('critical_ages', [])
        for age in critical_ages:
            if age in results['per_class_recall']:
                recall = results['per_class_recall'][age]
                min_recall = self.thresholds['per_class_recall']['threshold']

                if recall < min_recall:
                    failures.append(f"Age {age} recall below threshold: {recall:.4f} < {min_recall:.4f}")
                    print(f"✗ Age {age} Recall FAIL: {recall:.4f} < {min_recall:.4f}")
                else:
                    print(f"✓ Age {age} Recall PASS: {recall:.4f} >= {min_recall:.4f}")

        return failures

    def save_results(self, results, output_path="./reports/metrics.json"):
        """Save evaluation results to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved results to {output_path}")

    def run(self):
        """Run complete evaluation gate"""
        print("=" * 60)
        print("HW4 Evaluation Gate")
        print("=" * 60)

        # Load model (use the one from HW3)
        model_path = "./data/1000-bandpass8-100Hz.pth"
        model, _ = self.load_model(model_path)

        # Load golden set
        dataset, label_encoder = self.load_golden_set()

        # Evaluate model
        results = self.evaluate_model(model, dataset, label_encoder)

        # Check thresholds
        failures = self.check_thresholds(results)

        # Save results
        self.save_results(results)

        # Determine exit code
        if failures:
            print(f"\n✗ EVALUATION FAILED: {len(failures)} threshold(s) violated")
            for failure in failures:
                print(f"  - {failure}")
            return False, failures
        else:
            print(f"\n✓ EVALUATION PASSED: All thresholds met")
            return True, []


def main():
    """Main function for CI integration"""
    eval_gate = EvalGate()
    success, failures = eval_gate.run()

    # Exit with appropriate code
    if success:
        sys.exit(0)
    else:
        print(f"\nFailing CI due to evaluation gate failure")
        sys.exit(1)


if __name__ == "__main__":
    main()