#!/usr/bin/env python3
"""
Simplified Evaluation Gate for HW4
Compares simulated model performance against thresholds
"""

import json
import yaml
import numpy as np
import os
import sys
from pathlib import Path

class SimpleEvalGate:
    def __init__(self, config_path="../configs/thresholds.yaml"):
        """Initialize with thresholds"""
        self.config_path = config_path
        self.load_thresholds()

    def load_thresholds(self):
        """Load thresholds from YAML config"""
        with open(self.config_path, 'r') as f:
            self.thresholds = yaml.safe_load(f)
        print("Thresholds loaded:", json.dumps(self.thresholds, indent=2))

    def simulate_model_performance(self):
        """Simulate model performance on golden set"""
        # Simulate performance metrics
        accuracy = np.random.uniform(0.75, 0.90)  # 75-90% accuracy
        avg_recall = np.random.uniform(0.70, 0.85)  # 70-85% recall

        # Create realistic-looking per-class results
        ages = list(range(16))  # 0-15 age classes
        per_class_recall = {}

        for age in ages:
            # Make younger ages harder to predict
            if age < 5:
                recall = np.random.uniform(0.55, 0.70)
            else:
                recall = np.random.uniform(0.65, 0.85)
            per_class_recall[str(age)] = float(recall)

        results = {
            'accuracy': float(accuracy),
            'avg_recall': float(avg_recall),
            'per_class_recall': per_class_recall,
            'num_samples': 40,  # From golden set
            'timestamp': np.datetime64('now').astype(str),
            'model_version': 'simulated_v1.0'
        }

        print(f"Simulated Accuracy: {accuracy:.4f}")
        print(f"Simulated Avg Recall: {avg_recall:.4f}")
        return results

    def check_thresholds(self, results):
        """Check results against thresholds"""
        print("\n" + "=" * 50)
        print("Threshold Validation")
        print("=" * 50)

        failures = []

        # Check accuracy
        acc_threshold = self.thresholds['accuracy']['threshold']
        if results['accuracy'] < acc_threshold:
            failures.append(f"Accuracy: {results['accuracy']:.4f} < {acc_threshold:.4f}")
            print(f"✗ Accuracy FAIL")
        else:
            print(f"✓ Accuracy PASS: {results['accuracy']:.4f} >= {acc_threshold:.4f}")

        # Check average recall
        recall_threshold = self.thresholds['avg_recall']['threshold']
        if results['avg_recall'] < recall_threshold:
            failures.append(f"Avg Recall: {results['avg_recall']:.4f} < {recall_threshold:.4f}")
            print(f"✗ Avg Recall FAIL")
        else:
            print(f"✓ Avg Recall PASS: {results['avg_recall']:.4f} >= {recall_threshold:.4f}")

        # Check critical ages
        critical_ages = self.thresholds.get('critical_ages', [])
        for age in critical_ages:
            age_key = str(age)
            if age_key in results['per_class_recall']:
                recall = results['per_class_recall'][age_key]
                min_recall = self.thresholds['per_class_recall']['threshold']

                if recall < min_recall:
                    failures.append(f"Age {age} Recall: {recall:.4f} < {min_recall:.4f}")
                    print(f"✗ Age {age} Recall FAIL")
                else:
                    print(f"✓ Age {age} Recall PASS: {recall:.4f} >= {min_recall:.4f}")

        return failures

    def save_results(self, results, output_dir="./reports"):
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {metrics_path}")

        # Save summary for CI
        summary_path = os.path.join(output_dir, "eval_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("EVALUATION GATE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Avg Recall: {results['avg_recall']:.4f}\n")
            f.write(f"Model Version: {results['model_version']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Samples: {results['num_samples']}\n")

        return metrics_path

    def run(self):
        """Run complete evaluation gate"""
        print("=" * 50)
        print("HW4 Evaluation Gate (Simplified)")
        print("=" * 50)

        # Simulate model evaluation
        results = self.simulate_model_performance()

        # Check against thresholds
        failures = self.check_thresholds(results)

        # Save results
        self.save_results(results)

        # Determine outcome
        if failures:
            print(f"\n✗ EVALUATION FAILED: {len(failures)} issues")
            for f in failures:
                print(f"  - {f}")
            return False
        else:
            print(f"\n✓ EVALUATION PASSED")
            return True

def main():
    """Main function for CI integration"""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run evaluation gate
    eval_gate = SimpleEvalGate()
    success = eval_gate.run()

    # Exit with appropriate code
    if success:
        sys.exit(0)
    else:
        print("\nCI will fail due to evaluation gate failure")
        sys.exit(1)

if __name__ == "__main__":
    main()