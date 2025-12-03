"""
Standalone evaluation script for LOTL detection models.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import time

from data_loader import load_dataset, filter_label_agreement, get_labels, split_dataset
from ensemble import LOTLEnsemble
from models import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate LOTL detection models")
    parser.add_argument("--dataset", type=str, default="data.jsonl", help="Path to dataset file")
    parser.add_argument(
        "--model-dir", type=str, default="models", help="Directory containing trained models"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LOTL Detection - Model Evaluation")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    events = load_dataset(args.dataset)
    print(f"Loaded {len(events)} events")

    # Filter events where Claude and ground truth agree
    print("\nFiltering events where Claude and ground truth labels agree...")
    filtered_events, disagreement_events = filter_label_agreement(events)
    print(f"Kept {len(filtered_events)} events with agreement")
    print(f"Removed {len(disagreement_events)} events with disagreement")

    # Get labels (use Claude's predictions as target)
    labels = get_labels(filtered_events, use_claude_label=True)

    # Count labels
    from collections import Counter

    label_counts = Counter(labels)
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")

    # Split dataset
    print(f"\nSplitting dataset (test_size={args.test_size})...")
    train_events, test_events, train_indices, test_indices = split_dataset(
        filtered_events, test_size=args.test_size, random_seed=args.random_seed
    )

    test_labels = [labels[i] for i in test_indices]

    print(f"Test set: {len(test_events)} events")

    # Load model
    print(f"\nLoading model from {args.model_dir}...")
    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        print(f"❌ Model directory {args.model_dir} does not exist!")
        print("Please train the model first using: make train")
        return

    try:
        ensemble = LOTLEnsemble()
        ensemble.load(str(model_dir))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("Please train the model first using: make train")
        return

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)

    # Measure inference time
    start_time = time.time()
    test_predictions = ensemble.predict(test_events)
    inference_time = time.time() - start_time
    avg_latency = inference_time / len(test_events) * 1000  # ms per event

    print(f"\nInference Performance:")
    print(f"  Total time: {inference_time:.4f} seconds")
    print(f"  Average latency: {avg_latency:.4f} ms per event")
    print(f"  Throughput: {len(test_events)/inference_time:.2f} events/second")

    # Evaluate metrics
    metrics = evaluate_model(test_labels, test_predictions, "Ensemble Model")

    # Calculate cost comparison
    print("\n" + "=" * 60)
    print("Cost Analysis")
    print("=" * 60)

    claude_cost_per_alert = 0.0018
    our_cost_per_alert = 0.00006  # Estimated

    claude_total = claude_cost_per_alert * len(test_events)
    our_total = our_cost_per_alert * len(test_events)

    print(f"\nCost per million alerts:")
    print(f"  Claude-Sonnet-4.5: ${claude_cost_per_alert * 1_000_000:,.2f}")
    print(f"  Our Solution: ${our_cost_per_alert * 1_000_000:,.2f}")
    print(f"  Savings: ${(claude_cost_per_alert - our_cost_per_alert) * 1_000_000:,.2f}")
    print(f"  Cost Reduction: {claude_cost_per_alert / our_cost_per_alert:.1f}x")

    # Save results
    results = {
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        },
        "performance": {
            "inference_time_seconds": inference_time,
            "avg_latency_ms": avg_latency,
            "throughput_events_per_sec": len(test_events) / inference_time,
            "num_test": len(test_events),
        },
        "cost_analysis": {
            "claude_cost_per_million": claude_cost_per_alert * 1_000_000,
            "our_cost_per_million": our_cost_per_alert * 1_000_000,
            "savings_per_million": (claude_cost_per_alert - our_cost_per_alert) * 1_000_000,
            "cost_reduction_factor": claude_cost_per_alert / our_cost_per_alert,
        },
    }

    results_path = model_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Evaluation results saved to {results_path}")

    # Analyze failures
    print("\n" + "=" * 60)
    print("Failure Analysis")
    print("=" * 60)

    failures = []
    for i, (true_label, pred_label) in enumerate(zip(test_labels, test_predictions)):
        if true_label != pred_label:
            failures.append(
                {
                    "index": i,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "event": test_events[i],
                }
            )

    print(f"\nTotal misclassifications: {len(failures)}")
    print(f"False Positives: {sum(1 for f in failures if f['true_label'] == 'benign')}")
    print(f"False Negatives: {sum(1 for f in failures if f['true_label'] == 'malicious')}")

    if failures:
        print("\nSample failures:")
        for i, failure in enumerate(failures[:3]):
            event = failure["event"]
            cmdline = event.get("CommandLine", "") or event.get("commandLine", "")
            print(f"\n  Failure {i+1}:")
            print(f"    True: {failure['true_label']}, Predicted: {failure['predicted_label']}")
            print(f"    Command: {cmdline[:100]}...")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
