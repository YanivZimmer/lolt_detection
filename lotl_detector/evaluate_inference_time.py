"""
Evaluate inference time for Random Forest model.
Optimizes and benchmarks performance.
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from data_loader import load_dataset, sanitize_event_for_inference
from models import RandomForestModel
from feature_extractor import ComprehensiveFeatureExtractor


def benchmark_rf_inference(
    model: RandomForestModel,
    events: List[Dict[str, Any]],
    feature_extractor: ComprehensiveFeatureExtractor,
    feature_names: List[str],
    n_iterations: int = 10,
) -> Dict[str, float]:
    """
    Benchmark Random Forest inference time.

    Args:
        model: Trained Random Forest model
        events: List of events to predict
        feature_extractor: Feature extractor
        feature_names: List of feature names
        n_iterations: Number of iterations for benchmarking

    Returns:
        Dictionary with timing metrics
    """
    # Extract features once (cached)
    feature_list = []
    for event in events:
        features = feature_extractor.extract_all_features(event)
        feature_list.append(features)

    X = np.array([[features.get(name, 0) for name in feature_names] for features in feature_list])

    # Warm-up
    _ = model.predict(X[:10])

    # Benchmark prediction
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Benchmark feature extraction
    extraction_times = []
    for event in events[:100]:  # Sample for speed
        start = time.perf_counter()
        _ = feature_extractor.extract_all_features(event)
        end = time.perf_counter()
        extraction_times.append((end - start) * 1000)

    return {
        "avg_inference_ms": np.mean(times),
        "std_inference_ms": np.std(times),
        "min_inference_ms": np.min(times),
        "max_inference_ms": np.max(times),
        "avg_per_event_ms": np.mean(times) / len(events),
        "avg_feature_extraction_ms": np.mean(extraction_times),
        "total_events": len(events),
        "throughput_events_per_sec": len(events) / (np.mean(times) / 1000),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Random Forest inference time")
    parser.add_argument("--model-dir", type=str, default="models", help="Path to model directory")
    parser.add_argument("--dataset", type=str, default="data.jsonl", help="Path to dataset")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument(
        "--output", type=str, default="inference_benchmark.json", help="Output file for results"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Random Forest Inference Time Evaluation")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {args.model_dir}...")
    model_dir = Path(args.model_dir)

    if not (model_dir / "random_forest.pkl").exists():
        print(f"❌ Model not found at {model_dir / 'random_forest.pkl'}")
        print("Please train the model first using `make train`")
        return

    rf_model = RandomForestModel()
    rf_model.load(str(model_dir / "random_forest.pkl"))

    # Load feature names
    import pickle

    with open(model_dir / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    events = load_dataset(args.dataset)

    # Sanitize events
    sanitized_events = [sanitize_event_for_inference(event) for event in events]

    print(f"Loaded {len(sanitized_events)} events")

    # Initialize feature extractor and restrict it to the same selected feature subset
    # used during training (so we only compute those features, especially for embeddings).
    feature_extractor = ComprehensiveFeatureExtractor()
    feature_extractor.selected_feature_names = set(feature_names)

    # Benchmark
    print(f"\nBenchmarking inference time ({args.iterations} iterations)...")
    metrics = benchmark_rf_inference(
        rf_model, sanitized_events, feature_extractor, feature_names, args.iterations
    )

    # Print results
    print("\n" + "=" * 60)
    print("Inference Performance Results")
    print("=" * 60)
    print(f"\nTotal Events: {metrics['total_events']}")
    print(f"\nInference Time (per batch):")
    print(f"  Average: {metrics['avg_inference_ms']:.4f} ms")
    print(f"  Std Dev: {metrics['std_inference_ms']:.4f} ms")
    print(f"  Min:     {metrics['min_inference_ms']:.4f} ms")
    print(f"  Max:     {metrics['max_inference_ms']:.4f} ms")
    print(f"\nPer-Event Latency:")
    print(f"  Average: {metrics['avg_per_event_ms']:.4f} ms per event")
    print(f"\nFeature Extraction:")
    print(f"  Average: {metrics['avg_feature_extraction_ms']:.4f} ms per event")
    print(f"\nThroughput:")
    print(f"  {metrics['throughput_events_per_sec']:.2f} events/second")
    print(f"\nPer-Event total process time (fe+latency) :")
    print(
        f"  Average: {(metrics['avg_per_event_ms']+metrics['avg_feature_extraction_ms']):.4f} ms per event total process time"
    )

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")

    # Comparison with Claude baseline
    claude_latency_ms = 500  # Estimated Claude latency
    speedup = claude_latency_ms / (
        metrics["avg_per_event_ms"] + metrics["avg_feature_extraction_ms"]
    )

    print(f"\n" + "=" * 60)
    print("Comparison with Claude-Sonnet-4.5 Baseline")
    print("=" * 60)
    print(f"Claude latency (estimated): {claude_latency_ms} ms per event")
    print(
        f"Our fe+latency: {(metrics['avg_per_event_ms']+metrics['avg_feature_extraction_ms']):.4f} ms per event"
    )
    print(f"Speedup: {speedup:.2f}x faster")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
