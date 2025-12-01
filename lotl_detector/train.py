"""
Main training script for LOTL detection models.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import time

from data_loader import load_dataset, filter_label_agreement, get_labels
from ensemble import LOTLEnsemble
from models import evaluate_model


def main():
    parser = argparse.ArgumentParser(description='Train LOTL detection models')
    parser.add_argument('--dataset', type=str, default='data.jsonl',
                       help='Path to dataset file')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--use-rf', action='store_true', default=True,
                       help='Use Random Forest')
    parser.add_argument('--use-nn', action='store_true', default=True,
                       help='Use Neural Network')
    parser.add_argument('--use-llm', action='store_true', default=True,
                       help='Use LLM reasoning')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LOTL Detection Model Training")
    print("="*60)
    
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
    from data_loader import split_dataset
    train_events, test_events, train_indices, test_indices = split_dataset(
        filtered_events, test_size=args.test_size, random_seed=args.random_seed
    )
    
    train_labels = [labels[i] for i in train_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"Training set: {len(train_events)} events")
    print(f"Test set: {len(test_events)} events")
    
    # Train ensemble
    print("\n" + "="*60)
    print("Training Ensemble Model")
    print("="*60)
    
    ensemble = LOTLEnsemble(
        use_random_forest=args.use_rf,
        use_neural_network=args.use_nn,
        use_llm_reasoning=args.use_llm
    )
    
    start_time = time.time()
    ensemble.fit(train_events, train_labels)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluation on Test Set")
    print("="*60)
    
    start_time = time.time()
    test_predictions = ensemble.predict(test_events)
    inference_time = time.time() - start_time
    avg_latency = inference_time / len(test_events) * 1000  # ms per event
    
    print(f"\nInference time: {inference_time:.2f} seconds for {len(test_events)} events")
    print(f"Average latency: {avg_latency:.4f} ms per event")
    
    # Evaluate
    metrics = evaluate_model(test_labels, test_predictions, "Ensemble Model")
    
    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ensemble.save(args.output_dir)
    print("Model saved!")
    
    # Save evaluation results
    results = {
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1'])
        },
        'performance': {
            'training_time_seconds': training_time,
            'inference_time_seconds': inference_time,
            'avg_latency_ms': avg_latency,
            'num_train': len(train_events),
            'num_test': len(test_events)
        }
    }
    
    results_path = Path(args.output_dir) / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {results_path}")
    
    # Test explanation generation
    print("\n" + "="*60)
    print("Testing Explanation Generation")
    print("="*60)
    
    sample_events = test_events[:3]
    explanations = ensemble.predict_with_explanation(sample_events)
    
    for i, (event, expl) in enumerate(zip(sample_events, explanations)):
        print(f"\nSample {i+1}:")
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        print(f"  Command: {cmdline[:100]}...")
        print(f"  Prediction: {expl['prediction']}")
        print(f"  Confidence: {expl['confidence']:.2f}")
        print(f"  Explanation: {expl['explanation']}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

