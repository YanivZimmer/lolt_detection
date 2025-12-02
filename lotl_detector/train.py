"""
Main training script for LOTL detection models with k-fold cross-validation.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import time
import numpy as np

from data_loader import load_dataset, filter_label_agreement, get_labels, get_kfold_splits
from ensemble import LOTLEnsemble
from disagreement_detector import DisagreementDetector
from augmentation import DataAugmenter
from models import evaluate_model


def train_with_kfold(events: List[Dict[str, Any]], labels: List[str], 
                     n_splits: int = 5, random_seed: int = 42,
                     use_augmentation: bool = False, use_disagreement_detector: bool = False,
                     use_rf: bool = True, use_nn: bool = True):
    """
    Train models using k-fold cross-validation.
    
    Args:
        events: List of event dictionaries
        labels: List of labels
        n_splits: Number of folds
        random_seed: Random seed for reproducibility
        use_augmentation: Whether to use data augmentation
        use_disagreement_detector: Whether to train disagreement detector
        use_rf: Use Random Forest
        use_nn: Use Neural Network
    """
    # Get k-fold splits
    print(f"\nCreating {n_splits}-fold cross-validation splits...")
    kfold_splits = get_kfold_splits(events, labels, n_splits=n_splits, random_seed=random_seed)
    print(f"Created {len(kfold_splits)} folds")
    
    # Store results for each fold
    fold_results = []
    all_test_labels = []
    all_test_predictions = []
    
    # Train on each fold
    for fold_idx, (train_indices, test_indices) in enumerate(kfold_splits):
        print("\n" + "="*60)
        print(f"Fold {fold_idx + 1}/{n_splits}")
        print("="*60)
        
        # Get fold data
        train_events = [events[i] for i in train_indices]
        test_events = [events[i] for i in test_indices]
        train_labels_fold = [labels[i] for i in train_indices]
        test_labels_fold = [labels[i] for i in test_indices]
        
        print(f"Training: {len(train_events)} events")
        print(f"Test: {len(test_events)} events")
        
        # Apply augmentation if requested
        if use_augmentation:
            print("Applying data augmentation...")
            augmenter = DataAugmenter(augmentation_factor=0.5, random_seed=random_seed + fold_idx)
            train_events, train_labels_fold = augmenter.augment_dataset(train_events, train_labels_fold)
            print(f"After augmentation: {len(train_events)} training events")
        
        # Train ensemble
        ensemble = LOTLEnsemble(
            use_random_forest=use_rf,
            use_neural_network=use_nn,
            use_llm_reasoning=False
        )
        
        start_time = time.time()
        ensemble.fit(train_events, train_labels_fold)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        start_time = time.time()
        test_predictions = ensemble.predict(test_events)
        inference_time = time.time() - start_time
        
        # Store predictions
        all_test_labels.extend(test_labels_fold)
        all_test_predictions.extend(test_predictions)
        
        # Evaluate fold
        metrics = evaluate_model(test_labels_fold, test_predictions, f"Fold {fold_idx + 1}")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1'])
            },
            'training_time': training_time,
            'inference_time': inference_time
        })
    
    # Overall evaluation across all folds
    print("\n" + "="*60)
    print("Overall K-Fold Cross-Validation Results")
    print("="*60)
    
    overall_metrics = evaluate_model(all_test_labels, all_test_predictions, "Overall (All Folds)")
    
    # Calculate average metrics
    avg_accuracy = np.mean([r['metrics']['accuracy'] for r in fold_results])
    avg_precision = np.mean([r['metrics']['precision'] for r in fold_results])
    avg_recall = np.mean([r['metrics']['recall'] for r in fold_results])
    avg_f1 = np.mean([r['metrics']['f1'] for r in fold_results])
    std_f1 = np.std([r['metrics']['f1'] for r in fold_results])
    
    print(f"\nAverage Metrics across {n_splits} folds:")
    print(f"  Accuracy:  {avg_accuracy:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1-Score:  {avg_f1:.4f} (±{std_f1:.4f})")
    
    return {
        'overall_metrics': {
            'accuracy': float(overall_metrics['accuracy']),
            'precision': float(overall_metrics['precision']),
            'recall': float(overall_metrics['recall']),
            'f1': float(overall_metrics['f1'])
        },
        'average_metrics': {
            'accuracy': float(avg_accuracy),
            'precision': float(avg_precision),
            'recall': float(avg_recall),
            'f1': float(avg_f1),
            'f1_std': float(std_f1)
        },
        'fold_results': fold_results
    }


def main():
    parser = argparse.ArgumentParser(description='Train LOTL detection models with k-fold CV')
    parser.add_argument('--dataset', type=str, default='data.jsonl',
                       help='Path to dataset file')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--use-rf', action='store_true', default=True,
                       help='Use Random Forest')
    parser.add_argument('--use-nn', action='store_true', default=True,
                       help='Use Neural Network')
    parser.add_argument('--use-augmentation', action='store_true', default=False,
                       help='Use data augmentation')
    parser.add_argument('--use-disagreement-detector', action='store_true', default=False,
                       help='Train disagreement detector (V2 model)')
    parser.add_argument('--train-final-model', action='store_true', default=False,
                       help='Train final model on all data after k-fold evaluation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LOTL Detection Model Training (K-Fold Cross-Validation)")
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
    
    # Train with k-fold cross-validation
    results = train_with_kfold(
        filtered_events, labels,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
        use_augmentation=args.use_augmentation,
        use_disagreement_detector=args.use_disagreement_detector,
        use_rf=args.use_rf,
        use_nn=args.use_nn
    )
    
    # Train disagreement detector if requested
    if args.use_disagreement_detector and len(disagreement_events) > 0:
        print("\n" + "="*60)
        print("Training Disagreement Detector (V2)")
        print("="*60)
        
        disagreement_detector = DisagreementDetector(random_state=args.random_seed)
        # Use some agreement events as negative class
        agreement_sample = filtered_events[:len(disagreement_events)]
        disagreement_detector.fit(disagreement_events, agreement_sample)
        
        # Save disagreement detector
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        disagreement_detector.save(str(output_dir / 'disagreement_detector.pkl'))
        print(f"Disagreement detector saved to {output_dir / 'disagreement_detector.pkl'}")
    
    # Train final model on all data if requested
    if args.train_final_model:
        print("\n" + "="*60)
        print("Training Final Model on All Data")
        print("="*60)
        
        # Apply augmentation if requested
        train_events = filtered_events
        train_labels = labels
        if args.use_augmentation:
            print("Applying data augmentation...")
            augmenter = DataAugmenter(augmentation_factor=0.5, random_seed=args.random_seed)
            train_events, train_labels = augmenter.augment_dataset(train_events, train_labels)
        
        # Train final ensemble
        ensemble = LOTLEnsemble(
            use_random_forest=args.use_rf,
            use_neural_network=args.use_nn,
            use_llm_reasoning=False
        )
        
        start_time = time.time()
        ensemble.fit(train_events, train_labels)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ensemble.save(str(output_dir))
        print(f"Final model saved to {output_dir}")
    
    # Save evaluation results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'kfold_evaluation_results.json'
    
    results['config'] = {
        'n_splits': args.n_splits,
        'random_seed': args.random_seed,
        'use_augmentation': args.use_augmentation,
        'use_disagreement_detector': args.use_disagreement_detector,
        'num_events': len(filtered_events),
        'num_disagreements': len(disagreement_events)
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to {results_path}")
    
    # Test explanation generation
    if args.train_final_model:
        print("\n" + "="*60)
        print("Testing Explanation Generation")
        print("="*60)
        
        sample_events = filtered_events[:3]
        explanations = ensemble.predict_with_explanation(sample_events)
        
        for i, (event, expl) in enumerate(zip(sample_events, explanations)):
            print(f"\nSample {i+1}:")
            cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
            print(f"  Command: {cmdline[:100] if cmdline else 'N/A'}...")
            print(f"  Prediction: {expl['prediction']}")
            print(f"  Confidence: {expl['confidence']:.2f}")
            print(f"  Explanation: {expl['explanation'][:200]}...")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
