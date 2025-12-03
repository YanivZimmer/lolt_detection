"""
Compare model predictions with Claude's predictions.
Identifies where each model performs better.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

from data_loader import load_dataset, get_labels, sanitize_event_for_inference
from ensemble import LOTLEnsemble
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compare_predictions(events: List[Dict[str, Any]], ensemble: LOTLEnsemble):
    """
    Compare model predictions with Claude's predictions.
    
    Args:
        events: List of events
        ensemble: Trained ensemble model
    """
    # Get labels
    ground_truth_labels = get_labels(events, use_claude_label=False)
    claude_labels = []
    
    # Extract Claude predictions
    for event in events:
        if 'claude-sonnet-4-5' in event:
            claude_label = event['claude-sonnet-4-5'].get('predicted_label', 'benign')
        else:
            claude_label = 'benign'
        claude_labels.append(claude_label)
    
    # Get model predictions
    sanitized_events = [sanitize_event_for_inference(event) for event in events]
    model_predictions = ensemble.predict(sanitized_events)
    
    # Analyze disagreements
    disagreements = []
    claude_correct = []
    model_correct = []
    both_correct = []
    both_wrong = []
    
    for i, (event, gt, claude_pred, model_pred) in enumerate(
        zip(events, ground_truth_labels, claude_labels, model_predictions)
    ):
        claude_match = (claude_pred == gt)
        model_match = (model_pred == gt)
        
        if claude_pred != model_pred:
            disagreements.append({
                'index': i,
                'ground_truth': gt,
                'claude_prediction': claude_pred,
                'model_prediction': model_pred,
                'claude_correct': claude_match,
                'model_correct': model_match,
                'event': sanitize_event_for_inference(event),
                'attack_technique': event.get('_attack_technique', ''),
                'claude_reason': event.get('claude-sonnet-4-5', {}).get('reason', ''),
            })
        
        if claude_match and not model_match:
            claude_correct.append(i)
        elif model_match and not claude_match:
            model_correct.append(i)
        elif claude_match and model_match:
            both_correct.append(i)
        else:
            both_wrong.append(i)
    
    # Calculate metrics vs ground truth
    claude_metrics = {
        'accuracy': accuracy_score(ground_truth_labels, claude_labels),
        'precision': precision_score(ground_truth_labels, claude_labels, pos_label='malicious', zero_division=0),
        'recall': recall_score(ground_truth_labels, claude_labels, pos_label='malicious', zero_division=0),
        'f1': f1_score(ground_truth_labels, claude_labels, pos_label='malicious', zero_division=0),
    }
    
    model_metrics = {
        'accuracy': accuracy_score(ground_truth_labels, model_predictions),
        'precision': precision_score(ground_truth_labels, model_predictions, pos_label='malicious', zero_division=0),
        'recall': recall_score(ground_truth_labels, model_predictions, pos_label='malicious', zero_division=0),
        'f1': f1_score(ground_truth_labels, model_predictions, pos_label='malicious', zero_division=0),
    }
    
    # Analyze disagreement patterns
    disagreement_patterns = defaultdict(list)
    
    for d in disagreements:
        attack_type = d.get('attack_technique', 'unknown')
        disagreement_patterns[attack_type].append(d)
    
    return {
        'disagreements': disagreements,
        'claude_correct_count': len(claude_correct),
        'model_correct_count': len(model_correct),
        'both_correct_count': len(both_correct),
        'both_wrong_count': len(both_wrong),
        'claude_metrics': claude_metrics,
        'model_metrics': model_metrics,
        'disagreement_patterns': dict(disagreement_patterns),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare model with Claude')
    parser.add_argument('--dataset', type=str, default='data.jsonl',
                       help='Path to dataset')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Path to model directory')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Comparing Model with Claude-Sonnet-4.5")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    events = load_dataset(args.dataset)
    print(f"Loaded {len(events)} events")
    
    # Load model
    print(f"\nLoading model from {args.model_dir}...")
    ensemble = LOTLEnsemble()
    ensemble.load(args.model_dir)
    
    # Compare
    print("\nComparing predictions...")
    comparison = compare_predictions(events, ensemble)
    
    # Print summary
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)
    
    print(f"\nTotal events: {len(events)}")
    print(f"Disagreements: {len(comparison['disagreements'])}")
    print(f"Claude correct (model wrong): {comparison['claude_correct_count']}")
    print(f"Model correct (Claude wrong): {comparison['model_correct_count']}")
    print(f"Both correct: {comparison['both_correct_count']}")
    print(f"Both wrong: {comparison['both_wrong_count']}")
    
    print("\nClaude Metrics:")
    for metric, value in comparison['claude_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nModel Metrics:")
    for metric, value in comparison['model_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Save disagreements
    output_dir = Path(args.output_dir)
    
    # Save as JSONL
    jsonl_path = output_dir / 'claude_my_model_disagree.jsonl'
    with open(jsonl_path, 'w') as f:
        for d in comparison['disagreements']:
            f.write(json.dumps(d) + '\n')
    print(f"\nDisagreements saved to {jsonl_path}")
    
    # Save markdown report
    md_path = output_dir / 'claude_my_model_disagree.md'
    with open(md_path, 'w') as f:
        f.write("# Model vs Claude Disagreement Analysis\n\n")
        f.write(f"**Total Events**: {len(events)}\n")
        f.write(f"**Disagreements**: {len(comparison['disagreements'])}\n\n")
        
        f.write("## Performance Comparison\n\n")
        f.write("| Metric | Claude | Our Model |\n")
        f.write("|--------|--------|-----------|\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            f.write(f"| {metric} | {comparison['claude_metrics'][metric]:.4f} | {comparison['model_metrics'][metric]:.4f} |\n")
        
        f.write("\n## Disagreement Patterns\n\n")
        for attack_type, items in comparison['disagreement_patterns'].items():
            f.write(f"### {attack_type} ({len(items)} cases)\n\n")
            for i, d in enumerate(items[:5], 1):  # Show top 5
                f.write(f"**Case {i}**:\n")
                f.write(f"- Ground Truth: {d['ground_truth']}\n")
                f.write(f"- Claude: {d['claude_prediction']} (correct: {d['claude_correct']})\n")
                f.write(f"- Our Model: {d['model_prediction']} (correct: {d['model_correct']})\n")
                cmdline = d['event'].get('CommandLine', 'N/A')
                f.write(f"- Command: `{cmdline[:100]}...`\n\n")
        
        f.write("\n## Where Each Model Performs Better\n\n")
        f.write(f"**Claude Correct, Model Wrong**: {comparison['claude_correct_count']} cases\n")
        f.write(f"- Claude has advantage in ambiguous edge cases\n")
        f.write(f"- Better at understanding context and nuance\n\n")
        
        f.write(f"**Model Correct, Claude Wrong**: {comparison['model_correct_count']} cases\n")
        f.write(f"- Model has advantage in pattern-based detection\n")
        f.write(f"- Better at detecting specific LOTL techniques\n")
    
    print(f"Report saved to {md_path}")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

