"""
Main entry point for LOTL detection system.
"""
import argparse
from pathlib import Path

from train import main as train_main
from evaluate import main as evaluate_main


def main():
    parser = argparse.ArgumentParser(
        description='LOTL Attack Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --dataset data.jsonl
  python main.py evaluate --model-dir models
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the detection models')
    train_parser.add_argument('--dataset', type=str, default='data.jsonl',
                             help='Path to dataset file')
    train_parser.add_argument('--output-dir', type=str, default='models',
                             help='Directory to save trained models')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                             help='Fraction of data to use for testing')
    train_parser.add_argument('--random-seed', type=int, default=42,
                             help='Random seed for reproducibility')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('--dataset', type=str, default='data.jsonl',
                            help='Path to dataset file')
    eval_parser.add_argument('--model-dir', type=str, default='models',
                            help='Directory containing trained models')
    eval_parser.add_argument('--test-size', type=float, default=0.2,
                            help='Fraction of data to use for testing')
    eval_parser.add_argument('--random-seed', type=int, default=42,
                            help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_main()
    elif args.command == 'evaluate':
        evaluate_main()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

