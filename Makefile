.PHONY: setup serve train evaluate preprocess clean help

help:
	@echo "LOTL Detection - Available Commands:"
	@echo ""
	@echo "  make setup       - Create environment and install dependencies"
	@echo "  make serve       - Start Chainlit app for local testing"
	@echo "  make train       - Train the detection models"
	@echo "  make evaluate    - Run evaluation on test set"
	@echo "  make preprocess  - Preprocess dataset (filter label agreement)"
	@echo "  make clean       - Clean generated files and models"
	@echo ""

setup:
	@echo "Setting up environment..."
	uv venv
	uv pip install -e .
	@echo "✅ Setup complete!"

serve:
	@echo "Starting Chainlit app..."
	chainlit run lotl_detector/app.py

train:
	@echo "Training LOTL detection models..."
	python lotl_detector/train.py --dataset data.jsonl --output-dir models
	@echo "✅ Training complete!"

train-aug:
	@echo "Training LOTL detection models with augmentation..."
	python lotl_detector/train.py --dataset data.jsonl --output-dir models --use-augmentation
	@echo "✅ Training complete!"

evaluate:
	@echo "Evaluating models..."
	python lotl_detector/train.py --dataset data.jsonl --output-dir models
	@echo "✅ Evaluation complete! Check models/evaluation_results.json"

preprocess:
	@echo "Preprocessing dataset..."
	python -c "from lotl_detector.data_loader import load_dataset, filter_label_agreement; events = load_dataset('data.jsonl'); filtered, _ = filter_label_agreement(events); print(f'Filtered dataset: {len(filtered)} events')"
	@echo "✅ Preprocessing complete!"

explain:
	@echo "Generating feature explanations..."
	python lotl_detector/explain_features.py --model-dir models --data data.jsonl --top-k 10 --output feature_explanations.md
	@echo "✅ Feature explanations generated!"

benchmark:
	@echo "Benchmarking inference time..."
	python lotl_detector/evaluate_inference_time.py --model-dir models --dataset data.jsonl
	@echo "✅ Inference benchmark complete!"

clean:
	@echo "Cleaning generated files..."
	rm -rf lotl_detector/models/*.pkl lotl_detector/models/*.pt lotl_detector/models/*.json
	rm -rf __pycache__ */__pycache__ */*.pyc
	rm -rf .chainlit
	@echo "✅ Clean complete!"

