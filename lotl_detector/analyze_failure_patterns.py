"""
Analyze where the LOTL detector underperforms or outperforms Claude
and label failure patterns.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from .data_loader import load_dataset, get_labels, sanitize_event_for_inference
from .ensemble import LOTLEnsemble
from .failure_patterns import classify_failure_pattern


def _extract_claude_label(event: Dict[str, Any]) -> str:
    claude_info = event.get("claude-sonnet-4-5", {})
    return claude_info.get("predicted_label", "benign")


def _build_event_snapshot(event: Dict[str, Any], index: int, ground_truth: str) -> Dict[str, Any]:
    return {
        "index": index,
        "ground_truth": ground_truth,
        "command_line": event.get("CommandLine") or event.get("commandLine") or "",
        "image": event.get("Image") or event.get("SourceImage") or "",
        "user": event.get("User") or event.get("AccountName") or "",
        "parent_image": event.get("ParentImage") or event.get("parentImage") or "",
    }


def analyze_failures(dataset: str, model_dir: str, output_dir: str) -> Dict[str, int]:
    events = load_dataset(dataset)
    ensemble = LOTLEnsemble()
    ensemble.load(model_dir)

    sanitized_events = [sanitize_event_for_inference(event) for event in events]
    ground_truth = get_labels(events, use_claude_label=False)
    model_predictions = ensemble.predict(sanitized_events)
    claude_predictions = [_extract_claude_label(event) for event in events]

    less_than_entries: List[Dict[str, Any]] = []
    better_than_entries: List[Dict[str, Any]] = []
    failure_pattern_entries: List[Dict[str, Any]] = []

    for idx, (raw_event, sanitized, gt, model_pred, claude_pred) in enumerate(
        zip(events, sanitized_events, ground_truth, model_predictions, claude_predictions)
    ):
        claude_correct = claude_pred == gt
        model_correct = model_pred == gt
        snapshot = _build_event_snapshot(sanitized, idx, gt)
        snapshot.update(
            {
                "model_prediction": model_pred,
                "claude_prediction": claude_pred,
            }
        )

        if claude_correct and not model_correct:
            less_than_entries.append(snapshot)
            pattern, evidence = classify_failure_pattern(sanitized)
            failure_pattern_entries.append(
                {
                    **snapshot,
                    "failure_pattern": pattern,
                    "evidence": evidence,
                }
            )
        elif model_correct and not claude_correct:
            better_than_entries.append(snapshot)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    less_than_file = output_path / "less_than_cluade.jsonl"
    with open(less_than_file, "w", encoding="utf-8") as f:
        for entry in less_than_entries:
            f.write(json.dumps(entry) + "\n")

    better_than_file = output_path / "better_than_cluade.json"
    with open(better_than_file, "w", encoding="utf-8") as f:
        json.dump(better_than_entries, f, indent=2)

    failure_patterns_file = output_path / "failure_patterns.jsonl"
    with open(failure_patterns_file, "w", encoding="utf-8") as f:
        for entry in failure_pattern_entries:
            f.write(json.dumps(entry) + "\n")

    return {
        "less_than": len(less_than_entries),
        "better_than": len(better_than_entries),
        "labeled_failures": len(failure_pattern_entries),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze failure modes vs Claude.")
    parser.add_argument(
        "--dataset", type=str, default="data.jsonl", help="Relative path to dataset"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models", help="Directory with trained model"
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Where to write analysis files")
    args = parser.parse_args()

    stats = analyze_failures(args.dataset, args.model_dir, args.output_dir)

    print("Analysis complete:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
