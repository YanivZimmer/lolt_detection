"""
Data augmentation for LOTL detection training.
Creates variations of events to improve model robustness.
"""

import random
from typing import Dict, Any, List, Tuple
import re


class DataAugmenter:
    """
    Augments training data by creating variations of events.
    Helps model generalize to edge cases.
    """

    def __init__(self, augmentation_factor: float = 0.5, random_seed: int = 42):
        """
        Initialize data augmenter.

        Args:
            augmentation_factor: Fraction of data to augment (0.0 to 1.0)
            random_seed: Random seed for reproducibility
        """
        self.augmentation_factor = augmentation_factor
        random.seed(random_seed)

    def augment_event(self, event: Dict[str, Any]) -> Dict[Any, Any]:
        """
        Create an augmented version of an event.

        Args:
            event: Original event dictionary

        Returns:
            Augmented event dictionary
        """
        augmented = event.copy()

        # Random augmentation techniques
        technique = random.choice(
            [
                "case_variation",
                "path_variation",
                "whitespace_variation",
                "parameter_reorder",
            ]
        )

        cmdline = event.get("CommandLine", "") or event.get("commandLine", "")

        if technique == "case_variation" and cmdline:
            # Vary case of command (some commands are case-insensitive)
            if random.random() < 0.3:
                # Randomly capitalize some parts
                words = cmdline.split()
                augmented_cmdline = " ".join(
                    word.upper() if random.random() < 0.2 else word for word in words
                )
                if "CommandLine" in augmented:
                    augmented["CommandLine"] = augmented_cmdline
                if "commandLine" in augmented:
                    augmented["commandLine"] = augmented_cmdline

        elif technique == "path_variation":
            # Vary path separators or case
            image = event.get("Image", "") or event.get("SourceImage", "")
            if image:
                # Sometimes use forward slashes instead of backslashes
                if random.random() < 0.1:
                    varied_image = image.replace("\\", "/")
                    if "Image" in augmented:
                        augmented["Image"] = varied_image
                    if "SourceImage" in augmented:
                        augmented["SourceImage"] = varied_image

        elif technique == "whitespace_variation" and cmdline:
            # Add/remove whitespace
            if random.random() < 0.3:
                # Add extra spaces occasionally
                augmented_cmdline = re.sub(r"\s+", " ", cmdline)
                if random.random() < 0.1:
                    # Add trailing space
                    augmented_cmdline += " "
                if "CommandLine" in augmented:
                    augmented["CommandLine"] = augmented_cmdline
                if "commandLine" in augmented:
                    augmented["commandLine"] = augmented_cmdline

        elif technique == "parameter_reorder" and cmdline:
            # Reorder parameters (for commands where order doesn't matter)
            # This is conservative - only for safe commands
            safe_commands = ["dir", "type", "echo"]
            if any(cmd in cmdline.lower() for cmd in safe_commands):
                # Simple parameter shuffling (conservative)
                pass  # Skip for now to avoid breaking commands

        return augmented

    def augment_dataset(
        self, events: List[Dict[str, Any]], labels: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Augment a dataset.

        Args:
            events: List of event dictionaries
            labels: List of labels

        Returns:
            Tuple of (augmented_events, augmented_labels)
        """
        num_to_augment = int(len(events) * self.augmentation_factor)
        indices_to_augment = random.sample(range(len(events)), num_to_augment)

        augmented_events = events.copy()
        augmented_labels = labels.copy()

        for idx in indices_to_augment:
            augmented_event = self.augment_event(events[idx])
            augmented_events.append(augmented_event)
            augmented_labels.append(labels[idx])

        return augmented_events, augmented_labels


# Fix import
from typing import Tuple
