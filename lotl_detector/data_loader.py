"""
Data loading utilities for LOTL dataset.
"""
import json
from typing import List, Dict, Any, Tuple, Iterator
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    events = []
    #current dir is lotl_detector
    file_path = Path(__file__).parent / file_path
    print(f"Loading dataset from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def split_dataset(events: List[Dict[str, Any]], test_size: float = 0.2, 
                  random_seed: int = 42, stratify: bool = True) -> Tuple[List[Dict], List[Dict], List[int], List[int]]:
    """Split dataset into train and test sets.
    
    Args:
        events: List of event dictionaries
        test_size: Fraction of data to use for testing (default: 0.2)
        random_seed: Random seed for reproducibility (default: 42)
        stratify: If True, maintain class balance in splits (default: True)
    
    Returns:
        Tuple of (train_events, test_events, train_indices, test_indices)
    """
    import random
    
    # Get labels for stratification
    labels = []
    for event in events:
        #if 'claude-sonnet-4-5' in event:
        #    label = event['claude-sonnet-4-5']['predicted_label']#.get('predicted_label', 'benign')
        #else:
        #   label = event.get('_label', 'benign')
        label = event['_label']
        labels.append(label)
    
    # Create indices
    indices = list(range(len(events)))
    
    if stratify:
        # Stratified split to maintain class balance
        from collections import defaultdict
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)
        
        # Shuffle each class separately
        random.seed(random_seed)
        train_indices = []
        test_indices = []
        
        for label, label_indices in label_to_indices.items():
            random.shuffle(label_indices)
            split_idx = int(len(label_indices) * (1 - test_size))
            train_indices.extend(label_indices[:split_idx])
            test_indices.extend(label_indices[split_idx:])
        
        # Shuffle final lists
        random.shuffle(train_indices)
        random.shuffle(test_indices)
    else:
        # Simple random split
        random.seed(random_seed)
        random.shuffle(indices)
        split_idx = int(len(indices) * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
    
    # Create event lists
    train_events = [events[i] for i in train_indices]
    test_events = [events[i] for i in test_indices]
    
    return train_events, test_events, train_indices, test_indices


def get_labels(events: List[Dict[str, Any]], use_claude_label: bool = False) -> List[str]:
    """Extract labels from events.
    
    Args:
        events: List of event dictionaries
        use_claude_label: If True, use Claude's prediction as label, else use ground truth
    """
    labels = []
    for event in events:
        if use_claude_label:
            label = event['claude-sonnet-4-5']['predicted_label']
        else:
            label = event['_label']
        labels.append(label)
    return labels


def filter_label_agreement(events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter events to only include those where Claude's label matches ground truth.
    
    Args:
        events: List of event dictionaries
    
    Returns:
        Tuple of (filtered_events, disagreement_events)
        - filtered_events: Events where Claude and ground truth agree
        - disagreement_events: Events where they disagree
    """
    filtered_events = []
    disagreement_events = []
    
    for event in events:
        ground_truth = event.get('_label', 'benign')
        claude_label = None
        
        if 'claude-sonnet-4-5' in event:
            claude_label = event['claude-sonnet-4-5'].get('predicted_label')
        
        # Only include if both labels exist and match
        if claude_label is not None and ground_truth == claude_label:
            filtered_events.append(event)
        else:
            disagreement_events.append(event)
    
    return filtered_events, disagreement_events


def create_kfold_splits(events: List[Dict[str, Any]], labels: List[str], 
                       n_splits: int = 5, random_seed: int = 42) -> Iterator[Tuple[List[int], List[int]]]:
    """
    Create k-fold cross-validation splits with reproducible folds.
    
    Args:
        events: List of event dictionaries
        labels: List of labels
        n_splits: Number of folds (default: 5)
        random_seed: Random seed for reproducibility
        
    Yields:
        Tuple of (train_indices, test_indices) for each fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # Convert labels to numpy array for sklearn
    labels_array = np.array(labels)
    indices = np.arange(len(events))
    
    for train_idx, test_idx in skf.split(indices, labels_array):
        yield train_idx.tolist(), test_idx.tolist()


def get_kfold_splits(events: List[Dict[str, Any]], labels: List[str],
                    n_splits: int = 5, random_seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """
    Get all k-fold splits as a list.
    
    Args:
        events: List of event dictionaries
        labels: List of labels
        n_splits: Number of folds (default: 5)
        random_seed: Random seed for reproducibility
        
    Returns:
        List of (train_indices, test_indices) tuples for each fold
    """
    return list(create_kfold_splits(events, labels, n_splits, random_seed))


def sanitize_event_for_inference(event: Dict[str, Any]) -> Dict[str, Any]:
    """Remove metadata fields that shouldn't be available during inference.
    
    Only keeps fields that would be available in production:
    - EventID, EventTime, UtcTime, Hostname, Channel
    - SourceImage/Image, CommandLine
    - User, Domain, AccountName, IntegrityLevel
    - ParentImage, ParentProcessId, ProcessId, etc. (standard Sysmon fields)
    
    Removes:
    - _label, _attack_technique, _source (metadata fields)
    - claude-sonnet-4-5 (and all nested fields)
    - prompt (metadata)
    - Any other fields starting with underscore
    
    Args:
        event: Original event dictionary
    
    Returns:
        Sanitized event with only production-available fields
    """
    # Allowed fields (production-available Sysmon fields)
    allowed_fields = {
        # Core event fields
        'EventID', 'EventTime', 'UtcTime', 'Hostname', 'Channel',
        # Process fields
        'Image', 'SourceImage', 'CommandLine', 'ProcessId', 'ProcessGUID',
        'ParentImage', 'ParentProcessId', 'ParentProcessGUID', 'ParentCommandLine',
        # User/Account fields
        'User', 'Domain', 'AccountName', 'IntegrityLevel', 'LogonId',
        # Network fields
        'SourcePort', 'DestinationIp', 'DestinationPort', 'DestinationHostname', 'Protocol',
        # File/Registry fields
        'TargetFilename', 'TargetObject', 'EventType', 'CreationUtcTime',
        # Other standard Sysmon fields
        'CurrentDirectory', 'Hashes', 'Company', 'Description', 'Product', 'Version',
        'OriginalFileName', 'FileVersion', 'Signed', 'Signature', 'SignatureStatus'
    }
    
    # Create sanitized event
    sanitized = {}
    
    for key, value in event.items():
        # Skip metadata fields (starting with _)
        if key.startswith('_'):
            continue
        
        # Skip Claude-related fields
        if key == 'claude-sonnet-4-5' or key.startswith('claude'):
            continue
        
        # Skip prompt field
        if key == 'prompt':
            continue
        
        # Only include allowed fields
        if key in allowed_fields:
            sanitized[key] = value
    
    return sanitized

