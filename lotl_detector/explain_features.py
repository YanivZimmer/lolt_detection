"""
Feature explanation script.
Explains what each feature represents and their importance.
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from shap_explainer import SHAPExplainer
from models import RandomForestModel
from feature_extractor import ComprehensiveFeatureExtractor


FEATURE_DESCRIPTIONS = {
    # Survivalism features
    'is_native_binary': 'Indicates if the process is a native Windows binary commonly used in LOTL attacks',
    'native_binary_abuse_score': 'Score indicating likelihood of native binary abuse (0-10)',
    'apt_lateral_movement': 'Indicator of lateral movement techniques',
    'apt_credential_access': 'Indicator of credential access attempts',
    'apt_persistence': 'Indicator of persistence mechanisms',
    'apt_defense_evasion': 'Indicator of defense evasion techniques',
    'apt_collection': 'Indicator of data collection activities',
    'apt_exfiltration': 'Indicator of data exfiltration attempts',
    
    # Obfuscation features
    'obfuscation_score': 'Overall obfuscation score (0-1)',
    'cmdline_entropy': 'Shannon entropy of command line (high = more random/obfuscated)',
    'high_entropy': 'Binary indicator of high entropy (>4.5)',
    'obfuscation_url_encoding': 'Presence of URL encoding patterns',
    'obfuscation_hex_encoding': 'Presence of hexadecimal encoding',
    'obfuscation_base64_encoding': 'Presence of Base64 encoding',
    'obfuscation_powershell_obfuscation': 'PowerShell-specific obfuscation patterns',
    
    # Text embedding features
    'text_embedding_mean': 'Mean value across all embedding dimensions (semantic similarity)',
    'text_embedding_std': 'Standard deviation of embedding (captures variability in semantic meaning)',
    'text_embedding_min': 'Minimum embedding value (lowest semantic component)',
    'text_embedding_max': 'Maximum embedding value (highest semantic component)',
    'text_embedding_dim_0': 'First dimension of semantic embedding (captures primary semantic signal)',
    'text_embedding_dim_1': 'Second dimension of semantic embedding',
    'text_embedding_dim_2': 'Third dimension of semantic embedding',
    'text_embedding_dim_3': 'Fourth dimension of semantic embedding',
    'text_embedding_dim_4': 'Fifth dimension of semantic embedding',
    'text_embedding_dim_5': 'Sixth dimension of semantic embedding',
    'text_embedding_dim_6': 'Seventh dimension of semantic embedding',
    'text_embedding_dim_7': 'Eighth dimension of semantic embedding',
    'text_embedding_dim_8': 'Ninth dimension of semantic embedding',
    'text_embedding_dim_9': 'Tenth dimension of semantic embedding',
    
    # Process features
    'explorer_from_userinit': 'Explorer launched by userinit.exe (lateral movement indicator)',
    'system_binary_from_explorer': 'System binary launched from Explorer (unusual)',
    'suspicious_parent_child': 'Suspicious parent-child process combination',
    
    # Command-line features
    'suspicious_path_operation': 'Operations in suspicious paths (Public Downloads, Temp)',
    'system_file_modification': 'Attempts to modify system files',
    'compression_operation': 'Compression/archiving operations (data staging)',
    'process_discovery': 'Process discovery commands',
    'network_discovery': 'Network discovery commands',
    
    # Sysmon features
    'is_system_user': 'Process running as SYSTEM or NT AUTHORITY account',
    'is_high_integrity': 'Process running with high or system integrity level',
    'execution_context_risk': 'Overall risk score based on execution context',
}


def explain_text_embeddings():
    """
    Explain what text embedding features represent.
    
    Text embeddings are generated using sentence-transformers (all-MiniLM-L6-v2),
    which converts text into a 384-dimensional vector space where similar meanings
    have similar vectors.
    """
    explanation = """
# Text Embedding Features Explanation

## Overview
Text embedding features capture the semantic meaning of event text (command line, 
image path, parent process, user context) in a vector space. Similar events have 
similar embedding values.

## Feature Dimensions

### Statistical Features (4 features)
- **text_embedding_mean**: Average value across all 384 dimensions
  - Captures overall semantic similarity to training data
  - Higher values = more similar to malicious patterns
  
- **text_embedding_std**: Standard deviation across dimensions
  - Measures variability in semantic meaning
  - High std = complex/mixed semantic content
  
- **text_embedding_min/max**: Extreme values
  - Capture the range of semantic signals present

### Dimensional Features (10 features)
- **text_embedding_dim_0 through dim_9**: First 10 dimensions of the 384-D embedding
  - Each dimension captures different aspects of semantic meaning
  - Dim 0: Primary semantic signal (most important)
  - Dim 1-9: Secondary semantic signals
  - These are the most informative dimensions for classification

## How They Work
The sentence-transformer model processes the combined text:
- Command line
- Image path
- Parent process
- User context

And generates a 384-dimensional vector. We use:
1. Statistical summaries (mean, std, min, max) - 4 features
2. First 10 dimensions - 10 features
Total: 14 text embedding features

## Interpretation
- High embedding values often indicate similarity to malicious patterns seen in training
- Low values indicate benign or unusual patterns
- The combination of all dimensions captures the full semantic context
"""
    return explanation


def explain_top_features(model_path: str, data_path: str, top_k: int = 10):
    """
    Explain top K features using SHAP values.
    
    Args:
        model_path: Path to trained model
        data_path: Path to dataset
        top_k: Number of top features to explain
    """
    # Load model - handle both absolute and relative paths
    model_dir = Path(model_path)
    if not model_dir.is_absolute():
        # Try relative to current file location
        base_dir = Path(__file__).parent.parent
        model_dir = base_dir / model_path
    
    rf_model_path = model_dir / 'random_forest.pkl'
    if not rf_model_path.exists():
        # Try alternative locations
        alt_paths = [
            Path('models') / 'random_forest.pkl',
            Path('lotl_detector') / 'models' / 'random_forest.pkl',
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                rf_model_path = alt_path
                model_dir = rf_model_path.parent
                break
        else:
            raise FileNotFoundError(f"Model not found at {model_dir / 'random_forest.pkl'}")
    
    rf_model = RandomForestModel()
    rf_model.load(str(rf_model_path))
    
    # Load feature names
    feature_names_path = model_dir / 'feature_names.pkl'
    if not feature_names_path.exists():
        # Fallback: extract from model or use default
        if rf_model.feature_names:
            feature_names = rf_model.feature_names
        else:
            raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
    else:
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
    
    # Load data for SHAP calculation
    from data_loader import load_dataset
    # Handle path relative to current file
    data_path_full = Path(data_path)
    if not data_path_full.is_absolute():
        base_dir = Path(__file__).parent
        data_path_full = base_dir / data_path
    
    if not data_path_full.exists():
        # Try alternative locations
        alt_paths = [
            Path('lotl_detector') / data_path,
            Path('data.jsonl'),
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                data_path_full = alt_path
                break
    
    if data_path_full.exists():
        events = load_dataset(str(data_path_full))
        
        # Extract features
        feature_extractor = ComprehensiveFeatureExtractor()
        feature_list = []
        for event in events[:100]:  # Use subset for faster SHAP calculation
            features = feature_extractor.extract_all_features(event)
            feature_list.append(features)
        
        X = np.array([[features.get(name, 0) for name in feature_names] 
                      for features in feature_list])
    else:
        raise FileNotFoundError(f"Dataset not found at {data_path_full}")
    
    # Get SHAP feature importance
    shap_explainer = SHAPExplainer(rf_model, feature_names)
    feature_importance = shap_explainer.get_feature_importance(X, top_k=top_k)
    
    # Generate explanation
    explanation = f"# Top {top_k} Most Important Features\n\n"
    explanation += "Based on SHAP values (mean absolute SHAP across all samples):\n\n"
    
    for i, (feat_name, importance) in enumerate(feature_importance, 1):
        description = FEATURE_DESCRIPTIONS.get(feat_name, 'Feature description not available')
        explanation += f"{i}. **{feat_name}** (SHAP: {importance:.4f})\n"
        explanation += f"   - {description}\n\n"
    
    return explanation


def main():
    parser = argparse.ArgumentParser(description='Explain features')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Path to model directory')
    parser.add_argument('--data', type=str, default='data.jsonl',
                       help='Path to dataset')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top features to explain')
    parser.add_argument('--output', type=str, default='feature_explanations.md',
                       help='Output file for explanations')
    
    args = parser.parse_args()
    
    explanations = []
    
    # Explain text embeddings
    explanations.append(explain_text_embeddings())
    explanations.append("\n\n---\n\n")
    
    # Explain top features
    try:
        top_features_explanation = explain_top_features(
            args.model_dir, args.data, args.top_k
        )
        explanations.append(top_features_explanation)
    except FileNotFoundError as e:
        explanations.append(f"\n\n⚠️ Could not generate top features explanation: Model files not found.\n")
        explanations.append(f"Please train the model first using `make train`.\n")
        explanations.append(f"Error details: {e}\n")
    except Exception as e:
        explanations.append(f"\n\n⚠️ Could not generate top features explanation: {e}\n")
    
    # Write to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write('\n'.join(explanations))
    
    print(f"Feature explanations written to {output_path}")


if __name__ == '__main__':
    import numpy as np
    main()

