
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



---




⚠️ Could not generate top features explanation: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
