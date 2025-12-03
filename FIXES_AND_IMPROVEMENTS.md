# Fixes and Improvements Summary

This document tracks all bugs to fix and improvements to implement.

## Bugs Fixed ✅

1. ✅ Makefile augmentation flag - Added `train-aug` target
2. ✅ JSON parsing in app - Fixed backslash escaping in example
3. ✅ Always use `_label` field - Updated train.py to use `use_claude_label=False` by default
4. ✅ Default include all data - Added `--exclude-disagreements` flag (default: include all)

## Bugs To Fix

1. ⏳ Augmentation for LLM distillation - Need to add to notebook
2. ⏳ Ensemble default to Forest only - Updated default in ensemble.py

## Improvements To Implement

### New Options
1. ⏳ Top K feature selection flag
2. ⏳ Performance comparison with Claude
3. ⏳ Avg latency in performance report  
4. ⏳ Always 4+ features in explanation
5. ⏳ Top 50 SHAP features in evaluation report

### Explainability
1. ⏳ SHAP-based explainability
2. ⏳ Distilled LLM explanation in app (2 explanations)
3. ⏳ LLM explanation flag
4. ⏳ RF explanation with 3+ features and values
5. ⏳ make explain command
6. ⏳ Explain text_embedding features
7. ⏳ Show top K features in explain

This document will be updated as fixes are implemented.

