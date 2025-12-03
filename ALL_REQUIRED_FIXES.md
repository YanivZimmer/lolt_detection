# All Required Fixes and Improvements

## Critical Bugs (Must Fix)

### 1. ✅ JSON Parsing Error in App
**Status**: Fixed - Updated example with proper escaping

### 2. ⏳ Augmentation Flag Bug
**Issue**: `make use-augmentation` doesn't work
**Fix**: Added `train-aug` target, but also need to fix the flag usage

### 3. ✅ Always Use _label Field
**Status**: Fixed - Updated to use `use_claude_label=False` by default

### 4. ⏳ Augmentation for LLM Distillation
**Issue**: Notebook doesn't use augmentation
**Fix**: Need to add augmentation option to notebook

## Required Improvements

### Explanation Improvements
- [ ] Always include 4+ top contributing features (currently variable)
- [ ] Include feature values (not just names)
- [ ] Add SHAP-based explanations
- [ ] Show 2 explanations in app (RF + LLM)
- [ ] Add LLM explanation flag (default: False)

### New Options
- [ ] Top K feature selection flag
- [ ] Performance comparison with Claude
- [ ] Avg latency in inference report
- [ ] Default include all data (add exclude flag)
- [ ] Top 50 SHAP features in evaluation report
- [ ] Ensemble default to Forest only

### Documentation
- [ ] Explain text_embedding features
- [ ] Add make explain command (✅ done)
- [ ] Update all MD files

## Implementation Plan

1. Fix remaining bugs
2. Improve explanation generation (4+ features, values, SHAP)
3. Update app to show 2 explanations
4. Add all missing flags
5. Update evaluation reports
6. Update documentation

