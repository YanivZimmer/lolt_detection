# Critical Fix: Evaluation vs App Performance Mismatch

## Problem Identified

The evaluation showed ~100% accuracy but the app gave poor results. This was caused by a **data processing mismatch**:

- **During Training/Evaluation**: Used **unsanitized events** (with metadata fields like `_label`, `claude-sonnet-4-5`, `prompt`)
- **During App Inference**: Used **sanitized events** (metadata fields removed via `sanitize_event_for_inference`)

This created a **distribution shift** where the model was trained on one data format but tested on another.

## Root Cause

The feature extractor should only use production-available Sysmon fields, but:
1. Training saw events with metadata that could leak information
2. Evaluation also saw those same metadata fields
3. App saw sanitized events without metadata
4. This mismatch caused poor generalization

## Fix Applied

**All events are now sanitized during training and evaluation** to match production conditions:

1. **In `train.py`**:
   - K-fold training: Events sanitized before training and evaluation
   - Final model training: Events sanitized before training
   - Explanation testing: Events sanitized before testing

2. **In `ensemble.py`**:
   - `predict()` method simplified to use only Random Forest (no ensemble)
   - Removed all ensemble voting logic

3. **Consistency**:
   - Training, evaluation, and app all use the same `sanitize_event_for_inference()` function
   - This ensures the model sees the same data distribution in all scenarios

## Changes Made

### `lotl_detector/train.py`
- Added `sanitize_event_for_inference` import
- Sanitize events before k-fold training
- Sanitize events before final model training
- Sanitize events before explanation testing

### `lotl_detector/ensemble.py`
- Simplified `predict()` to use only Random Forest
- Removed ensemble voting logic (no longer needed)

## Expected Impact

- **Evaluation metrics will now reflect real-world performance**
- **App predictions should match evaluation results**
- **Model trained on production-like data from the start**

## Verification Steps

1. Retrain the model: `make train`
2. Check evaluation metrics (should be more realistic, not ~100%)
3. Test in app: `make serve`
4. Compare app predictions with evaluation results (should match)

## Important Note

This fix ensures that:
- The model never sees metadata fields during training
- Evaluation uses the same data format as production
- Performance metrics are realistic and trustworthy

