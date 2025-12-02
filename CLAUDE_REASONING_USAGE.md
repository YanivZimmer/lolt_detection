# Claude-Sonnet-4-5 Reasoning Usage

## Overview

This document explains how Claude-Sonnet-4-5's reasoning output is used in the LOTL detection system.

## Where Claude Reasoning is Used

### 1. **LLM Distiller (`llm_distiller.py`)**

The primary component that interacts with Claude's reasoning:

- **Extracts reasoning features**: `extract_reasoning_features()` method extracts indicators from Claude's reasoning text:
  - Mentions of obfuscation, suspicious patterns, legitimate activity
  - Confidence levels
  - Reasoning length and complexity
  
- **Prepares training data**: `prepare_training_data()` creates (prompt, response) pairs from Claude's reasoning for LLM distillation

- **Generates explanations**: `generate_explanation()` uses Claude's reasoning patterns to create human-readable explanations

**Note**: Currently, the LLM distiller extracts features and prepares data, but the actual student LLM training is done in the `train_llm_distillation.ipynb` notebook.

### 2. **Feature Extraction (`feature_extractor.py`)**

**NOT directly used** - The feature extractor does NOT use Claude's reasoning output for feature calculation. It only uses production-available Sysmon event fields.

However, **Claude's reasoning insights** were used to **design new features**:
- Explorer from userinit.exe pattern (identified from Claude's reasoning)
- System account admin tool usage
- Suspicious path operations
- System file modifications

These features are **inspired by** Claude's reasoning but calculated from raw event data.

### 3. **Disagreement Detector (`disagreement_detector.py`)**

Uses Claude's reasoning to detect cases where Claude and ground truth disagree:

- Extracts `label_mismatch` indicator
- Uses Claude's confidence and reasoning text
- Analyzes reasoning patterns to identify ambiguous cases

## What is NOT Used

The following are **explicitly excluded** from model training and inference:

- `claude-sonnet-4-5.predicted_label` - Only used as training target, not as feature
- `claude-sonnet-4-5.reason` - Only used for feature design inspiration and explanation generation
- `claude-sonnet-4-5.confidence` - Only used in disagreement detector
- `_label`, `_attack_technique`, `_source`, `prompt` - Metadata fields, never used

## Distillation Process

### Current Implementation

1. **Feature Extraction**: Claude's reasoning is analyzed to extract semantic indicators (obfuscation mentions, suspicious patterns, etc.)

2. **Training Data Preparation**: Claude's (prompt, response) pairs are prepared for student LLM training

3. **Student LLM Training**: Done in `train_llm_distillation.ipynb` notebook:
   - Uses DialoGPT-small as student model
   - Trains on Claude's reasoning patterns
   - Can be used for explanation generation

### Future Improvements

- Fully integrate trained student LLM into ensemble
- Use student LLM for real-time explanation generation
- Fine-tune on domain-specific LOTL patterns

## Summary

**Claude's reasoning is used for**:
- ✅ Training target (predicted_label)
- ✅ Feature design inspiration
- ✅ Explanation generation patterns
- ✅ Disagreement detection
- ✅ LLM distillation training data

**Claude's reasoning is NOT used for**:
- ❌ Direct feature calculation (only insights)
- ❌ Model inference (only training)
- ❌ Production predictions (only explanations)

This ensures the model works in production where Claude's output is not available.

