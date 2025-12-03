"""
SHAP-based explainability for LOTL detection models.
"""

import numpy as np
import shap
from typing import Dict, Any, List, Tuple
from models import RandomForestModel


class SHAPExplainer:
    """SHAP explainer for model interpretability."""

    def __init__(self, model: RandomForestModel, feature_names: List[str]):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained RandomForestModel
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names

        # Create SHAP explainer
        try:
            # Use TreeExplainer for Random Forest
            self.explainer = shap.TreeExplainer(model.model)
            self.is_available = True
        except Exception as e:
            print(f"Warning: SHAP not available: {e}")
            self.explainer = None
            self.is_available = False

    def explain_prediction(self, X: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.

        Args:
            X: Feature matrix (n_samples, n_features)
            sample_idx: Index of sample to explain

        Returns:
            Dictionary with SHAP values and explanation
        """
        if not self.is_available:
            return {"error": "SHAP not available"}

        # Get SHAP values
        shap_values = self.explainer.shap_values(X[sample_idx : sample_idx + 1])

        # Handle binary classification (shap_values is list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        shap_values = shap_values[0]  # Get single sample

        # Get feature importance - ensure shap_values is 1D array
        if len(shap_values.shape) > 1:
            shap_values = shap_values.flatten()

        feature_importance = list(zip(self.feature_names, shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        # Handle expected_value
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[1] if len(base_value) > 1 else base_value[0])
        else:
            base_value = float(base_value)

        return {
            "shap_values": dict(zip(self.feature_names, shap_values)),
            "top_features": feature_importance[:10],
            "base_value": base_value,
        }

    def get_feature_importance(self, X: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Get top K most important features using SHAP.

        Args:
            X: Feature matrix
            top_k: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if not self.is_available:
            return []

        # Calculate SHAP values for all samples
        shap_values = self.explainer.shap_values(X)

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Get top K features
        feature_importance = list(zip(self.feature_names, mean_abs_shap))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        return feature_importance[:top_k]
