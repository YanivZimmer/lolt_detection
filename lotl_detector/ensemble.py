"""
Ensemble model combining multiple approaches for LOTL detection.
Includes explanation generation for predictions.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import traceback
from models import RandomForestModel, SmallNeuralNetwork
from llm_distiller import LLMDistiller
from feature_extractor import ComprehensiveFeatureExtractor

try:
    from shap_explainer import SHAPExplainer

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class LOTLEnsemble:
    """
    Ensemble model combining Random Forest, Neural Network, and LLM-based reasoning.
    Provides explainable predictions.
    """

    def __init__(
        self,
        use_random_forest: bool = True,
        use_neural_network: bool = False,
        use_llm_reasoning: bool = True,
        ensemble_method: str = "weighted_vote",
        top_k_features: Union[int, None] = 1,
    ):
        """
        Initialize ensemble model.

        Args:
            use_random_forest: Whether to use Random Forest
            use_neural_network: Whether to use Neural Network
            use_llm_reasoning: Whether to use LLM reasoning features
            ensemble_method: 'weighted_vote', 'average_proba', or 'majority_vote'
        """
        self.use_random_forest = use_random_forest
        self.use_neural_network = use_neural_network  # NN disabled for inference
        # Whether to enable distilled LLM reasoning for explanations (not classification)
        self.use_llm_reasoning = use_llm_reasoning
        self.ensemble_method = ensemble_method
        self.top_k_features: Optional[int] = top_k_features

        self.random_forest = (
            RandomForestModel(n_estimators=6, max_depth=6, random_state=42)
            if use_random_forest
            else None
        )
        self.neural_network = None  # Will be initialized after we know input_dim
        self.llm_distiller = LLMDistiller() if use_llm_reasoning else None
        self.feature_extractor = ComprehensiveFeatureExtractor()

        self.is_fitted = False
        self.feature_names = None
        self.shap_explainer = None
        self.use_llm_explanation = use_llm_reasoning  # Separate flag for LLM explanation
        # Subset of selected features (used to avoid computing unused expensive features)
        self.selected_feature_names: Optional[List[str]] = None

    def fit(self, events: List[Dict[str, Any]], labels: List[str]):
        """
        Train all ensemble components.

        Args:
            events: List of event dictionaries
            labels: List of labels ('malicious' or 'benign')
        """
        # Extract features
        print("Extracting features...")
        feature_list = []
        for event in events:
            features = self.feature_extractor.extract_all_features(event)
            feature_list.append(features)

        # Convert to feature matrix
        if not feature_list:
            raise ValueError("No features extracted")

        self.feature_names = sorted(feature_list[0].keys())
        X = np.array(
            [[features.get(name, 0) for name in self.feature_names] for features in feature_list]
        )

        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {len(self.feature_names)}")

        # Train Random Forest (optionally with top-K feature selection)
        if self.use_random_forest:
            print("\nTraining Random Forest...")
            self.random_forest.fit(X, labels, feature_names=self.feature_names)
            print("Random Forest trained")

            # Optional top-K feature selection to reduce feature set at inference time
            if self.top_k_features is not None and self.top_k_features > 0:
                print(f"\nSelecting top {self.top_k_features} features by importance...")
                importances = self.random_forest.get_feature_importances()
                # Sort features by importance (descending) and take top-K
                sorted_feats = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
                selected = [name for name, _ in sorted_feats[: self.top_k_features]]

                # Rebuild feature matrix with selected features only
                selected_indices = [self.feature_names.index(name) for name in selected]
                X_selected = X[:, selected_indices]

                # Re-train RF on reduced feature set
                self.random_forest.fit(X_selected, labels, feature_names=selected)

                # Update internal feature list and selection
                self.feature_names = selected
                self.selected_feature_names = selected
                # Inform feature extractor so it can skip unused expensive features
                self.feature_extractor.selected_feature_names = set(selected)

                print(f"Selected features ({len(selected)}): {', '.join(selected)}")

        # Train Neural Network
        if self.use_neural_network:
            print("\nTraining Neural Network...")
            input_dim = X.shape[1]
            self.neural_network = SmallNeuralNetwork(input_dim=input_dim)
            self.neural_network.fit(X, labels)
            print("Neural Network trained")

        self.is_fitted = True

        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE and self.use_random_forest and self.random_forest.is_fitted:
            try:
                self.shap_explainer = SHAPExplainer(self.random_forest, self.feature_names)
            except Exception as e:
                print(f"Warning: Could not initialize SHAP explainer: {e}")
                self.shap_explainer = None

        print("\nEnsemble training complete!")

    # def predict(self, events: List[Dict[str, Any]]) -> List[str]:
    #     """
    #     Predict labels for events using only the Random Forest model.

    #     Args:
    #         events: List of event dictionaries

    #     Returns:
    #         List of predicted labels
    #     """
    #     if not self.is_fitted:
    #         raise ValueError("Model must be fitted before prediction")

    #     # Extract features
    #     feature_list = []
    #     for event in events:
    #         features = self.feature_extractor.extract_all_features(event)
    #         feature_list.append(features)

    #     # Convert to feature matrix
    #     X = np.array([
    #         [features.get(name, 0) for name in self.feature_names]
    #         for features in feature_list
    #     ])

    #     # --- Only Random Forest predictions ---
    #     rf_preds = self.random_forest.predict(X)

    #     # Return predictions directly
    #     return rf_preds.tolist()

    def predict(self, events: List[Dict[str, Any]]) -> List[str]:
        """
        Predict labels for events.

        Args:
            events: List of event dictionaries

        Returns:
            List of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Extract features
        feature_list = []
        for event in events:
            features = self.feature_extractor.extract_all_features(event)
            feature_list.append(features)

        # Convert to feature matrix
        X = np.array(
            [[features.get(name, 0) for name in self.feature_names] for features in feature_list]
        )

        # Only use Random Forest (no ensemble)
        if self.use_random_forest:
            rf_preds = self.random_forest.predict(X)
            return rf_preds.tolist()

        explanation = self.explain_prediction(events)
        print(explanation)
        # Fallback if RF not available
        raise ValueError("Random Forest must be enabled for prediction")

    def predict_with_explanation(
        self, events: List[Dict[str, Any]], use_llm_explanation: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Predict labels with explanations.

        Args:
            events: List of event dictionaries
            use_llm_explanation: Whether to include LLM explanation (default: False)

        Returns:
            List of dictionaries with 'prediction', 'confidence', 'rf_explanation',
            and optionally 'llm_explanation'
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get predictions
        predictions = self.predict(events)

        # Extract features for explanation
        feature_list = []
        for event in events:
            features = self.feature_extractor.extract_all_features(event)
            feature_list.append(features)

        X = np.array(
            [[features.get(name, 0) for name in self.feature_names] for features in feature_list]
        )

        results = []

        for i, (event, pred) in enumerate(zip(events, predictions)):
            # Get probabilities
            probs = []
            if self.use_random_forest:
                rf_probs = self.random_forest.predict_proba(X[i : i + 1])[0]
                probs.append(rf_probs)
            if self.use_neural_network:
                nn_probs = self.neural_network.predict_proba(X[i : i + 1])[0]
                probs.append(nn_probs)

            avg_prob = np.mean(probs, axis=0) if probs else np.array([0.5, 0.5])
            confidence = float(max(avg_prob))

            # Generate RF explanation (always included)
            rf_explanation = self._generate_rf_explanation(event, pred, X[i], confidence)

            # Generate LLM explanation (optional)
            llm_explanation = None
            if use_llm_explanation and self.use_llm_reasoning and self.llm_distiller:
                try:
                    llm_explanation = self.llm_distiller.generate_explanation(
                        event, rf_explanation, pred
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not generate LLM explanation: {e} {traceback.format_exc()}"
                    )

            result = {
                "prediction": pred,
                "confidence": confidence,
                "rf_explanation": rf_explanation,
            }

            if llm_explanation:
                result["llm_explanation"] = llm_explanation

            results.append(result)

        return results

    def _generate_rf_explanation(
        self, event: Dict[str, Any], prediction: str, features: np.ndarray, confidence: float
    ) -> str:
        """
        Generate human-readable explanation using 5+ features with values, SHAP, and natural language.

        Args:
            event: Event dictionary
            prediction: Predicted label
            features: Feature vector
            confidence: Prediction confidence

        Returns:
            Natural language explanation string with 5+ features
        """
        # Build dictionary of raw feature values
        feature_dict = dict(zip(self.feature_names, features))

        # Get SHAP / contribution values for ALL features
        top_features_with_values: List[Dict[str, Any]] = []

        # Use SHAP if available
        if self.shap_explainer and self.shap_explainer.is_available:
            try:
                shap_result = self.shap_explainer.explain_prediction(
                    features.reshape(1, -1), sample_idx=0
                )
                shap_values = shap_result.get("shap_values", {})
                for feat_name in self.feature_names:
                    shap_val = float(shap_values.get(feat_name, 0.0))
                    feat_value = feature_dict.get(feat_name, 0.0)
                    top_features_with_values.append(
                        {
                            "name": feat_name,
                            "value": feat_value,
                            "shap_value": shap_val,
                            "contribution": shap_val,
                        }
                    )
            except Exception as e:
                print(f"Warning: SHAP explanation failed: {e}, falling back to RF importances")
                top_features_with_values = []

        # Fallback to RF feature importance if SHAP not available/failed
        if (not top_features_with_values) and self.use_random_forest:
            rf_explanations = self.random_forest.explain_prediction(
                features.reshape(1, -1),
                feature_names=self.feature_names,
            )
            contrib_lookup = {}
            if rf_explanations:
                for f in rf_explanations[0]["top_features"]:
                    contrib_lookup[f["name"]] = f.get("contribution", 0.0)

            for feat_name in self.feature_names:
                contrib = float(contrib_lookup.get(feat_name, 0.0))
                feat_value = feature_dict.get(feat_name, 0.0)
                top_features_with_values.append(
                    {
                        "name": feat_name,
                        "value": feat_value,
                        "shap_value": contrib,
                        "contribution": contrib,
                    }
                )

        # Final safety: if still empty, create zero entries
        if not top_features_with_values:
            for name in self.feature_names:
                top_features_with_values.append(
                    {
                        "name": name,
                        "value": feature_dict.get(name, 0.0),
                        "shap_value": 0.0,
                        "contribution": 0.0,
                    }
                )

        # Ensure we have at least 5 features (pad if needed)
        while len(top_features_with_values) < 5:
            for name in self.feature_names:
                if name not in [f["name"] for f in top_features_with_values]:
                    top_features_with_values.append(
                        {
                            "name": name,
                            "value": feature_dict.get(name, 0.0),
                            "shap_value": 0.0,
                            "contribution": 0.0,
                        }
                    )
                    break

        # Sort by absolute contribution to highlight strongest signals for narrative,
        # but we will still include ALL features in the structured explanation.
        top_features_with_values.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_5_features = top_features_with_values[:5]

        # Extract event details
        cmdline = event.get("CommandLine", "") or event.get("commandLine", "")
        image = event.get("Image", "") or event.get("SourceImage", "") or event.get("image", "")
        parent_image = event.get("ParentImage", "") or event.get("parentImage", "")
        user = event.get("User", "") or event.get("user", "")

        # Determine attack type using top contributing features only
        attack_type = self._infer_attack_type(top_5_features)

        # Build explanation with 5+ features
        explanation_parts = []

        if prediction == "malicious":
            explanation_parts.append(
                f"This event is classified as **malicious** with {confidence:.0%} confidence."
            )

            if attack_type:
                explanation_parts.append(f"**Attack Type**: {attack_type}")

            explanation_parts.append("\n**Top Contributing Features:**\n")

            # Feature explanations with values
            feature_explanations = []
            for i, feat in enumerate(top_5_features, 1):
                feat_name = feat["name"]
                feat_value = feat["value"]
                contribution = feat["contribution"]

                # Generate natural language explanation for each feature
                feat_explanation = self._explain_feature(feat_name, feat_value, contribution, event)
                if feat_explanation:
                    feature_explanations.append(f"{i}. {feat_explanation}")

            explanation_parts.extend(feature_explanations)
        else:
            explanation_parts.append(
                f"This event is classified as **benign** with {confidence:.0%} confidence."
            )
            explanation_parts.append("\n**Key Indicators (Top Features):**\n")

            # Show top features indicating benign
            for i, feat in enumerate(top_5_features, 1):
                feat_name = feat["name"]
                feat_value = feat["value"]
                feat_explanation = self._explain_feature(
                    feat_name, feat_value, feat["contribution"], event, is_benign=True
                )
                if feat_explanation:
                    explanation_parts.append(f"{i}. {feat_explanation}")

        return "\n".join(explanation_parts)

    def _explain_feature(
        self,
        feat_name: str,
        feat_value: float,
        contribution: float,
        event: Dict[str, Any],
        is_benign: bool = False,
    ) -> str:
        """
        Explain a single feature with its value and contribution.

        Args:
            feat_name: Feature name
            feat_value: Feature value
            contribution: SHAP/contribution value
            event: Event dictionary
            is_benign: Whether explaining benign case

        Returns:
            Natural language explanation of the feature
        """
        # Format value based on type
        if isinstance(feat_value, float):
            if abs(feat_value) < 0.001:
                value_str = "0"
            elif abs(feat_value) < 1:
                value_str = f"{feat_value:.3f}"
            else:
                value_str = f"{feat_value:.2f}"
        else:
            value_str = str(feat_value)

        contribution_str = f"{contribution:.3f}" if abs(contribution) >= 0.001 else "0.000"

        # Generate context-aware explanation
        explanations = []

        # Process relationship features
        if "explorer_from_userinit" in feat_name and feat_value > 0:
            explanations.append(
                f"**Explorer from userinit.exe** (value: {value_str}, SHAP: {contribution_str}): Windows Explorer launched by userinit.exe - suspicious as Explorer should be executed by user GUI, indicating potential lateral movement."
            )

        elif "system_binary_from_explorer" in feat_name and feat_value > 0:
            image = event.get("Image", "") or event.get("SourceImage", "")
            exe_name = Path(image).name if image else "system binary"
            explanations.append(
                f"**System binary from Explorer** (value: {value_str}, SHAP: {contribution_str}): {exe_name} launched from Explorer - unusual as system binaries typically run from system processes."
            )

        elif "suspicious_parent_child" in feat_name and feat_value > 0:
            explanations.append(
                f"**Suspicious parent-child process** (value: {value_str}, SHAP: {contribution_str}): Unusual process relationship indicating potential lateral movement or process injection."
            )

        # Obfuscation features
        elif feat_name.startswith("obfuscation_") and feat_value > 0:
            obf_type = feat_name.replace("obfuscation_", "").replace("_", " ")
            explanations.append(
                f"**{obf_type.title()} obfuscation detected** (value: {value_str}, SHAP: {contribution_str}): Command contains {obf_type} patterns used to hide malicious intent."
            )

        elif "cmdline_entropy" in feat_name:
            if feat_value > 4.5:
                explanations.append(
                    f"**High command-line entropy** (value: {value_str}, SHAP: {contribution_str}): High randomness suggests obfuscation or encoding in the command."
                )
            else:
                explanations.append(
                    f"**Command-line entropy** (value: {value_str}, SHAP: {contribution_str}): Normal entropy level."
                )

        # System account features
        elif "is_system_user" in feat_name and feat_value > 0:
            user = event.get("User", "N/A")
            explanations.append(
                f"**System account execution** (value: {value_str}, SHAP: {contribution_str}): Process running as {user} - unusual for interactive commands, may indicate privilege escalation."
            )

        elif "system_account_admin_tool" in feat_name and feat_value > 0:
            explanations.append(
                f"**System account using admin tools** (value: {value_str}, SHAP: {contribution_str}): System service account using administrative tools - anomalous pattern."
            )

        # Path features
        elif "suspicious_path_operation" in feat_name and feat_value > 0:
            explanations.append(
                f"**Suspicious path operation** (value: {value_str}, SHAP: {contribution_str}): Command operates on files in suspicious locations (Public Downloads, Temp) commonly used for malware staging."
            )

        elif "system_file_modification" in feat_name and feat_value > 0:
            explanations.append(
                f"**System file modification attempt** (value: {value_str}, SHAP: {contribution_str}): Attempts to modify critical system files (e.g., hosts file) - defense evasion technique."
            )

        # Compression features
        elif "compression_operation" in feat_name and feat_value > 0:
            explanations.append(
                f"**Compression/archiving operation** (value: {value_str}, SHAP: {contribution_str}): File compression detected - may indicate data staging or exfiltration preparation."
            )

        # Discovery features
        elif "process_discovery" in feat_name and feat_value > 0:
            explanations.append(
                f"**Process discovery command** (value: {value_str}, SHAP: {contribution_str}): Command lists running processes - common in reconnaissance phase."
            )

        elif "network_discovery" in feat_name and feat_value > 0:
            explanations.append(
                f"**Network discovery command** (value: {value_str}, SHAP: {contribution_str}): Command enumerates network resources - reconnaissance activity."
            )

        # APT features
        elif feat_name.startswith("apt_") and feat_value > 0:
            apt_type = feat_name.replace("apt_", "").replace("_", " ")
            explanations.append(
                f"**APT {apt_type} indicator** (value: {value_str}, SHAP: {contribution_str}): Detected pattern associated with {apt_type} techniques."
            )

        # Native binary features
        elif "native_binary_abuse_score" in feat_name and feat_value > 0:
            explanations.append(
                f"**Native binary abuse score** (value: {value_str}, SHAP: {contribution_str}): High score indicates abuse of legitimate Windows tools for malicious purposes."
            )

        # Text embedding features
        elif feat_name.startswith("text_embedding_"):
            if "mean" in feat_name:
                explanations.append(
                    f"**Text embedding mean** (value: {value_str}, SHAP: {contribution_str}): Overall semantic similarity to malicious patterns in training data."
                )
            elif "std" in feat_name:
                explanations.append(
                    f"**Text embedding std** (value: {value_str}, SHAP: {contribution_str}): Variability in semantic meaning - high values indicate complex/mixed content."
                )
            elif "dim_" in feat_name:
                dim_num = feat_name.split("_")[-1]
                explanations.append(
                    f"**Text embedding dimension {dim_num}** (value: {value_str}, SHAP: {contribution_str}): Semantic feature from sentence transformer capturing command/path/context meaning."
                )

        # Generic fallback
        if not explanations:
            direction = (
                "suggests malicious activity"
                if contribution > 0 and not is_benign
                else "indicates benign activity"
            )
            explanations.append(
                f"**{feat_name.replace('_', ' ').title()}** (value: {value_str}, SHAP: {contribution_str}): This feature {direction}."
            )

        return explanations[0] if explanations else ""

    def _infer_attack_type(self, top_features: List[Dict[str, Any]]) -> str:
        """
        Infer attack type using weighted contributions from top features.

        Args:
            top_features: List of dicts with 'name' and 'contribution'

        Returns:
            Attack type string or 'attack type uncertain'
        """
        if not top_features:
            return "attack type uncertain"

        from collections import defaultdict

        # Map feature names (or substrings) to high-level attack types
        feature_to_attack = {
            "apt_lateral_movement": "lateral_movement",
            "lateral_movement": "lateral_movement",
            "apt_credential_access": "credential_access",
            "credential": "credential_access",
            "apt_persistence": "persistence",
            "persistence": "persistence",
            "apt_defense_evasion": "defense_evasion",
            "defense_evasion": "defense_evasion",
            "apt_collection": "collection",
            "collection": "collection",
            "apt_exfiltration": "exfiltration",
            "exfiltration": "exfiltration",
            "explorer_from_userinit": "lateral_movement",
            "system_file_modification": "defense_evasion",
            "compression_operation": "data_staging",
            "data_staging": "data_staging",
            "process_discovery": "discovery",
            "network_discovery": "discovery",
            "discovery": "discovery",
        }

        attack_scores = defaultdict(float)

        for feature in top_features[:5]:
            feat_name = feature.get("name", "")
            contribution = float(feature.get("contribution", 0.0))
            if not feat_name or contribution == 0:
                continue

            matched_attack = None
            if feat_name in feature_to_attack:
                matched_attack = feature_to_attack[feat_name]
            else:
                # Fallback to substring matching (handles feature variants)
                lower_name = feat_name.lower()
                for key, attack in feature_to_attack.items():
                    if key in lower_name:
                        matched_attack = attack
                        break

            if matched_attack:
                attack_scores[matched_attack] += abs(contribution)

        if not attack_scores:
            return "attack type uncertain"

        # Return attack type with highest total contribution
        sorted_attacks = sorted(attack_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_attacks[0][0] if sorted_attacks else "attack type uncertain"

    def save(self, directory: str):
        """
        Save all ensemble components.

        Args:
            directory: Directory to save models
        """
        Path(directory).mkdir(parents=True, exist_ok=True)

        if self.use_random_forest:
            self.random_forest.save(f"{directory}/random_forest.pkl")

        # Save feature names (selected feature subset after top-K, if applied)
        import pickle

        with open(f"{directory}/feature_names.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)

    def load(self, directory: str):
        """
        Load all ensemble components.

        Args:
            directory: Directory containing saved models
        """
        # Load Random Forest model
        if self.use_random_forest:
            self.random_forest = RandomForestModel()
            self.random_forest.load(f"{directory}/random_forest.pkl")

        # Load feature names (used for RF + explanations + selected feature subset)
        import pickle

        feature_names_path = Path(directory) / "feature_names.pkl"
        if feature_names_path.exists():
            with open(feature_names_path, "rb") as f:
                self.feature_names = pickle.load(f)
            # Selected feature subset is exactly the saved feature list
            self.selected_feature_names = list(self.feature_names)
            # Inform feature extractor so it can avoid computing unused expensive features
            self.feature_extractor.selected_feature_names = set(self.feature_names)

        # Initialize SHAP explainer if available
        if (
            SHAP_AVAILABLE
            and self.use_random_forest
            and self.random_forest.is_fitted
            and self.feature_names
        ):
            try:
                self.shap_explainer = SHAPExplainer(self.random_forest, self.feature_names)
            except Exception as e:
                print(f"Warning: Could not initialize SHAP explainer during load: {e}")
                self.shap_explainer = None

        self.is_fitted = True
