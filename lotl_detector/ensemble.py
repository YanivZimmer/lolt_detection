"""
Ensemble model combining multiple approaches for LOTL detection.
Includes explanation generation for predictions.
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from models import RandomForestModel, SmallNeuralNetwork
from llm_distiller import LLMDistiller
from feature_extractor import ComprehensiveFeatureExtractor


class LOTLEnsemble:
    """
    Ensemble model combining Random Forest, Neural Network, and LLM-based reasoning.
    Provides explainable predictions.
    """
    
    def __init__(self, 
                 use_random_forest: bool = True,
                 use_neural_network: bool = True,
                 use_llm_reasoning: bool = True,
                 ensemble_method: str = 'weighted_vote'):
        """ 
        Initialize ensemble model.
        
        Args:
            use_random_forest: Whether to use Random Forest
            use_neural_network: Whether to use Neural Network
            use_llm_reasoning: Whether to use LLM reasoning features
            ensemble_method: 'weighted_vote', 'average_proba', or 'majority_vote'
        """
        self.use_random_forest = use_random_forest
        self.use_neural_network = False
        self.use_llm_reasoning = False
        self.ensemble_method = ensemble_method
        
        self.random_forest = RandomForestModel() if use_random_forest else None
        self.neural_network = None  # Will be initialized after we know input_dim
        self.llm_distiller = LLMDistiller() if use_llm_reasoning else None
        self.feature_extractor = ComprehensiveFeatureExtractor()
        
        self.is_fitted = False
        self.feature_names = None
    
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
        X = np.array([[features.get(name, 0) for name in self.feature_names] 
                      for features in feature_list])
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        # Train Random Forest
        if self.use_random_forest:
            print("\nTraining Random Forest...")
            self.random_forest.fit(X, labels, feature_names=self.feature_names)
            print("Random Forest trained")
        
        # Train Neural Network
        if self.use_neural_network:
            print("\nTraining Neural Network...")
            input_dim = X.shape[1]
            self.neural_network = SmallNeuralNetwork(input_dim=input_dim)
            self.neural_network.fit(X, labels)
            print("Neural Network trained")
        
        self.is_fitted = True
        print("\nEnsemble training complete!")
    
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
        X = np.array([[features.get(name, 0) for name in self.feature_names] 
                      for features in feature_list])
        
        # Get predictions from each model
        predictions = []
        probabilities = []
        
        if self.use_random_forest:
            rf_preds = self.random_forest.predict(X)
            rf_probs = self.random_forest.predict_proba(X)
            predictions.append(rf_preds)
            probabilities.append(rf_probs)
        
        if self.use_neural_network:
            nn_preds = self.neural_network.predict(X)
            nn_probs = self.neural_network.predict_proba(X)
            predictions.append(nn_preds)
            probabilities.append(nn_probs)
        # Note: LLM reasoning is currently disabled (use_llm_reasoning=False)
        # The LLM distiller is used for explanation generation, not for ensemble voting
        # Future: When student LLM is trained, it can be integrated here
        # Combine predictions
        if self.ensemble_method == 'weighted_vote':
            # Weighted voting based on confidence
            final_predictions = []
            for i in range(len(events)):
                votes = {'malicious': 0.0, 'benign': 0.0}
                
                for j, probs in enumerate(probabilities):
                    pred = predictions[j][i]
                    # Use probability as weight
                    prob = probs[i][1] if pred == 'malicious' else probs[i][0]
                    votes[pred] += prob
                
                final_predictions.append(max(votes, key=votes.get))
            
            return final_predictions
        
        elif self.ensemble_method == 'average_proba':
            # Average probabilities
            avg_probs = np.mean(probabilities, axis=0)
            final_predictions = []
            for probs in avg_probs:
                final_predictions.append('malicious' if probs[1] > probs[0] else 'benign')
            return final_predictions
        
        else:  # majority_vote
            # Simple majority vote
            final_predictions = []
            for i in range(len(events)):
                votes = {'malicious': 0, 'benign': 0}
                for preds in predictions:
                    votes[preds[i]] += 1
                final_predictions.append(max(votes, key=votes.get))
            return final_predictions
    
    def predict_with_explanation(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict labels with explanations.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of dictionaries with 'prediction', 'confidence', and 'explanation'
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
        
        X = np.array([[features.get(name, 0) for name in self.feature_names] 
                      for features in feature_list])
        
        results = []
        
        for i, (event, pred) in enumerate(zip(events, predictions)):
            # Get probabilities
            probs = []
            if self.use_random_forest:
                rf_probs = self.random_forest.predict_proba(X[i:i+1])[0]
                probs.append(rf_probs)
            if self.use_neural_network:
                nn_probs = self.neural_network.predict_proba(X[i:i+1])[0]
                probs.append(nn_probs)
            
            avg_prob = np.mean(probs, axis=0) if probs else np.array([0.5, 0.5])
            confidence = float(max(avg_prob))
            
            # Generate explanation
            explanation = self._generate_explanation(
                event, pred, X[i], confidence
            )
            
            results.append({
                'prediction': pred,
                'confidence': confidence,
                'explanation': explanation
            })
        
        return results
    
    def _generate_explanation(self, event: Dict[str, Any], prediction: str, 
                             features: np.ndarray, confidence: float) -> str:
        """
        Generate human-readable explanation for a prediction using at least 3 features.
        Uses natural language and includes attack type information.
        
        Args:
            event: Event dictionary
            prediction: Predicted label
            features: Feature vector
            confidence: Prediction confidence
            
        Returns:
            Natural language explanation string
        """
        # Get top contributing features from Random Forest
        top_contributing_features = []
        if self.use_random_forest and self.random_forest.is_fitted:
            rf_explanations = self.random_forest.explain_prediction(
                features.reshape(1, -1), 
                feature_names=self.feature_names
            )
            if rf_explanations:
                top_features = rf_explanations[0]['top_features']
                # Get top 5 features with significant contribution
                top_contributing_features = [
                    feat for feat in top_features 
                    if abs(feat['contribution']) > 0.05
                ][:5]
        
        # Extract event details
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        image = event.get('Image', '') or event.get('SourceImage', '') or event.get('image', '')
        parent_image = event.get('ParentImage', '') or event.get('parentImage', '')
        user = event.get('User', '') or event.get('user', '')
        integrity = event.get('IntegrityLevel', '') or event.get('integrityLevel', '')
        
        # Determine attack type from features and event
        attack_type = self._infer_attack_type(event, features)
        
        # Build natural language explanation
        explanation_parts = []
        
        if prediction == 'malicious':
            explanation_parts.append(f"This event is classified as **malicious** with {confidence:.0%} confidence.")
            
            # Use top 3+ features for explanation
            feature_explanations = []
            
            # Feature 1: Process relationship
            if image and parent_image:
                image_exe = Path(image).name.lower() if image else ''
                parent_exe = Path(parent_image).name.lower() if parent_image else ''
                
                # Check for suspicious parent-child
                if any(feat['name'] == 'suspicious_parent_child' for feat in top_contributing_features):
                    feature_explanations.append(
                        f"The process {image_exe} was launched by {parent_exe}, which is an unusual parent-child relationship that often indicates lateral movement or process injection."
                    )
                elif any(feat['name'] == 'explorer_from_userinit' for feat in top_contributing_features):
                    feature_explanations.append(
                        f"Windows Explorer was launched by userinit.exe, which is suspicious because Explorer should typically be executed by the user GUI rather than through command-line initialization - this may indicate lateral movement."
                    )
                elif any(feat['name'] == 'system_binary_from_explorer' for feat in top_contributing_features):
                    feature_explanations.append(
                        f"A system binary ({image_exe}) was launched from Explorer, which is unusual as system binaries are typically launched by system processes rather than user-initiated GUI applications."
                    )
            
            # Feature 2: Command-line analysis
            if cmdline:
                cmd_lower = cmdline.lower()
                
                if any(feat['name'].startswith('obfuscation') for feat in top_contributing_features):
                    if '-enc' in cmd_lower or 'encoded' in cmd_lower:
                        feature_explanations.append(
                            "The command contains encoded PowerShell content, which is a common obfuscation technique used to evade detection and hide malicious payloads."
                        )
                    elif any(char in cmdline for char in ['%', '\\x', '0x']):
                        feature_explanations.append(
                            "The command contains encoding patterns (URL encoding, hex encoding) that suggest obfuscation attempts to hide the true intent of the command."
                        )
                
                if any(feat['name'] == 'has_powershell_bypass' for feat in top_contributing_features):
                    feature_explanations.append(
                        "The command attempts to bypass PowerShell's execution policy, which is a defense evasion technique commonly used by attackers to run unauthorized scripts."
                    )
                
                if any(feat['name'] == 'system_file_modification' for feat in top_contributing_features):
                    feature_explanations.append(
                        "The command attempts to modify critical system files (such as the hosts file), which is a defense evasion technique used to redirect network traffic or block security updates."
                    )
            
            # Feature 3: Execution context
            if any(feat['name'] == 'is_system_user' for feat in top_contributing_features) or 'SYSTEM' in user.upper():
                feature_explanations.append(
                    f"The command was executed by a system account ({user}), which is unusual for interactive commands and may indicate privilege escalation or system-level compromise."
                )
            elif any(feat['name'] == 'system_account_admin_tool' for feat in top_contributing_features):
                feature_explanations.append(
                    f"A system service account ({user}) is using administrative tools, which is anomalous and may indicate unauthorized system-level access."
                )
            
            # Feature 4: Additional indicators
            if any(feat['name'] == 'suspicious_path_operation' for feat in top_contributing_features):
                feature_explanations.append(
                    "The command operates on files in suspicious locations (such as Public Downloads or Temp directories), which are commonly used by attackers for staging malicious payloads."
                )
            
            if any(feat['name'] == 'compression_operation' for feat in top_contributing_features):
                feature_explanations.append(
                    "The command performs compression or archiving operations, which may indicate data staging or exfiltration preparation."
                )
            
            # Ensure we have at least 3 feature explanations
            if len(feature_explanations) < 3:
                # Add generic feature-based explanations
                remaining_features = [f for f in top_contributing_features 
                                    if not any(f['name'] in exp for exp in feature_explanations)][:3-len(feature_explanations)]
                for feat in remaining_features:
                    if feat['contribution'] > 0:
                        feature_explanations.append(
                            f"The {feat['name'].replace('_', ' ')} indicator strongly suggests malicious activity (contribution: {feat['contribution']:.2f})."
                        )
            
            # Add attack type
            if attack_type:
                explanation_parts.append(f"**Attack Type**: {attack_type}")
            
            # Combine feature explanations
            explanation_parts.extend(feature_explanations[:5])  # Use top 5
            
        else:  # benign
            explanation_parts.append(f"This event is classified as **benign** with {confidence:.0%} confidence.")
            
            # Explain why it's benign using features
            benign_indicators = []
            
            if cmdline:
                cmd_lower = cmdline.lower()
                if any(cmd in cmd_lower for cmd in ['dir', 'cd', 'type', 'echo', 'tasklist']):
                    benign_indicators.append(
                        "The command uses standard administrative tools for routine system management tasks."
                    )
            
            if image:
                image_exe = Path(image).name.lower() if image else ''
                if image_exe in ['explorer.exe', 'notepad.exe', 'calc.exe']:
                    benign_indicators.append(
                        f"The process ({image_exe}) is a standard Windows application launched through normal user interaction."
                    )
            
            if 'Medium' in integrity or 'High' in integrity:
                benign_indicators.append(
                    "The process runs with appropriate integrity level for user-initiated activities."
                )
            
            if not benign_indicators:
                benign_indicators.append(
                    "No suspicious indicators were detected in the process execution, command-line arguments, or execution context."
                )
            
            explanation_parts.extend(benign_indicators[:3])
        
        # Combine into natural language
        explanation = " ".join(explanation_parts)
        return explanation
    
    def _infer_attack_type(self, event: Dict[str, Any], features: np.ndarray) -> str:
        """
        Infer attack type from event and features.
        
        Args:
            event: Event dictionary
            features: Feature vector
            
        Returns:
            Attack type string or empty string
        """
        # Map feature names to attack types
        feature_to_attack = {
            'apt_lateral_movement': 'lateral_movement',
            'apt_credential_access': 'credential_access',
            'apt_persistence': 'persistence',
            'apt_defense_evasion': 'defense_evasion',
            'apt_collection': 'collection',
            'apt_exfiltration': 'exfiltration',
            'explorer_from_userinit': 'lateral_movement',
            'system_file_modification': 'defense_evasion',
            'compression_operation': 'data_staging',
            'process_discovery': 'discovery',
            'network_discovery': 'discovery',
        }
        
        # Check features for attack type indicators
        feature_dict = dict(zip(self.feature_names, features))
        
        attack_types = []
        for feat_name, attack_type in feature_to_attack.items():
            if feat_name in feature_dict and feature_dict[feat_name] > 0:
                attack_types.append(attack_type)
        
        # Also check event metadata
        attack_technique = event.get('_attack_technique', '')
        if attack_technique:
            attack_types.append(attack_technique)
        
        # Return most common or first
        if attack_types:
            from collections import Counter
            most_common = Counter(attack_types).most_common(1)
            return most_common[0][0] if most_common else ''
        
        return ''
    
    def save(self, directory: str):
        """
        Save all ensemble components.
        
        Args:
            directory: Directory to save models
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        if self.use_random_forest:
            self.random_forest.save(f"{directory}/random_forest.pkl")
        
        if self.use_neural_network:
            self.neural_network.save(f"{directory}/neural_network.pt")
        
        # Save feature names
        import pickle
        with open(f"{directory}/feature_names.pkl", 'wb') as f:
            pickle.dump(self.feature_names, f)
    
    def load(self, directory: str):
        """
        Load all ensemble components.
        
        Args:
            directory: Directory containing saved models
        """
        if self.use_random_forest:
            self.random_forest = RandomForestModel()
            self.random_forest.load(f"{directory}/random_forest.pkl")
        
        if self.use_neural_network:
            import pickle
            with open(f"{directory}/feature_names.pkl", 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Initialize NN with correct input_dim
            input_dim = len(self.feature_names)
            self.neural_network = SmallNeuralNetwork(input_dim=input_dim)
            self.neural_network.load(f"{directory}/neural_network.pt")
        
        self.is_fitted = True

