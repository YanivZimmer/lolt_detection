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
        self.use_neural_network = use_neural_network
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
        # CR: not using the model, just using the explanation
        if self.use_llm_reasoning:
            #CR: predict_with_explanation returns a tuple of (predicted_label, explanation) but not probabilities 
            #CR: we need to convert the explanation to a probability, but we don't have a model to do this
            #CR: we need to use the explanation to predict the label
            #CR: we need to use the explanation to predict the probability
            #CR: we need to use the explanation to predict the confidence
            #CR: we need to use the explanation to predict the explanation
            #CR: we need to use the explanation to predict the explanation
            llm_preds, llm_explanation = self.llm_distiller.predict_with_explanation(events)
            predictions.append(llm_preds)
            probabilities.append(llm_probs)
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
        Generate human-readable explanation for a prediction.
        
        Args:
            event: Event dictionary
            prediction: Predicted label
            features: Feature vector
            confidence: Prediction confidence
            
        Returns:
            Explanation string
        """
        explanations = []
        
        # Get top contributing features from Random Forest
        if self.use_random_forest and self.random_forest.is_fitted:
            rf_explanations = self.random_forest.explain_prediction(
                features.reshape(1, -1), 
                feature_names=self.feature_names
            )
            if rf_explanations:
                top_features = rf_explanations[0]['top_features'][:3]
                for feat in top_features:
                    if abs(feat['contribution']) > 0.1:
                        explanations.append(
                            f"{feat['name']} (contribution: {feat['contribution']:.2f})"
                        )
        
        # Add LLM-based reasoning if available
        if self.use_llm_reasoning and self.llm_distiller:
            llm_explanation = self.llm_distiller.generate_explanation(event, prediction)
            explanations.append(llm_explanation)
        
        # Add command-line specific details
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        image = event.get('Image', '') or event.get('SourceImage', '') or event.get('image', '')
        
        if prediction == 'malicious':
            if cmdline:
                cmd_lower = cmdline.lower()
                if '-enc' in cmd_lower:
                    explanations.append("Contains encoded PowerShell command")
                if 'bypass' in cmd_lower and 'executionpolicy' in cmd_lower:
                    explanations.append("Attempts to bypass execution policy")
                if '/node:' in cmdline:
                    explanations.append("Attempts remote execution")
            
            if image:
                exe_name = Path(image).name.lower() if image else ''
                if exe_name in ['cmd.exe', 'powershell.exe', 'wmic.exe']:
                    explanations.append(f"Uses native system binary: {exe_name}")
        else:
            explanations.append("No suspicious indicators detected")
            if cmdline:
                cmd_lower = cmdline.lower()
                if any(cmd in cmd_lower for cmd in ['dir', 'cd', 'type', 'echo']):
                    explanations.append("Uses common administrative commands")
        
        # Combine explanations
        if explanations:
            explanation = ". ".join(explanations[:5])  # Limit to top 5
        else:
            explanation = f"Predicted as {prediction} with confidence {confidence:.2f}"
        
        return explanation
    
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

