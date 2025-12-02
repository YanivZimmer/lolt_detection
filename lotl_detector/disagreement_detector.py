"""
Disagreement Detector Model V2.
Detects cases where Claude and ground truth labels disagree with high confidence.
Based on analysis of disagreement patterns.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from feature_extractor import ComprehensiveFeatureExtractor


class DisagreementDetector:
    """
    Detects events where Claude and ground truth labels disagree.
    These cases often represent edge cases or ambiguous scenarios.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize disagreement detector.
        
        Args:
            n_estimators: Number of trees
            random_state: Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_extractor = ComprehensiveFeatureExtractor()
        self.feature_names = None
        self.is_fitted = False
    
    def _extract_disagreement_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features specific to disagreement detection.
        Based on analysis of disagreement patterns.
        """
        features = {}
        
        # Get base features
        base_features = self.feature_extractor.extract_all_features(event)
        features.update(base_features)
        
        # Additional disagreement-specific features
        claude_response = event.get('claude-sonnet-4-5', {})
        ground_truth = event.get('_label', 'benign')
        attack_technique = event.get('_attack_technique', '')
        
        # Confidence mismatch features
        if claude_response:
            claude_label = claude_response.get('predicted_label', '')
            claude_confidence = claude_response.get('confidence', '')
            claude_reason = claude_response.get('reason', '').lower()
            
            # Label mismatch indicator
            features['label_mismatch'] = 1 if claude_label != ground_truth else 0
            
            # Confidence features
            confidence_map = {'high': 3, 'medium': 2, 'low': 1, '': 0}
            features['claude_confidence_encoded'] = confidence_map.get(claude_confidence.lower(), 0)
            
            # Reason analysis
            features['reason_mentions_standard'] = 1 if any(
                word in claude_reason for word in ['standard', 'normal', 'typical', 'legitimate']
            ) else 0
            
            features['reason_mentions_suspicious'] = 1 if any(
                word in claude_reason for word in ['suspicious', 'unusual', 'anomal', 'attack']
            ) else 0
            
            features['reason_length'] = len(claude_response.get('reason', ''))
        else:
            features['label_mismatch'] = 0
            features['claude_confidence_encoded'] = 0
            features['reason_mentions_standard'] = 0
            features['reason_mentions_suspicious'] = 0
            features['reason_length'] = 0
        
        # Attack technique features
        features['has_attack_technique'] = 1 if attack_technique else 0
        
        # Common disagreement patterns from analysis
        image = event.get('Image', '') or event.get('SourceImage', '') or event.get('image', '')
        parent_image = event.get('ParentImage', '') or event.get('parentImage', '')
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        
        if image and parent_image:
            image_exe = Path(image).name.lower()
            parent_exe = Path(parent_image).name.lower()
            
            # Explorer from userinit (common disagreement case)
            features['disagreement_explorer_userinit'] = 1 if (
                'userinit.exe' in parent_exe and 'explorer.exe' in image_exe
            ) else 0
            
            # System account operations
            user = event.get('User', '') or event.get('user', '')
            if 'SYSTEM' in user.upper() or 'NETWORK SERVICE' in user.upper():
                features['disagreement_system_account'] = 1
            else:
                features['disagreement_system_account'] = 0
        
        # Compression/archiving (data staging disagreement pattern)
        if cmdline:
            cmdline_lower = cmdline.lower()
            features['disagreement_compression'] = 1 if any(
                cmd in cmdline_lower for cmd in ['compress-archive', 'zip', 'archive']
            ) else 0
            
            # Process/network discovery (common in disagreements)
            features['disagreement_discovery'] = 1 if any(
                cmd in cmdline_lower for cmd in ['tasklist', 'net view', 'get-process']
            ) else 0
        else:
            features['disagreement_compression'] = 0
            features['disagreement_discovery'] = 0
        
        return features
    
    def fit(self, disagreement_events: List[Dict[str, Any]], 
            agreement_events: List[Dict[str, Any]]):
        """
        Train the disagreement detector.
        
        Args:
            disagreement_events: Events where Claude and ground truth disagree
            agreement_events: Events where they agree (negative class)
        """
        # Create labels: 1 for disagreement, 0 for agreement
        disagreement_labels = [1] * len(disagreement_events)
        agreement_labels = [0] * len(agreement_events)
        
        all_events = disagreement_events + agreement_events
        all_labels = disagreement_labels + agreement_labels
        
        # Extract features
        print("Extracting disagreement features...")
        feature_list = []
        for event in all_events:
            features = self._extract_disagreement_features(event)
            feature_list.append(features)
        
        # Convert to feature matrix
        self.feature_names = sorted(feature_list[0].keys())
        X = np.array([[features.get(name, 0) for name in self.feature_names] 
                      for features in feature_list])
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(all_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        print(f"Training disagreement detector on {len(all_events)} events...")
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        
        # Feature importance
        importances = self.model.feature_importances_
        top_features = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print("\nTop 10 features for disagreement detection:")
        for feat_name, importance in top_features:
            print(f"  {feat_name}: {importance:.4f}")
    
    def predict(self, events: List[Dict[str, Any]]) -> List[int]:
        """
        Predict disagreement probability.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of predictions (1 = disagreement likely, 0 = agreement likely)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        feature_list = []
        for event in events:
            features = self._extract_disagreement_features(event)
            feature_list.append(features)
        
        # Convert to feature matrix
        X = np.array([[features.get(name, 0) for name in self.feature_names] 
                      for features in feature_list])
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions.tolist()
    
    def predict_proba(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """
        Predict disagreement probabilities.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Probability matrix (n_samples, 2) where [:, 1] is disagreement probability
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        feature_list = []
        for event in events:
            features = self._extract_disagreement_features(event)
            feature_list.append(features)
        
        # Convert to feature matrix
        X = np.array([[features.get(name, 0) for name in self.feature_names] 
                      for features in feature_list])
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, filepath: str):
        """Save model to file."""
        import pickle
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load model from file."""
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True

