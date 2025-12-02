"""
Machine learning models for LOTL detection.
Includes Random Forest and Neural Network implementations.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class RandomForestModel:
    """
    Random Forest model for LOTL detection.
    Uses comprehensive feature set including numeric and text features.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 random_state: int = 42, n_jobs: int = -1):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight='balanced'  # Handle class imbalance
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: List[str], feature_names: Optional[List[str]] = None):
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (list of strings: 'malicious' or 'benign')
            feature_names: Optional list of feature names
        """
        self.feature_names = feature_names
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> List[str]:
        """
        Predict labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        y_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting importances")
        
        importances = self.model.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}
    
    def explain_prediction(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Explain predictions using feature importances and SHAP-like analysis.
        
        Args:
            X: Feature matrix (can be single sample or batch)
            feature_names: Optional feature names
            
        Returns:
            List of explanation dictionaries
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explanation")
        
        if feature_names is None:
            feature_names = self.feature_names
        
        X_scaled = self.scaler.transform(X)
        importances = self.model.feature_importances_
        
        explanations = []
        
        for sample in X_scaled:
            # Get top contributing features
            contributions = sample * importances
            top_indices = np.argsort(np.abs(contributions))[-5:][::-1]
            
            explanation = {
                'top_features': [],
                'reasoning': []
            }
            
            for idx in top_indices:
                feature_name = feature_names[idx] if feature_names else f"feature_{idx}"
                value = sample[idx]
                importance = importances[idx]
                contribution = contributions[idx]
                
                explanation['top_features'].append({
                    'name': feature_name,
                    'value': float(value),
                    'importance': float(importance),
                    'contribution': float(contribution)
                })
            
            # Generate reasoning
            top_positive = [f for f in explanation['top_features'] if f['contribution'] > 0][:3]
            top_negative = [f for f in explanation['top_features'] if f['contribution'] < 0][:3]
            
            reasons = []
            if top_positive:
                reasons.append(f"Key indicators: {', '.join([f['name'] for f in top_positive])}")
            if top_negative:
                reasons.append(f"Mitigating factors: {', '.join([f['name'] for f in top_negative])}")
            
            explanation['reasoning'] = reasons
            explanations.append(explanation)
        
        return explanations
    
    def save(self, filepath: str):
        """Save model to file."""
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
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True


class SmallNeuralNetwork:
    """
    Small neural network for LOTL detection.
    Uses PyTorch for training (can be run on Colab).
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout: float = 0.3, random_state: int = 42):
        """
        Initialize neural network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            random_state: Random seed
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def _build_model(self):
        """Build PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            
            class LOTLNet(nn.Module):
                def __init__(self, input_dim, hidden_dims, dropout):
                    super(LOTLNet, self).__init__()
                    
                    layers = []
                    prev_dim = input_dim
                    
                    for hidden_dim in hidden_dims:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout))
                        prev_dim = hidden_dim
                    
                    # Output layer (binary classification)
                    layers.append(nn.Linear(prev_dim, 2))
                    
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            self.model = LOTLNet(self.input_dim, self.hidden_dims, self.dropout)
            return True
        except ImportError:
            print("Warning: PyTorch not available, neural network disabled")
            return False
    
    def fit(self, X: np.ndarray, y: List[str], epochs: int = 50, 
            batch_size: int = 32, learning_rate: float = 0.001):
        """
        Train the neural network.
        
        Args:
            X: Feature matrix
            y: Labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        if not self._build_model():
            raise RuntimeError("PyTorch not available")
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y_encoded)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Train
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> List[str]:
        """Predict labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        import torch
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            y_encoded = predicted.numpy()
        
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        import torch
        import torch.nn.functional as F
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1)
            return probs.numpy()
    
    def save(self, filepath: str):
        """Save model to file."""
        import torch
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        torch.save(model_data, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        import torch
        
        model_data = torch.load(filepath, map_location='cpu')
        
        self.input_dim = model_data['input_dim']
        self.hidden_dims = model_data['hidden_dims']
        self.dropout = model_data['dropout']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        
        self._build_model()
        self.model.load_state_dict(model_data['model_state_dict'])
        self.is_fitted = True


def evaluate_model(y_true: List[str], y_pred: List[str], model_name: str = "Model"):
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for reporting
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*60}\n")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='malicious', zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label='malicious', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='malicious', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['benign', 'malicious'])
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Benign  Malicious")
    print(f"Actual Benign    {cm[0][0]:4d}      {cm[0][1]:4d}")
    print(f"      Malicious  {cm[1][0]:4d}      {cm[1][1]:4d}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=['benign', 'malicious']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

