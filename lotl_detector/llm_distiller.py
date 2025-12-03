"""
Teacher-Student LLM distillation for LOTL detection.
Uses Claude-Sonnet-4.5 reasoning to train a lightweight LLM.
"""
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re


class LLMDistiller:
    """
    Distills Claude-Sonnet-4.5 reasoning into a lightweight LLM.
    Uses the reasoning from Claude to create training examples for a small model.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize the distiller.
        
        Args:
            model_name: Name of the lightweight LLM to use as student
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the student model."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
                
                # Add padding token if it doesn't exist
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            except ImportError:
                print("Warning: transformers not available, LLM distillation disabled")
                return None, None
        return self._model, self._tokenizer
    
    def create_prompt_from_event(self, event: Dict[str, Any]) -> str:
        """
        Create a prompt from event fields (excluding metadata).
        This is the format that would be sent to the LLM.
        
        Args:
            event: Sysmon event dictionary
            
        Returns:
            Formatted prompt string
        """
        # Extract relevant fields (same as what Claude sees)
        event_time = event.get('EventTime', '') or event.get('eventTime', '')
        pid = event.get('ProcessId', '') or event.get('processId', '')
        user = event.get('User', '') or event.get('user', '')
        image = event.get('Image', '') or event.get('SourceImage', '') or event.get('image', '')
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        
        # Build prompt similar to the one shown to Claude
        parts = []
        if event_time:
            parts.append(f"[{event_time}]")
        if pid:
            parts.append(f"PID={pid}")
        if user:
            parts.append(f"User={user}")
        if image:
            parts.append(f"Image={image}")
        if cmdline:
            parts.append(f"CMD={cmdline}")
        
        return " | ".join(parts) if parts else ""
    
    def extract_reasoning_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract reasoning-based features from Claude's response.
        These can be used as additional features for other models.
        
        Args:
            event: Event dictionary with claude-sonnet-4-5 field
            
        Returns:
            Dictionary of reasoning-based features
        """
        features = {}
        
        claude_response = event.get('claude-sonnet-4-5', {})
        if not claude_response:
            return features
        
        reason = claude_response.get('reason', '')
        confidence = claude_response.get('confidence', '')
        attack_technique = claude_response.get('attack_technique', '')
        
        # Extract key indicators from reasoning
        reason_lower = reason.lower()
        
        # Common LOTL indicators mentioned in reasoning
        features['reasoning_mentions_obfuscation'] = 1 if any(
            word in reason_lower for word in ['obfuscat', 'encod', 'hidden', 'stealth']
        ) else 0
        
        features['reasoning_mentions_suspicious'] = 1 if any(
            word in reason_lower for word in ['suspicious', 'unusual', 'anomal', 'abnormal']
        ) else 0
        
        features['reasoning_mentions_legitimate'] = 1 if any(
            word in reason_lower for word in ['legitimate', 'normal', 'typical', 'expected']
        ) else 0
        
        features['reasoning_mentions_system'] = 1 if any(
            word in reason_lower for word in ['system', 'binary', 'native', 'built-in']
        ) else 0
        
        features['reasoning_mentions_network'] = 1 if any(
            word in reason_lower for word in ['network', 'download', 'connect', 'remote']
        ) else 0
        
        features['reasoning_mentions_execution'] = 1 if any(
            word in reason_lower for word in ['execut', 'spawn', 'launch', 'run']
        ) else 0
        
        # Confidence encoding
        confidence_map = {'high': 3, 'medium': 2, 'low': 1, '': 0}
        features['claude_confidence_encoded'] = confidence_map.get(confidence.lower(), 0)
        
        # Reasoning length (longer reasoning might indicate complexity)
        features['reasoning_length'] = len(reason)
        features['reasoning_word_count'] = len(reason.split())
        
        # Attack technique presence
        features['has_attack_technique'] = 1 if attack_technique else 0
        
        return features
    
    def prepare_training_data(self, events: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Prepare training data for student LLM.
        Creates (prompt, response) pairs from Claude's reasoning.
        
        Args:
            events: List of events with Claude responses
            
        Returns:
            List of (prompt, response) tuples
        """
        training_pairs = []
        
        for event in events:
            # Skip if no Claude response
            claude_response = event.get('claude-sonnet-4-5', {})
            if not claude_response:
                continue
            
            # Create prompt from event
            prompt = self.create_prompt_from_event(event)
            if not prompt:
                continue
            
            # Create response from Claude's reasoning
            label = claude_response.get('predicted_label', 'benign')
            reason = claude_response.get('reason', '')
            
            # Format response: label + reasoning
            response = f"Label: {label}. {reason}"
            
            training_pairs.append((prompt, response))
        
        return training_pairs
    
    def generate_explanation(self, event: Dict[str, Any], rf_fe:str, predicted_label: str) -> str:
        """
        Generate an explanation for a prediction using distilled reasoning.
        This is a simplified version that uses pattern matching and heuristics.
        
        Args:
            event: Sysmon event
            predicted_label: Predicted label (malicious/benign)
            
        Returns:
            Human-readable explanation
        """
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        image = event.get('Image', '') or event.get('SourceImage', '') or event.get('image', '')
        user = event.get('User', '') or event.get('user', '')
        integrity = event.get('IntegrityLevel', '') or event.get('integrityLevel', '')
        
        reasons = []
        
        if predicted_label == 'malicious':
            # Explain why it's malicious
            if image:
                exe_name = Path(image).name.lower() if image else ''
                if exe_name in ['cmd.exe', 'powershell.exe', 'wmic.exe']:
                    reasons.append(f"Uses native system binary {exe_name}")
            
            if cmdline:
                cmd_lower = cmdline.lower()
                if '-enc' in cmd_lower or 'encoded' in cmd_lower:
                    reasons.append("Contains encoded commands")
                if 'bypass' in cmd_lower and 'executionpolicy' in cmd_lower:
                    reasons.append("Attempts to bypass execution policy")
                if '/node:' in cmdline:
                    reasons.append("Attempts remote execution")
                if any(pattern in cmdline for pattern in ['%', '\\x', '0x']):
                    reasons.append("Contains obfuscation patterns")
            
            if 'SYSTEM' in user.upper() or 'NT AUTHORITY' in user.upper():
                reasons.append("Executed by system account")
            
            if not reasons:
                reasons.append("Exhibits suspicious behavioral patterns")
        else:
            # Explain why it's benign
            if cmdline:
                cmd_lower = cmdline.lower()
                if any(cmd in cmd_lower for cmd in ['dir', 'cd', 'type', 'echo']):
                    reasons.append("Uses common administrative commands")
            
            if 'Medium' in integrity or 'High' in integrity:
                reasons.append(f"Executed with appropriate integrity level. integrity is {integrity}")
            
            if not reasons:
                reasons.append("No suspicious indicators detected")
        
        explanation = ". ".join(reasons) if reasons else "Standard system activity"

        prompt = f'''You are a LLM that is responsible for generating reasoning for a classification of other modesl. 
        You will recieve 1.json representing an event, 2.label telling you if it is a living of the land atack. then you will write an explanation to a security researcher on why is was classified like that. you can also use 3. and 4. feature based explanations as hints. feature based explanation: 1.{str(event)} 2.{predicted_label} 3.{rf_fe} 4.{explanation}'''
        inputs = self._tokenizer(prompt, return_tensors="pt") # Tokenize the prompt
        outputs = self._model.generate(**inputs, max_new_tokens=50, do_sample=True, num_beams=1)
        generated_text = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return f"{generated_text}."
    
    def predict_with_explanation(self, event: Dict[str, Any]) -> Tuple[str, str]:
        """
        Predict label and generate explanation.
        This is a placeholder - in practice, you'd use the trained student model.
        
        Args:
            event: Sysmon event
            
        Returns:
            Tuple of (predicted_label, explanation)
        """
        # For now, return a simple heuristic-based prediction
        # In production, this would use the trained student model
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        image = event.get('Image', '') or event.get('SourceImage', '') or event.get('image', '')
        
        # Simple heuristic (will be replaced by actual model)
        malicious_indicators = 0
        
        if cmdline:
            cmd_lower = cmdline.lower()
            if '-enc' in cmd_lower or 'encoded' in cmd_lower:
                malicious_indicators += 2
            if 'bypass' in cmd_lower and 'executionpolicy' in cmd_lower:
                malicious_indicators += 2
            if '/node:' in cmdline:
                malicious_indicators += 2
            if any(pattern in cmdline for pattern in ['%', '\\x']):
                malicious_indicators += 1
        
        if image:
            exe_name = Path(image).name.lower() if image else ''
            if exe_name in ['cmd.exe', 'powershell.exe', 'wmic.exe']:
                malicious_indicators += 1
        
        predicted_label = 'malicious' if malicious_indicators >= 2 else 'benign'
        explanation = self.generate_explanation(event, predicted_label)
        
        return predicted_label, explanation

# CR: not training the model, just using the explanation
# CR: not using the model, just using the explanation
# CR: the explanation is not very good and we are not using the model, just using the explanation