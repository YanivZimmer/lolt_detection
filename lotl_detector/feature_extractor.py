"""
Comprehensive feature extractor combining numeric and text features for LOTL detection.
Integrates survivalism features, obfuscation detection, and text embeddings.
"""
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

from survivalism_features import SurvivalismFeatureExtractor
from obfuscation_detector import ObfuscationDetector


class ComprehensiveFeatureExtractor:
    """
    Extracts comprehensive features for LOTL attack detection.
    Combines:
    - Survivalism/LOTL behavioral features
    - Obfuscation detection features
    - Text embedding features (lightweight)
    - Sysmon-specific features
    """
    
    def __init__(self, use_text_embeddings: bool = True, embedding_model: Optional[str] = None):
        """
        Initialize feature extractor.
        
        Args:
            use_text_embeddings: Whether to use text embeddings (default: True)
            embedding_model: Name of embedding model to use (default: 'sentence-transformers/all-MiniLM-L6-v2')
        """
        self.survivalism_extractor = SurvivalismFeatureExtractor()
        self.obfuscation_detector = ObfuscationDetector()
        self.use_text_embeddings = use_text_embeddings
        
        # Use lightweight embedding model
        if embedding_model is None:
            embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
        self.embedding_model_name = embedding_model
        self._embedding_model = None
        
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and self.use_text_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                print("Warning: sentence-transformers not available, skipping text embeddings")
                self.use_text_embeddings = False
        return self._embedding_model
    
    def extract_all_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all features from a Sysmon event.
        
        Args:
            event: Sysmon event dictionary
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # 1. Survivalism/LOTL behavioral features
        survivalism_features = self.survivalism_extractor.extract_survivalism_features(event)
        features.update(survivalism_features)
        
        # 2. Obfuscation detection features
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        obfuscation_features = self.obfuscation_detector.detect_obfuscation(cmdline)
        features.update(obfuscation_features)
        
        # 3. Sysmon-specific numeric features
        sysmon_features = self._extract_sysmon_features(event)
        features.update(sysmon_features)
        
        # 4. Text embedding features (lightweight)
        if self.use_text_embeddings:
            text_features = self._extract_text_embedding_features(event)
            features.update(text_features)
        
        # 5. Command-line specific features
        cmdline_features = self._extract_cmdline_features(event)
        features.update(cmdline_features)
        
        # 6. Process relationship features
        process_features = self._extract_process_features(event)
        features.update(process_features)
        
        return features
    
    def _extract_sysmon_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numeric features from Sysmon event fields."""
        features = {}
        
        # Event ID
        event_id = event.get('EventID', 0)
        features['event_id'] = event_id
        features['is_process_create'] = 1 if event_id == 1 else 0
        features['is_network_connection'] = 1 if event_id == 3 else 0
        features['is_file_create'] = 1 if event_id == 11 else 0
        
        # Integrity level encoding
        integrity = event.get('IntegrityLevel', '').lower()
        integrity_map = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'system': 4
        }
        features['integrity_level_encoded'] = integrity_map.get(integrity, 0)
        features['is_low_integrity'] = 1 if 'low' in integrity else 0
        features['is_high_integrity'] = 1 if 'high' in integrity or 'system' in integrity else 0
        
        # User/Domain features
        user = event.get('User', '') or event.get('user', '')
        domain = event.get('Domain', '') or event.get('domain', '')
        account = event.get('AccountName', '') or event.get('accountName', '')
        
        features['is_system_user'] = 1 if 'SYSTEM' in user.upper() or 'NT AUTHORITY' in user.upper() else 0
        features['is_network_service'] = 1 if 'NETWORK SERVICE' in user.upper() else 0
        features['has_domain'] = 1 if domain else 0
        features['user_length'] = len(user)
        features['domain_length'] = len(domain)
        
        # Process ID features
        pid = event.get('ProcessId', 0) or event.get('processId', 0)
        parent_pid = event.get('ParentProcessId', 0) or event.get('parentProcessId', 0)
        features['process_id'] = pid if pid else 0
        features['parent_process_id'] = parent_pid if parent_pid else 0
        features['has_parent'] = 1 if parent_pid else 0
        
        # Time features (if available)
        event_time = event.get('EventTime', '') or event.get('eventTime', '')
        if event_time:
            try:
                # Extract hour from timestamp
                hour_match = re.search(r'(\d{2}):\d{2}:\d{2}', event_time)
                if hour_match:
                    hour = int(hour_match.group(1))
                    features['event_hour'] = hour
                    features['is_off_hours'] = 1 if hour < 6 or hour > 22 else 0
                else:
                    features['event_hour'] = 12
                    features['is_off_hours'] = 0
            except:
                features['event_hour'] = 12
                features['is_off_hours'] = 0
        else:
            features['event_hour'] = 12
            features['is_off_hours'] = 0
        
        # Network features
        features['has_network_activity'] = 1 if (
            event.get('SourcePort') or event.get('DestinationIp') or 
            event.get('DestinationPort') or event.get('Protocol')
        ) else 0
        features['source_port'] = event.get('SourcePort', 0) or 0
        features['destination_port'] = event.get('DestinationPort', 0) or 0
        
        return features
    
    def _extract_cmdline_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract command-line specific features."""
        features = {}
        
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        if not cmdline:
            cmdline = ''
        
        # Basic command-line statistics
        features['cmdline_length'] = len(cmdline)
        features['cmdline_word_count'] = len(cmdline.split())
        features['cmdline_char_count'] = len(cmdline)
        
        # Path features
        image = event.get('Image', '') or event.get('SourceImage', '') or event.get('image', '')
        parent_image = event.get('ParentImage', '') or event.get('parentImage', '')
        
        if image:
            image_lower = image.lower()
            features['is_system32_binary'] = 1 if 'system32' in image_lower else 0
            features['is_syswow64_binary'] = 1 if 'syswow64' in image_lower else 0
            features['is_windows_binary'] = 1 if 'windows' in image_lower else 0
            features['is_temp_binary'] = 1 if 'temp' in image_lower or 'appdata' in image_lower else 0
            features['image_path_depth'] = len(Path(image).parts) if image else 0
        else:
            features['is_system32_binary'] = 0
            features['is_syswow64_binary'] = 0
            features['is_windows_binary'] = 0
            features['is_temp_binary'] = 0
            features['image_path_depth'] = 0
        
        if parent_image:
            parent_lower = parent_image.lower()
            features['parent_is_explorer'] = 1 if 'explorer.exe' in parent_lower else 0
            features['parent_is_system'] = 1 if 'system' in parent_lower else 0
            features['parent_is_svchost'] = 1 if 'svchost.exe' in parent_lower else 0
        else:
            features['parent_is_explorer'] = 0
            features['parent_is_system'] = 0
            features['parent_is_svchost'] = 0
        
        # Command-line pattern features
        cmdline_lower = cmdline.lower()
        
        # Common LOTL indicators
        features['has_redirection'] = 1 if any(c in cmdline for c in ['>', '<', '>>']) else 0
        features['has_piping'] = 1 if '|' in cmdline else 0
        features['has_chaining'] = 1 if '&&' in cmdline or '||' in cmdline else 0
        features['has_encoded_content'] = 1 if any(pattern in cmdline_lower for pattern in ['-enc', 'encoded', '%', '\\x']) else 0
        features['has_quotes'] = 1 if '"' in cmdline or "'" in cmdline else 0
        features['num_quotes'] = cmdline.count('"') + cmdline.count("'")
        features['num_slashes'] = cmdline.count('/') + cmdline.count('\\')
        features['num_dashes'] = cmdline.count('-')
        features['num_equals'] = cmdline.count('=')
        
        # PowerShell specific
        features['is_powershell'] = 1 if 'powershell' in cmdline_lower or 'pwsh' in cmdline_lower else 0
        features['has_powershell_bypass'] = 1 if 'bypass' in cmdline_lower and 'executionpolicy' in cmdline_lower else 0
        
        # Network-related commands
        features['has_network_cmd'] = 1 if any(cmd in cmdline_lower for cmd in [
            'curl', 'wget', 'invoke-webrequest', 'download', 'ftp', 'net use', 'bitsadmin'
        ]) else 0
        
        # File operations
        features['has_file_ops'] = 1 if any(cmd in cmdline_lower for cmd in [
            'copy', 'move', 'del', 'rm', 'type', 'cat', 'findstr', 'xcopy', 'robocopy'
        ]) else 0
        
        # Registry operations
        features['has_registry_ops'] = 1 if any(cmd in cmdline_lower for cmd in [
            'reg add', 'reg delete', 'reg query', 'reg save', 'reg export'
        ]) else 0
        
        # Service operations
        features['has_service_ops'] = 1 if any(cmd in cmdline_lower for cmd in [
            'sc create', 'sc start', 'sc stop', 'sc config', 'net start', 'net stop'
        ]) else 0
        
        return features
    
    def _extract_process_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract process relationship and context features."""
        features = {}
        
        image = event.get('Image', '') or event.get('SourceImage', '') or event.get('image', '')
        parent_image = event.get('ParentImage', '') or event.get('parentImage', '')
        parent_cmdline = event.get('ParentCommandLine', '') or event.get('parentCommandLine', '')
        
        # Process name extraction
        if image:
            exe_name = Path(image).name.lower()
            features['exe_name_length'] = len(exe_name)
            features['exe_is_common_binary'] = 1 if exe_name in [
                'cmd.exe', 'powershell.exe', 'pwsh.exe', 'wmic.exe', 'rundll32.exe',
                'reg.exe', 'sc.exe', 'net.exe', 'explorer.exe', 'svchost.exe'
            ] else 0
        else:
            features['exe_name_length'] = 0
            features['exe_is_common_binary'] = 0
        
        if parent_image:
            parent_exe_name = Path(parent_image).name.lower()
            features['parent_exe_name_length'] = len(parent_exe_name)
        else:
            features['parent_exe_name_length'] = 0
        
        # Parent-child relationship analysis
        if image and parent_image:
            image_exe = Path(image).name.lower()
            parent_exe = Path(parent_image).name.lower()
            
            # Suspicious parent-child combinations
            suspicious_combos = [
                ('explorer.exe', 'cmd.exe'),
                ('explorer.exe', 'powershell.exe'),
                ('svchost.exe', 'cmd.exe'),
                ('winlogon.exe', 'cmd.exe'),
                ('services.exe', 'cmd.exe'),
            ]
            
            features['suspicious_parent_child'] = 0
            for parent_pattern, child_pattern in suspicious_combos:
                if parent_pattern in parent_exe and child_pattern in image_exe:
                    features['suspicious_parent_child'] = 1
                    break
        
        # Command-line similarity (if parent command line available)
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        if cmdline and parent_cmdline:
            # Simple similarity: check if parent cmdline is substring
            features['parent_cmdline_similarity'] = 1 if parent_cmdline.lower() in cmdline.lower() else 0
        else:
            features['parent_cmdline_similarity'] = 0
        
        return features
    
    def _extract_text_embedding_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text embedding features using lightweight sentence transformer.
        Returns a reduced-dimension embedding (e.g., PCA or mean pooling).
        """
        features = {}
        
        model = self._get_embedding_model()
        if not model:
            return features
        
        # Construct text representation from event
        text_parts = []
        
        # Command line (most important)
        cmdline = event.get('CommandLine', '') or event.get('commandLine', '')
        if cmdline:
            text_parts.append(f"Command: {cmdline}")
        
        # Image path
        image = event.get('Image', '') or event.get('SourceImage', '') or event.get('image', '')
        if image:
            text_parts.append(f"Image: {image}")
        
        # Parent image
        parent_image = event.get('ParentImage', '') or event.get('parentImage', '')
        if parent_image:
            text_parts.append(f"Parent: {parent_image}")
        
        # User context
        user = event.get('User', '') or event.get('user', '')
        if user:
            text_parts.append(f"User: {user}")
        
        # Combine into single text
        combined_text = " | ".join(text_parts) if text_parts else ""
        
        if combined_text:
            try:
                # Get embedding (384 dimensions for all-MiniLM-L6-v2)
                embedding = model.encode(combined_text, convert_to_numpy=True)
                
                # Reduce dimensionality by taking statistics
                # Use mean, std, min, max of embedding dimensions
                features['text_embedding_mean'] = float(np.mean(embedding))
                features['text_embedding_std'] = float(np.std(embedding))
                features['text_embedding_min'] = float(np.min(embedding))
                features['text_embedding_max'] = float(np.max(embedding))
                
                # Use first 10 dimensions as features (for manageable feature count)
                for i in range(min(10, len(embedding))):
                    features[f'text_embedding_dim_{i}'] = float(embedding[i])
                
            except Exception as e:
                print(f"Warning: Failed to generate text embedding: {e}")
                # Return zero features
                features['text_embedding_mean'] = 0.0
                features['text_embedding_std'] = 0.0
                features['text_embedding_min'] = 0.0
                features['text_embedding_max'] = 0.0
                for i in range(10):
                    features[f'text_embedding_dim_{i}'] = 0.0
        else:
            # Empty text
            features['text_embedding_mean'] = 0.0
            features['text_embedding_std'] = 0.0
            features['text_embedding_min'] = 0.0
            features['text_embedding_max'] = 0.0
            for i in range(10):
                features[f'text_embedding_dim_{i}'] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that will be extracted.
        Useful for model initialization and feature selection.
        """
        # Create a dummy event to extract feature names
        dummy_event = {
            'EventID': 1,
            'CommandLine': 'cmd.exe /c dir',
            'Image': 'C:\\Windows\\System32\\cmd.exe',
            'User': 'CORP\\user',
            'IntegrityLevel': 'Medium',
        }
        
        features = self.extract_all_features(dummy_event)
        return sorted(features.keys())

