"""
Advanced obfuscation detection based on research papers.
Implements NLP-based detection for command-line obfuscation.
"""
import re
import math
from typing import Dict, List, Tuple
from collections import Counter


class ObfuscationDetector:
    """Detects command-line obfuscation using multiple techniques."""
    
    # Obfuscation patterns from research
    OBFUSCATION_PATTERNS = {
        'url_encoding': [
            r'%[0-9a-fA-F]{2}',  # %XX encoding
            r'%u[0-9a-fA-F]{4}',  # %uXXXX encoding
        ],
        'hex_encoding': [
            r'\\x[0-9a-fA-F]{2}',  # \xXX
            r'0x[0-9a-fA-F]+',  # 0x hex
        ],
        'base64_encoding': [
            r'[A-Za-z0-9+/]{20,}={0,2}',  # Base64-like strings
        ],
        'unicode_encoding': [
            r'\\u[0-9a-fA-F]{4}',  # \uXXXX
            r'&#x[0-9a-fA-F]+;',  # HTML entity
        ],
        'case_manipulation': [
            r'[A-Z]{3,}',  # All caps (suspicious)
            r'[a-z]{3,}[A-Z]{3,}',  # Mixed case patterns
        ],
        'whitespace_manipulation': [
            r'\s{3,}',  # Multiple spaces
            r'\t+',  # Tabs
        ],
        'string_concatenation': [
            r'\+["\']',  # String concatenation
            r'["\']\s*\+\s*["\']',  # Multiple concatenations
        ],
        'environment_variables': [
            r'%[A-Z_]+%',  # %VAR% expansion
            r'\$\{[A-Z_]+\}',  # ${VAR}
        ],
        'command_chaining': [
            r'&&',  # Command chaining
            r'\|\|',  # OR chaining
            r'`.*`',  # Backtick execution
        ],
        'powershell_obfuscation': [
            r'-enc\s+[A-Za-z0-9+/=]+',  # Encoded PowerShell
            r'-e\s+[A-Za-z0-9+/=]+',  # Short form
            r'\[char\]',  # Character casting
            r'\.Replace\(',  # String replacement
        ],
        'javascript_obfuscation': [
            r'javascript:',  # JS protocol
            r'eval\(',  # Eval function
            r'String\.fromCharCode\(',  # Char code obfuscation
        ],
        'cmd_obfuscation': [
            r'cmd\s+/c\s+/c',  # Double /c
            r'^[^a-zA-Z0-9\s/\\-]+',  # Starts with special chars
        ],
    }
    
    # Suspicious character patterns
    SUSPICIOUS_CHARS = {
        'high_entropy': ['$', '@', '#', '!', '^', '&', '*'],
        'encoding_indicators': ['%', '\\', 'x', 'u', '0'],
    }
    
    def __init__(self):
        self.compiled_patterns = {}
        for category, patterns in self.OBFUSCATION_PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def detect_obfuscation(self, command_line: str) -> Dict[str, float]:
        """Detect obfuscation in command line.
        
        Returns:
            Dictionary with obfuscation scores and features
        """
        if not command_line:
            return self._empty_features()
        
        cmd_lower = command_line.lower()
        features = {}
        
        # Pattern-based detection
        obfuscation_score = 0.0
        detected_patterns = []
        
        for category, patterns in self.compiled_patterns.items():
            matches = sum(1 for pattern in patterns if pattern.search(command_line))
            if matches > 0:
                detected_patterns.append(category)
                obfuscation_score += matches * 0.1
                features[f'obfuscation_{category}'] = 1
            else:
                features[f'obfuscation_{category}'] = 0
        
        # Entropy-based detection
        entropy = self._calculate_entropy(command_line)
        features['cmdline_entropy'] = entropy
        features['high_entropy'] = 1 if entropy > 4.5 else 0
        
        # Length and complexity
        features['cmdline_length'] = len(command_line)
        features['cmdline_ratio_alphanumeric'] = self._ratio_alphanumeric(command_line)
        features['cmdline_ratio_special'] = self._ratio_special_chars(command_line)
        
        # Suspicious character patterns
        features['num_suspicious_chars'] = sum(
            command_line.count(char) for char in self.SUSPICIOUS_CHARS['high_entropy']
        )
        
        # Encoding indicators
        features['encoding_indicators'] = sum(
            command_line.count(char) for char in self.SUSPICIOUS_CHARS['encoding_indicators']
        )
        
        # Nested quotes and brackets (common in obfuscation)
        features['nested_quotes'] = self._count_nested_quotes(command_line)
        features['nested_brackets'] = self._count_nested_brackets(command_line)
        
        # Command chaining complexity
        features['command_chains'] = (
            command_line.count('&&') + 
            command_line.count('||') + 
            command_line.count(';') +
            command_line.count('|')
        )
        
        # PowerShell-specific obfuscation
        if 'powershell' in cmd_lower or 'pwsh' in cmd_lower:
            features['powershell_obfuscation'] = self._detect_powershell_obfuscation(command_line)
        else:
            features['powershell_obfuscation'] = 0
        
        # Overall obfuscation score
        features['obfuscation_score'] = min(obfuscation_score + (entropy / 10.0), 1.0)
        features['num_obfuscation_patterns'] = len(detected_patterns)
        
        # Normalized features
        features['obfuscation_intensity'] = min(
            (len(detected_patterns) * 0.2 + entropy / 5.0), 1.0
        )
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy."""
        if not text:
            return 0.0
        entropy = 0.0
        char_counts = Counter(text)
        text_len = len(text)
        for count in char_counts.values():
            p = count / text_len
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def _ratio_alphanumeric(self, text: str) -> float:
        """Ratio of alphanumeric characters."""
        if not text:
            return 0.0
        alnum = sum(1 for c in text if c.isalnum())
        return alnum / len(text)
    
    def _ratio_special_chars(self, text: str) -> float:
        """Ratio of special characters."""
        if not text:
            return 0.0
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special / len(text)
    
    def _count_nested_quotes(self, text: str) -> int:
        """Count nested quote patterns."""
        single_quotes = text.count("'")
        double_quotes = text.count('"')
        backticks = text.count('`')
        return max(0, (single_quotes + double_quotes + backticks) - 2)
    
    def _count_nested_brackets(self, text: str) -> int:
        """Count nested bracket patterns."""
        parens = abs(text.count('(') - text.count(')'))
        brackets = abs(text.count('[') - text.count(']'))
        braces = abs(text.count('{') - text.count('}'))
        return parens + brackets + braces
    
    def _detect_powershell_obfuscation(self, command_line: str) -> int:
        """Detect PowerShell-specific obfuscation."""
        score = 0
        cmd_lower = command_line.lower()
        
        # Encoded commands
        if re.search(r'-enc\s+[A-Za-z0-9+/=]{20,}', command_line):
            score += 2
        
        # Character casting
        if '[char]' in cmd_lower or 'String.fromCharCode' in command_line:
            score += 1
        
        # String replacement chains
        if cmd_lower.count('.replace(') > 2:
            score += 1
        
        # IEX (Invoke-Expression) usage
        if 'iex' in cmd_lower or 'invoke-expression' in cmd_lower:
            score += 1
        
        # Download and execute patterns
        if 'downloadstring' in cmd_lower or 'downloadfile' in cmd_lower:
            score += 1
        
        return min(score, 5)  # Cap at 5
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature set."""
        features = {}
        for category in self.OBFUSCATION_PATTERNS.keys():
            features[f'obfuscation_{category}'] = 0
        features.update({
            'cmdline_entropy': 0.0,
            'high_entropy': 0,
            'cmdline_length': 0,
            'cmdline_ratio_alphanumeric': 0.0,
            'cmdline_ratio_special': 0.0,
            'num_suspicious_chars': 0,
            'encoding_indicators': 0,
            'nested_quotes': 0,
            'nested_brackets': 0,
            'command_chains': 0,
            'powershell_obfuscation': 0,
            'obfuscation_score': 0.0,
            'num_obfuscation_patterns': 0,
            'obfuscation_intensity': 0.0,
        })
        return features

