"""
Survivalism-based features inspired by Barr-Smith et al. 2021.
Implements behavioral analysis of native system binary usage patterns.
"""

import re
from typing import Dict, Any, List
from pathlib import Path
from collections import Counter


class SurvivalismFeatureExtractor:
    """
    Extracts features based on survivalism/LOTL behavioral patterns.

    Based on: "Survivalism: Systematic Analysis of Windows Malware Living-Off-The-Land"
    Key insights:
    1. LOTL techniques prevalent in APT malware (26.26% occurrence)
    2. Behavioral analysis of native system binary usage patterns
    3. Anomaly detection in legitimate tool execution
    4. System binary abuse patterns
    """

    # Native Windows binaries commonly abused (from survivalism research)
    NATIVE_BINARIES = {
        "cmd.exe",
        "powershell.exe",
        "pwsh.exe",
        "wmic.exe",
        "rundll32.exe",
        "reg.exe",
        "sc.exe",
        "net.exe",
        "nltest.exe",
        "takeown.exe",
        "icacls.exe",
        "certutil.exe",
        "bitsadmin.exe",
        "wevtutil.exe",
        "schtasks.exe",
        "at.exe",
        "wscript.exe",
        "cscript.exe",
        "mshta.exe",
        "msbuild.exe",
        "csc.exe",
        "installutil.exe",
        "msxsl.exe",
        "forfiles.exe",
        "where.exe",
        "whoami.exe",
        "systeminfo.exe",
        "netstat.exe",
        "tasklist.exe",
        "taskkill.exe",
    }

    # APT-specific patterns (26.26% of APT malware uses LOTL)
    APT_INDICATORS = {
        "lateral_movement": ["/node:", "Enter-PSSession", "Invoke-Command", "psexec"],
        "credential_access": ["cmdkey", "reg save", "mimikatz", "sekurlsa"],
        "persistence": ["schtasks", "sc create", "reg add", "wmic process"],
        "defense_evasion": ["wevtutil", "Clear-EventLog", "attrib +h", "icacls"],
        "collection": ["findstr", "type", "copy", "robocopy", "xcopy"],
        "exfiltration": ["bitsadmin", "certutil", "ftp", "net use"],
    }

    # Behavioral anomaly patterns
    BEHAVIORAL_ANOMALIES = {
        "unusual_binary_combination": [
            ("explorer.exe", "cmd.exe"),
            ("explorer.exe", "powershell.exe"),
            ("svchost.exe", "services.exe"),
            ("winlogon.exe", "cmd.exe"),
        ],
        "unusual_execution_context": [
            "system account with interactive tools",
            "network service with admin tools",
            "low integrity with system binaries",
        ],
        "unusual_timing": [
            "off-hours execution",
            "rapid sequential execution",
            "batch execution patterns",
        ],
    }

    def __init__(self):
        self.binary_usage_patterns = {}

    def extract_survivalism_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract survivalism/LOTL behavioral features."""
        features = {}

        # 1. Native binary abuse detection
        features.update(self._detect_native_binary_abuse(event))

        # 2. APT pattern detection (26.26% of APT uses LOTL)
        features.update(self._detect_apt_patterns(event))

        # 3. Behavioral anomaly detection
        features.update(self._detect_behavioral_anomalies(event))

        # 4. System binary usage patterns
        features.update(self._analyze_binary_usage_patterns(event))

        # 5. Execution context anomalies
        features.update(self._analyze_execution_context(event))

        return features

    def _detect_native_binary_abuse(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Detect abuse of native Windows binaries."""
        features = {}

        image = (event.get("Image") or event.get("SourceImage", "")).lower()
        exe_name = Path(image).name if image else ""

        # Check if it's a native binary
        is_native = exe_name in self.NATIVE_BINARIES
        features["is_native_binary"] = 1 if is_native else 0

        if is_native:
            # Native binary abuse indicators
            cmdline = event.get("CommandLine", "").lower()
            user = event.get("User", "")
            parent = event.get("ParentImage", "").lower()

            # Abuse score based on context
            abuse_score = 0

            # System account using native binary
            if user in ["NT AUTHORITY\\SYSTEM", "NT AUTHORITY\\NETWORK SERVICE"]:
                abuse_score += 2

            # Explorer spawning native binary (highly suspicious)
            if "explorer.exe" in parent:
                abuse_score += 3

            # Native binary with obfuscated command
            if cmdline and len(cmdline) > 50:
                if any(pattern in cmdline for pattern in ["-enc", "encoded", "%", "\\x"]):
                    abuse_score += 2

            # Native binary with network activity
            if event.get("DestinationIp") or event.get("SourcePort"):
                abuse_score += 1

            features["native_binary_abuse_score"] = min(abuse_score, 10)
            features["native_binary_high_risk"] = 1 if abuse_score >= 5 else 0
        else:
            features["native_binary_abuse_score"] = 0
            features["native_binary_high_risk"] = 0

        return features

    def _detect_apt_patterns(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Detect APT-specific LOTL patterns (26.26% of APT uses LOTL)."""
        features = {}

        cmdline = event.get("CommandLine", "").lower()

        # Count APT technique indicators
        apt_score = 0
        detected_techniques = []

        for technique, indicators in self.APT_INDICATORS.items():
            count = sum(1 for indicator in indicators if indicator in cmdline)
            if count > 0:
                detected_techniques.append(technique)
                apt_score += count
                features[f"apt_{technique}"] = 1
            else:
                features[f"apt_{technique}"] = 0

        features["apt_lotl_score"] = min(apt_score, 10)
        features["apt_lotl_detected"] = 1 if apt_score > 0 else 0
        features["num_apt_techniques"] = len(detected_techniques)

        # High-confidence APT indicator (multiple techniques)
        features["high_confidence_apt"] = 1 if len(detected_techniques) >= 2 else 0

        return features

    def _detect_behavioral_anomalies(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Detect behavioral anomalies in system binary usage."""
        features = {}

        parent = event.get("ParentImage", "").lower()
        image = (event.get("Image") or event.get("SourceImage", "")).lower()
        parent_exe = Path(parent).name if parent else ""
        exe_name = Path(image).name if image else ""
        user = event.get("User", "")

        # Unusual binary combinations
        unusual_combos = self.BEHAVIORAL_ANOMALIES["unusual_binary_combination"]
        features["unusual_binary_combo"] = 0

        for parent_pattern, child_pattern in unusual_combos:
            if parent_pattern in parent_exe and child_pattern in exe_name:
                features["unusual_binary_combo"] = 1
                break

        # Execution context anomalies
        features["execution_context_anomaly"] = 0

        # System account with interactive tools
        if user in ["NT AUTHORITY\\SYSTEM", "NT AUTHORITY\\NETWORK SERVICE"]:
            interactive_tools = ["cmd.exe", "powershell.exe", "wmic.exe"]
            if any(tool in exe_name for tool in interactive_tools):
                features["execution_context_anomaly"] = 1

        # Network service with admin tools
        if "NETWORK SERVICE" in user:
            admin_tools = ["reg.exe", "sc.exe", "takeown.exe", "icacls.exe"]
            if any(tool in exe_name for tool in admin_tools):
                features["execution_context_anomaly"] = 1

        # Low integrity with system binaries
        integrity = event.get("IntegrityLevel", "").lower()
        if "low" in integrity or "medium" in integrity:
            if exe_name in ["reg.exe", "sc.exe", "wevtutil.exe"]:
                features["execution_context_anomaly"] = 1

        return features

    def _analyze_binary_usage_patterns(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze usage patterns of system binaries."""
        features = {}

        cmdline = event.get("CommandLine", "")
        image = (event.get("Image") or event.get("SourceImage", "")).lower()
        exe_name = Path(image).name if image else ""

        if not cmdline or exe_name not in self.NATIVE_BINARIES:
            features["binary_usage_complexity"] = 0
            features["binary_usage_suspicious"] = 0
            return features

        # Analyze command complexity
        complexity_score = 0

        # Long command lines (often indicate obfuscation)
        if len(cmdline) > 200:
            complexity_score += 2
        elif len(cmdline) > 100:
            complexity_score += 1

        # Multiple flags/parameters
        flag_count = cmdline.count("/") + cmdline.count("-")
        if flag_count > 5:
            complexity_score += 1

        # Nested execution
        if cmdline.count('"') > 4 or cmdline.count("'") > 4:
            complexity_score += 1

        # Redirection/piping
        if any(char in cmdline for char in [">", "<", "|", ">>"]):
            complexity_score += 1

        features["binary_usage_complexity"] = min(complexity_score, 5)
        features["binary_usage_suspicious"] = 1 if complexity_score >= 3 else 0

        # Specific binary abuse patterns
        if exe_name == "powershell.exe":
            features["powershell_abuse"] = self._detect_powershell_abuse(cmdline)
        elif exe_name == "wmic.exe":
            features["wmic_abuse"] = self._detect_wmic_abuse(cmdline)
        elif exe_name == "rundll32.exe":
            features["rundll32_abuse"] = self._detect_rundll32_abuse(cmdline)
        else:
            features["powershell_abuse"] = 0
            features["wmic_abuse"] = 0
            features["rundll32_abuse"] = 0

        return features

    def _analyze_execution_context(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution context for anomalies."""
        features = {}

        user = event.get("User", "")
        integrity = event.get("IntegrityLevel", "").lower()
        parent = event.get("ParentImage", "").lower()
        image = (event.get("Image") or event.get("SourceImage", "")).lower()

        # Context risk score
        risk_score = 0

        # High-privilege context
        if "system" in integrity or "high" in integrity:
            risk_score += 2

        # System account
        if "SYSTEM" in user or "NETWORK SERVICE" in user:
            risk_score += 2

        # Unusual parent
        if "explorer.exe" in parent:
            risk_score += 1

        # System binary in unusual location
        if image and "system32" not in image.lower():
            if any(binary in image.lower() for binary in ["cmd.exe", "powershell.exe", "wmic.exe"]):
                risk_score += 2

        features["execution_context_risk"] = min(risk_score, 10)
        features["high_risk_context"] = 1 if risk_score >= 5 else 0

        return features

    def _detect_powershell_abuse(self, cmdline: str) -> int:
        """Detect PowerShell abuse patterns."""
        score = 0
        cmd_lower = cmdline.lower()

        # Encoded commands
        if "-enc" in cmd_lower or "-encodedcommand" in cmd_lower:
            score += 3

        # Bypass execution policy
        if "bypass" in cmd_lower and "executionpolicy" in cmd_lower:
            score += 2

        # Hidden window
        if "hidden" in cmd_lower and "windowstyle" in cmd_lower:
            score += 1

        # Download and execute
        if "downloadstring" in cmd_lower or "downloadfile" in cmd_lower:
            score += 2

        # IEX (Invoke-Expression)
        if "iex" in cmd_lower or "invoke-expression" in cmd_lower:
            score += 2

        return min(score, 5)

    def _detect_wmic_abuse(self, cmdline: str) -> int:
        """Detect WMIC abuse patterns."""
        score = 0
        cmd_lower = cmdline.lower()

        # Remote execution
        if "/node:" in cmdline:
            score += 3

        # Process creation
        if "process" in cmd_lower and "create" in cmd_lower:
            score += 2

        # With credentials
        if "/user:" in cmdline or "/password:" in cmdline:
            score += 2

        return min(score, 5)

    def _detect_rundll32_abuse(self, cmdline: str) -> int:
        """Detect Rundll32 abuse patterns."""
        score = 0
        cmd_lower = cmdline.lower()

        # JavaScript protocol
        if "javascript:" in cmdline:
            score += 3

        # Unusual DLL usage
        if "comsvcs.dll" in cmd_lower or "shell32.dll" in cmd_lower:
            score += 1

        # MiniDump (credential dumping)
        if "minidump" in cmd_lower:
            score += 3

        return min(score, 5)
