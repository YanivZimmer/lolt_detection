"""
Heuristics for classifying LOTL failure patterns.
"""

from typing import Dict, Any, Tuple

SUBTLE_OBFUSCATION = "Subtle Obfuscation Patterns"
CONTEXT_LEGIT = "Context-Dependent Legitimate Use"
NOVEL_TECH = "Novel Attack Techniques"
UNMAPPED = "New Pattern - Unmapped Behavior"

OBFUSCATION_INDICATORS = [
    "-enc",
    "frombase64string",
    "powershell -e",
    "base64",
    "invoke-expression",
    "iex ",
    "certutil",
    "var ",
]

ADMIN_TOOL_KEYWORDS = [
    "get-ad",
    "dsquery",
    "schtasks",
    "sc.exe",
    "net user",
    "net group",
    "ntdsutil",
    "wevtutil",
]

NOVEL_TOOL_KEYWORDS = [
    "pwsh",
    "wsl",
    "azuread",
    "procdump",
    "livingofftheland",
    "new-",
    "winget",
]

KNOWN_NATIVE_BINARIES = {
    "powershell.exe",
    "cmd.exe",
    "wscript.exe",
    "cscript.exe",
    "mshta.exe",
    "rundll32.exe",
    "regsvr32.exe",
    "wmic.exe",
    "explorer.exe",
}


def classify_failure_pattern(event: Dict[str, Any]) -> Tuple[str, str]:
    """
    Label an event according to expected failure patterns.

    Args:
        event: Sanitized event dictionary

    Returns:
        Tuple of (pattern_name, evidence_string)
    """
    cmd = (event.get("CommandLine") or event.get("commandLine") or "").lower()
    image = (event.get("Image") or event.get("SourceImage") or "").lower()
    user = (event.get("User") or event.get("AccountName") or "").lower()

    # Pattern 1: Subtle obfuscation / encoding clues
    for indicator in OBFUSCATION_INDICATORS:
        if indicator in cmd:
            return (
                SUBTLE_OBFUSCATION,
                f"Command line contains obfuscation indicator '{indicator}'.",
            )

    # Pattern 2: Legitimate admin activity that looks suspicious
    if any(keyword in cmd for keyword in ADMIN_TOOL_KEYWORDS):
        if "admin" in user or "svc" in user or "it" in user or user.endswith("$"):
            return (
                CONTEXT_LEGIT,
                "Administrative keyword detected while running under a privileged account.",
            )

    # Pattern 3: Novel or rarely seen tool usage
    if image and image not in KNOWN_NATIVE_BINARIES:
        for keyword in NOVEL_TOOL_KEYWORDS:
            if keyword in cmd or keyword in image:
                return (
                    NOVEL_TECH,
                    f"Command references newer platform/tool keyword '{keyword}'.",
                )

    return (
        UNMAPPED,
        "No heuristic matched; treat as emerging or previously unseen behavior.",
    )
