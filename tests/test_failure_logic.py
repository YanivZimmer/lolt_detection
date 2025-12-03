import unittest

from lotl_detector.ensemble import LOTLEnsemble
from lotl_detector.failure_patterns import (
    classify_failure_pattern,
    SUBTLE_OBFUSCATION,
    CONTEXT_LEGIT,
    NOVEL_TECH,
)


def _make_ensemble_without_init() -> LOTLEnsemble:
    # Avoid running the heavy __init__ (loads feature extractors/LLMs)
    return LOTLEnsemble.__new__(LOTLEnsemble)


class AttackTypeTests(unittest.TestCase):
    def test_infer_attack_type_uses_weighted_top_features(self):
        ensemble = _make_ensemble_without_init()
        top_features = [
            {"name": "process_discovery", "contribution": 0.5},
            {"name": "apt_exfiltration", "contribution": 1.5},
            {"name": "system_file_modification", "contribution": 0.2},
        ]
        attack_type = ensemble._infer_attack_type(top_features)
        self.assertEqual("exfiltration", attack_type)

    def test_infer_attack_type_returns_uncertain_when_no_signal(self):
        ensemble = _make_ensemble_without_init()
        top_features = [
            {"name": "totally_unknown_feature", "contribution": 0.9},
            {"name": "another_feature", "contribution": 0.3},
        ]
        attack_type = ensemble._infer_attack_type(top_features)
        self.assertEqual("attack type uncertain", attack_type)


class FailurePatternTests(unittest.TestCase):
    def test_classify_failure_pattern_obfuscation(self):
        event = {"CommandLine": "powershell -enc aGVsbG8=", "User": "svc_admin"}
        pattern, evidence = classify_failure_pattern(event)
        self.assertEqual(SUBTLE_OBFUSCATION, pattern)
        self.assertIn("obfuscation", evidence.lower())

    def test_classify_failure_pattern_context_dependent(self):
        event = {"CommandLine": "Get-ADUser -Filter *", "User": "DOMAIN\\AdminUser"}
        pattern, evidence = classify_failure_pattern(event)
        self.assertEqual(CONTEXT_LEGIT, pattern)
        self.assertIn("administrative", evidence.lower())

    def test_classify_failure_pattern_novel_tool(self):
        event = {"CommandLine": "pwsh -File deploy.ps1", "Image": "C:\\\\Tools\\\\pwsh.exe"}
        pattern, evidence = classify_failure_pattern(event)
        self.assertEqual(NOVEL_TECH, pattern)
        self.assertIn("keyword", evidence.lower())


if __name__ == "__main__":
    unittest.main()

