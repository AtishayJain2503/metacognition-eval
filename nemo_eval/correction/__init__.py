"""
nemo_eval.correction
--------------------
Intermediate assertion verification and iterative self-correction metrics (Milestone 5).

Exports:
    - IntermediateVerifier: Schema/shape/assertion checker for tool outputs.
    - SelfCorrectMetrics: SCSR, CEI, TOP metric calculator.
"""

from nemo_eval.correction.verifier import IntermediateVerifier, VerificationResult
from nemo_eval.correction.self_correct import SelfCorrectMetrics, CorrectionStats

__all__ = [
    "IntermediateVerifier",
    "VerificationResult",
    "SelfCorrectMetrics",
    "CorrectionStats",
]
