"""
nemo_eval.correction.verifier
------------------------------
Intermediate assertion and schema/shape consistency checker.

Performs lightweight, hermetic checks on tool outputs between
FSM OBSERVATION and VERIFICATION states:

Checks:
    1. Non-empty: output is not None, empty list, or empty DataFrame.
    2. Schema match: if expected_schema provided, output keys/columns are a superset.
    3. Type match: if expected_type provided, output type is as expected.
    4. Numeric range: if bounds provided, all numeric values lie within bounds.
    5. No NaN: if strict, reject outputs containing NaN/null values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


class VerificationResult(BaseModel):
    """Result of an intermediate assertion check."""
    model_config = ConfigDict(extra="ignore")

    passed: bool
    check_name: str
    detail: str = ""
    severity: Literal["error", "warning", "info"] = "error"

    def __bool__(self) -> bool:
        return self.passed


class IntermediateVerifier:
    """
    Runs a configurable battery of checks on a ToolResult's data payload.

    Usage:
        verifier = IntermediateVerifier()
        results = verifier.verify(
            data=tool_result.data,
            expected_type="list",
            expected_schema=["col_a", "col_b"],
            strict_no_nan=True,
        )
        passed = all(r.passed for r in results)
    """

    def verify(
        self,
        data: Any,
        expected_type: Optional[str] = None,
        expected_schema: Optional[List[str]] = None,
        numeric_bounds: Optional[Tuple[float, float]] = None,
        strict_no_nan: bool = False,
    ) -> List[VerificationResult]:
        """
        Run all applicable checks on data.

        Args:
            data: The tool result data payload.
            expected_type: Expected Python type name ('list', 'dict', 'scalar', 'dataframe').
            expected_schema: Expected column/key names that must be present.
            numeric_bounds: (min_val, max_val) that all numbers must fall within.
            strict_no_nan: If True, reject any NaN or None values in the payload.

        Returns:
            List of VerificationResult, one per check run.
        """
        results: List[VerificationResult] = []

        results.append(self._check_non_empty(data))
        if not results[-1].passed:
            return results  # Short-circuit: no point checking further

        if expected_type:
            results.append(self._check_type(data, expected_type))

        if expected_schema:
            results.append(self._check_schema(data, expected_schema))

        if numeric_bounds is not None:
            results.append(self._check_numeric_bounds(data, numeric_bounds))

        if strict_no_nan:
            results.append(self._check_no_nan(data))

        return results

    def all_passed(self, results: List[VerificationResult]) -> bool:
        return all(r.passed for r in results)

    # ------------------------------------------------------------------ #
    # Individual checks
    # ------------------------------------------------------------------ #

    def _check_non_empty(self, data: Any) -> VerificationResult:
        """Check that data is not None, empty list, empty dict, or 0 rows."""
        if data is None:
            return VerificationResult(passed=False, check_name="non_empty", detail="Data is None.")
        if isinstance(data, (list, dict)) and len(data) == 0:
            return VerificationResult(passed=False, check_name="non_empty", detail="Data is empty collection.")
        if isinstance(data, str) and not data.strip():
            return VerificationResult(passed=False, check_name="non_empty", detail="Data is empty string.", severity="warning")
        return VerificationResult(passed=True, check_name="non_empty", detail="Data is non-empty.", severity="info")

    def _check_type(self, data: Any, expected_type: str) -> VerificationResult:
        """Check that data matches the expected type label."""
        type_map = {
            "list": (list,),
            "dict": (dict,),
            "scalar": (int, float, bool, str),
            "string": (str,),
            "boolean": (bool,),
            "int": (int,),
            "float": (float, int),
            "dataframe": None,  # handled specially
        }

        if expected_type == "dataframe":
            # Accept list-of-dicts as a valid proxy for dataframe
            if isinstance(data, list) and all(isinstance(r, dict) for r in data):
                return VerificationResult(passed=True, check_name="type_match", detail="Data is list-of-dicts (dataframe proxy).", severity="info")
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    return VerificationResult(passed=True, check_name="type_match", detail="Data is pandas DataFrame.", severity="info")
            except ImportError:
                pass
            return VerificationResult(passed=False, check_name="type_match", detail=f"Expected 'dataframe', got {type(data).__name__}.")

        expected_types = type_map.get(expected_type)
        if expected_types is None:
            return VerificationResult(passed=True, check_name="type_match", detail=f"Unknown expected type '{expected_type}', skipping.", severity="warning")

        if isinstance(data, expected_types):
            return VerificationResult(passed=True, check_name="type_match", detail=f"Type matches '{expected_type}'.", severity="info")
        return VerificationResult(passed=False, check_name="type_match", detail=f"Expected '{expected_type}', got '{type(data).__name__}'.")

    def _check_schema(self, data: Any, expected_schema: List[str]) -> VerificationResult:
        """Check that dict keys or list-of-dict keys are a superset of expected_schema."""
        if isinstance(data, dict):
            actual_keys = set(data.keys())
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            actual_keys = set(data[0].keys())
        else:
            return VerificationResult(
                passed=False, check_name="schema_match",
                detail=f"Cannot extract schema from type {type(data).__name__}.",
            )

        required = set(expected_schema)
        missing = required - actual_keys
        if missing:
            return VerificationResult(
                passed=False, check_name="schema_match",
                detail=f"Missing expected keys/columns: {sorted(missing)}.",
            )
        return VerificationResult(
            passed=True, check_name="schema_match",
            detail=f"All expected schema fields present: {expected_schema}.",
            severity="info",
        )

    def _check_numeric_bounds(
        self, data: Any, bounds: Tuple[float, float]
    ) -> VerificationResult:
        """Check that all numeric values in data fall within [min, max]."""
        lo, hi = bounds
        violations = []

        def _collect_nums(obj: Any) -> None:
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                if not (lo <= obj <= hi):
                    violations.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    _collect_nums(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect_nums(v)

        _collect_nums(data)
        if violations:
            return VerificationResult(
                passed=False, check_name="numeric_bounds",
                detail=f"{len(violations)} value(s) outside [{lo}, {hi}]: {violations[:5]}.",
            )
        return VerificationResult(
            passed=True, check_name="numeric_bounds",
            detail=f"All numeric values within [{lo}, {hi}].", severity="info",
        )

    def _check_no_nan(self, data: Any) -> VerificationResult:
        """Check that no NaN/None values appear in the data."""
        nan_count = 0

        def _count_nans(obj: Any) -> None:
            nonlocal nan_count
            if obj is None:
                nan_count += 1
            elif isinstance(obj, float):
                import math
                if math.isnan(obj):
                    nan_count += 1
            elif isinstance(obj, list):
                for item in obj:
                    _count_nans(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _count_nans(v)

        _count_nans(data)
        if nan_count > 0:
            return VerificationResult(
                passed=False, check_name="no_nan",
                detail=f"Found {nan_count} NaN/None value(s) in output.",
            )
        return VerificationResult(
            passed=True, check_name="no_nan",
            detail="No NaN/None values detected.", severity="info",
        )
