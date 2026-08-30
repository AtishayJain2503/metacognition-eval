"""
nemo_eval.eval.sql_match
========================
Multiset Counter SQL execution equivalence engine with cell normalization,
ORDER BY detection, read-only safety, and opcode timeout protection.
"""

from collections import Counter
import math
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from nemo_eval.eval.base import EvalResult


def normalize_sql_cell(val: Any) -> Any:
    """
    Normalize individual SQL cell value for canonical equivalence comparison.
    - Maps NULL variants (None, np.nan, 'NULL', 'null', 'None') to None
    - Normalizes numeric floats/ints (1.0 vs 1)
    - Strips and normalizes text strings
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return None
        # Float rounding for minor precision differences
        if isinstance(val, float) and val.is_integer():
            return int(val)
        elif isinstance(val, float):
            return round(val, 4)
        return val

    if isinstance(val, str):
        cleaned = val.strip()
        if cleaned.lower() in ("null", "none"):
            return None
        # Try numeric conversion
        try:
            float_val = float(cleaned)
            if float_val.is_integer():
                return int(float_val)
            return round(float_val, 4)
        except ValueError:
            return cleaned.lower()

    if isinstance(val, bytes):
        return val

    return str(val).strip().lower()


def normalize_row_tuple(row: Sequence[Any]) -> Tuple[Any, ...]:
    """Convert row collection into a normalized immutable tuple of cells."""
    return tuple(normalize_sql_cell(c) for c in row)


def extract_sql_from_text(text: str) -> str:
    """Extract clean SQL string from possible markdown code fences."""
    if not text:
        return ""
    cleaned = text.strip()
    sql_block = re.search(r"```(?:sql)?\s*\n([\s\S]*?)```", cleaned, re.IGNORECASE)
    if sql_block:
        cleaned = sql_block.group(1).strip()
    
    # Remove trailing semicolons
    cleaned = re.sub(r";\s*$", "", cleaned).strip()
    return cleaned


def has_order_by_clause(sql: str) -> bool:
    """Check whether a SQL query contains an active top-level ORDER BY clause."""
    if not sql:
        return False
    # Simple regex check for ORDER BY
    return bool(re.search(r"\border\s+by\b", sql, re.IGNORECASE))


def execute_sql_safely(
    db_path: str,
    query: str,
    timeout_seconds: float = 3.0,
    max_rows: int = 500
) -> Tuple[Optional[List[Tuple[Any, ...]]], Optional[List[str]], Optional[str]]:
    """
    Execute SQL query in a read-only sandboxed connection with opcode progress timeout.
    Returns (rows, column_names, error_message).
    """
    cleaned_query = extract_sql_from_text(query)
    if not cleaned_query:
        return None, None, "Empty SQL query."

    # Prohibit dangerous modification keywords if not strictly SELECT / PRAGMA / EXPLAIN
    prohibited_keywords = [
        "DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "ATTACH ", "DETACH ", "REINDEX "
    ]
    upper_query = cleaned_query.upper()
    for kw in prohibited_keywords:
        if upper_query.startswith(kw) or f";{kw}" in upper_query.replace(" ", ""):
            return None, None, f"Prohibited write statement: {kw.strip()}"

    start_perf = time.perf_counter()

    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        
        # Enforce read-only
        cursor.execute("PRAGMA query_only = ON;")
        cursor.execute("PRAGMA busy_timeout = 5000;")

        # Progress handler for execution timeout
        def _timeout_handler():
            if (time.perf_counter() - start_perf) > timeout_seconds:
                return 1 # Aborts query execution
            return 0

        conn.set_progress_handler(_timeout_handler, 1000)

        cursor.execute(cleaned_query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows)
        
        conn.set_progress_handler(None, 0)
        conn.close()
        return rows, columns, None
    except sqlite3.OperationalError as e:
        err_msg = str(e)
        if "interrupted" in err_msg.lower() or "progress" in err_msg.lower():
            return None, None, f"Query execution timed out after {timeout_seconds}s (Opcode limit reached)."
        return None, None, f"SQL OperationalError: {err_msg}"
    except Exception as e:
        return None, None, f"SQL Execution Exception: {type(e).__name__}: {str(e)}"


def evaluate_sql(
    candidate_sql_or_result: Any,
    gold_sql_or_result: Any,
    db_path: Optional[str] = None,
    check_order: Optional[bool] = None,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4
) -> EvalResult:
    """
    Evaluate candidate SQL query or result set against gold SQL reference.
    """
    start_perf = time.perf_counter()

    cand_rows: Optional[List[Tuple[Any, ...]]] = None
    gold_rows: Optional[List[Tuple[Any, ...]]] = None
    cand_cols: Optional[List[str]] = None
    gold_cols: Optional[List[str]] = None
    execution_diag = ""

    # 1. Resolve Candidate result set
    if isinstance(candidate_sql_or_result, str):
        if not db_path:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            return EvalResult(
                score=0.0,
                is_correct=False,
                eval_type="sql_multiset",
                candidate_parsed=candidate_sql_or_result,
                gold_target=gold_sql_or_result,
                diagnostic_message="Missing db_path for SQL execution.",
                execution_time_ms=round(elapsed_ms, 3)
            )
        cand_rows, cand_cols, cand_err = execute_sql_safely(db_path, candidate_sql_or_result)
        if cand_err:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            return EvalResult(
                score=0.0,
                is_correct=False,
                eval_type="sql_multiset",
                candidate_parsed=candidate_sql_or_result,
                gold_target=gold_sql_or_result,
                diagnostic_message=f"Candidate SQL failed: {cand_err}",
                execution_time_ms=round(elapsed_ms, 3),
                details={"error": cand_err}
            )
    elif isinstance(candidate_sql_or_result, (list, tuple)):
        cand_rows = [tuple(r) if isinstance(r, (list, tuple)) else (r,) for r in candidate_sql_or_result]
    else:
        cand_rows = [(candidate_sql_or_result,)]

    # 2. Resolve Gold reference result set
    if isinstance(gold_sql_or_result, str):
        if not db_path:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            return EvalResult(
                score=0.0,
                is_correct=False,
                eval_type="sql_multiset",
                candidate_parsed=candidate_sql_or_result,
                gold_target=gold_sql_or_result,
                diagnostic_message="Missing db_path for Gold SQL execution.",
                execution_time_ms=round(elapsed_ms, 3)
            )
        gold_rows, gold_cols, gold_err = execute_sql_safely(db_path, gold_sql_or_result)
        if gold_err:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            return EvalResult(
                score=0.0,
                is_correct=False,
                eval_type="sql_multiset",
                candidate_parsed=candidate_sql_or_result,
                gold_target=gold_sql_or_result,
                diagnostic_message=f"Gold SQL failed execution: {gold_err}",
                execution_time_ms=round(elapsed_ms, 3),
                details={"gold_error": gold_err}
            )
    elif isinstance(gold_sql_or_result, (list, tuple)):
        gold_rows = [tuple(r) if isinstance(r, (list, tuple)) else (r,) for r in gold_sql_or_result]
    else:
        gold_rows = [(gold_sql_or_result,)]

    # 3. Determine ordering requirement
    enforce_order = False
    if check_order is not None:
        enforce_order = check_order
    elif isinstance(gold_sql_or_result, str) and has_order_by_clause(gold_sql_or_result):
        enforce_order = True

    # 4. Compare Normalized Rows
    norm_cand_rows = [normalize_row_tuple(r) for r in (cand_rows or [])]
    norm_gold_rows = [normalize_row_tuple(r) for r in (gold_rows or [])]

    # Check projection width (if both have non-empty rows)
    if norm_cand_rows and norm_gold_rows:
        if len(norm_cand_rows[0]) != len(norm_gold_rows[0]):
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            return EvalResult(
                score=0.0,
                is_correct=False,
                eval_type="sql_multiset",
                candidate_parsed=norm_cand_rows,
                gold_target=norm_gold_rows,
                diagnostic_message=f"Column count mismatch: candidate projected {len(norm_cand_rows[0])} cols, gold projected {len(norm_gold_rows[0])} cols.",
                execution_time_ms=round(elapsed_ms, 3)
            )

    if enforce_order:
        is_match = (norm_cand_rows == norm_gold_rows)
        diag = "Ordered SQL match passed." if is_match else f"Ordered SQL mismatch: candidate={norm_cand_rows}, gold={norm_gold_rows}"
    else:
        # Multiset comparison via Counter
        cand_counter = Counter(norm_cand_rows)
        gold_counter = Counter(norm_gold_rows)
        is_match = (cand_counter == gold_counter)
        diag = "Multiset SQL match passed." if is_match else f"Multiset SQL mismatch: candidate={norm_cand_rows}, gold={norm_gold_rows}"

    elapsed_ms = (time.perf_counter() - start_perf) * 1000.0

    return EvalResult(
        score=1.0 if is_match else 0.0,
        is_correct=is_match,
        eval_type="sql_multiset",
        candidate_parsed=norm_cand_rows,
        gold_target=norm_gold_rows,
        diagnostic_message=diag,
        execution_time_ms=round(elapsed_ms, 3),
        details={
            "enforce_order": enforce_order,
            "cand_row_count": len(norm_cand_rows),
            "gold_row_count": len(norm_gold_rows)
        }
    )
