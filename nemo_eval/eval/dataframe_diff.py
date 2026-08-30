"""
nemo_eval.eval.dataframe_diff
=============================
Deep tabular DataFrame structural and cell-level diffing comparator with column
alignment, NaN pattern alignment, sorting tolerance, and float tolerance.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from nemo_eval.eval.base import EvalResult
from nemo_eval.eval.numerical import check_numerical_tolerance


def coerce_to_dataframe(data: Any) -> Optional[pd.DataFrame]:
    """Coerce various tabular structures (dicts, list of records, Series, DataFrame) to pandas DataFrame."""
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, dict):
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame([data])
    if isinstance(data, list):
        try:
            return pd.DataFrame(data)
        except Exception:
            return None
    return None


def align_dataframe_columns(
    df_cand: pd.DataFrame,
    df_gold: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Align column names between candidate and gold DataFrames.
    Tries:
    1. Case-insensitive and trimmed column matching
    2. Positional alignment if shape matches
    """
    cand = df_cand.copy()
    gold = df_gold.copy()

    # Clean column headers
    cand.columns = [str(c).strip() for c in cand.columns]
    gold.columns = [str(c).strip() for c in gold.columns]

    cand_cols_lower = {str(c).lower(): c for c in cand.columns}
    gold_cols_lower = {str(c).lower(): c for c in gold.columns}

    # If exact or case-insensitive match on all columns
    if set(cand_cols_lower.keys()) == set(gold_cols_lower.keys()):
        # Reorder cand to match gold order
        ordered_cand_cols = [cand_cols_lower[k] for k in gold_cols_lower.keys()]
        cand = cand[ordered_cand_cols]
        cand.columns = list(gold.columns)
        return cand, gold, True

    # Positional fallback if column count matches
    if len(cand.columns) == len(gold.columns):
        cand.columns = list(gold.columns)
        return cand, gold, True

    return cand, gold, False


def compare_cell_values(
    val_cand: Any,
    val_gold: Any,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01
) -> bool:
    """Compare two individual cell values with tolerance and NaN alignment."""
    # 1. Check for null / NaN
    cand_null = pd.isna(val_cand) or val_cand is None
    gold_null = pd.isna(val_gold) or val_gold is None
    if gold_null and cand_null:
        return True
    if gold_null or cand_null:
        return False

    # 2. Check numeric comparison
    if isinstance(val_gold, (int, float, np.number)) or isinstance(val_cand, (int, float, np.number)):
        try:
            f_cand = float(val_cand)
            f_gold = float(val_gold)
            is_match, _, _ = check_numerical_tolerance(f_cand, f_gold, rel_tol=rel_tol, abs_tol=abs_tol)
            return is_match
        except (ValueError, TypeError):
            pass

    # 3. String / Object comparison
    s_cand = str(val_cand).strip().lower()
    s_gold = str(val_gold).strip().lower()
    if s_cand == s_gold:
        return True

    # 4. Datetime comparison
    try:
        dt_cand = pd.to_datetime(val_cand)
        dt_gold = pd.to_datetime(val_gold)
        return dt_cand == dt_gold
    except Exception:
        pass

    return False


def evaluate_dataframe(
    candidate_df: Any,
    gold_df: Any,
    check_order: bool = False,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01
) -> EvalResult:
    """
    Deep tabular DataFrame diffing engine with shape verification,
    column alignment, and cell tolerance.
    """
    start_perf = time.perf_counter()

    cand = coerce_to_dataframe(candidate_df)
    gold = coerce_to_dataframe(gold_df)

    if cand is None or gold is None:
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        return EvalResult(
            score=0.0,
            is_correct=False,
            eval_type="dataframe_diff",
            candidate_parsed=str(candidate_df),
            gold_target=str(gold_df),
            diagnostic_message=f"DataFrame coercion failed: candidate is {type(candidate_df)}, gold is {type(gold_df)}",
            execution_time_ms=round(elapsed_ms, 3)
        )

    # 1. Shape Verification
    if cand.shape != gold.shape:
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        return EvalResult(
            score=0.0,
            is_correct=False,
            eval_type="dataframe_diff",
            candidate_parsed=f"DataFrame with shape {cand.shape}",
            gold_target=f"DataFrame with shape {gold.shape}",
            diagnostic_message=f"DataFrame shape mismatch: candidate shape {cand.shape} != gold shape {gold.shape}",
            execution_time_ms=round(elapsed_ms, 3),
            details={"cand_shape": cand.shape, "gold_shape": gold.shape}
        )

    # 2. Column Alignment
    cand_aligned, gold_aligned, cols_aligned = align_dataframe_columns(cand, gold)
    if not cols_aligned:
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        return EvalResult(
            score=0.0,
            is_correct=False,
            eval_type="dataframe_diff",
            candidate_parsed=list(cand.columns),
            gold_target=list(gold.columns),
            diagnostic_message=f"DataFrame column mismatch: candidate cols {list(cand.columns)} != gold cols {list(gold.columns)}",
            execution_time_ms=round(elapsed_ms, 3)
        )

    # 3. Row Sorting (if check_order is False)
    if not check_order and len(gold_aligned) > 1:
        sort_cols = list(gold_aligned.columns)
        try:
            cand_aligned = cand_aligned.sort_values(by=sort_cols).reset_index(drop=True)
            gold_aligned = gold_aligned.sort_values(by=sort_cols).reset_index(drop=True)
        except Exception:
            # If sorting fails due to unhashable types, compare row by row
            pass

    # 4. Cell-by-cell comparison
    diff_cells = []
    total_cells = cand_aligned.shape[0] * cand_aligned.shape[1]
    
    for row_idx in range(cand_aligned.shape[0]):
        for col_idx in range(cand_aligned.shape[1]):
            c_val = cand_aligned.iat[row_idx, col_idx]
            g_val = gold_aligned.iat[row_idx, col_idx]
            if not compare_cell_values(c_val, g_val, rel_tol=rel_tol, abs_tol=abs_tol):
                col_name = str(gold_aligned.columns[col_idx])
                diff_cells.append({
                    "row": row_idx,
                    "column": col_name,
                    "candidate_val": str(c_val),
                    "gold_val": str(g_val)
                })
                if len(diff_cells) >= 10:
                    break
        if len(diff_cells) >= 10:
            break

    is_match = (len(diff_cells) == 0)
    elapsed_ms = (time.perf_counter() - start_perf) * 1000.0

    diag = (
        "DataFrame match passed."
        if is_match
        else f"DataFrame cell diffs detected: {len(diff_cells)} mismatched cells (showing first {min(len(diff_cells), 5)}: {diff_cells[:5]})"
    )

    return EvalResult(
        score=1.0 if is_match else 0.0,
        is_correct=is_match,
        eval_type="dataframe_diff",
        candidate_parsed=cand_aligned.to_dict(orient="records"),
        gold_target=gold_aligned.to_dict(orient="records"),
        diagnostic_message=diag,
        execution_time_ms=round(elapsed_ms, 3),
        details={"shape": cand.shape, "diff_count": len(diff_cells), "sample_diffs": diff_cells[:5]}
    )
