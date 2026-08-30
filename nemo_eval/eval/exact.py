"""
nemo_eval.eval.exact
====================
Multi-stage text normalization, boolean matching, punctuation stripping,
and collection exact match engine.
"""

import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from nemo_eval.eval.base import EvalResult

# Truthy and Falsy string dictionaries
TRUTHY_VALUES: Set[str] = {
    "true", "yes", "1", "y", "t", "correct", "true.", "yes.", "right"
}
FALSY_VALUES: Set[str] = {
    "false", "no", "0", "n", "f", "incorrect", "false.", "no.", "wrong"
}

CURRENCY_SYMBOLS: List[str] = ["$", "€", "£", "¥", "₹", "元", "USD", "EUR", "GBP"]


def strip_markdown(text: str) -> str:
    """Strip markdown formatting such as bold, italics, code tags, and LaTeX."""
    if not text:
        return ""
    # LaTeX \boxed{...} or \text{...}
    text = re.sub(r"\\(?:boxed|text|textbf)\{([^{}]*)\}", r"\1", text)
    # Bold and italics
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def strip_articles(text: str) -> str:
    """Strip leading or standalone English articles ('a', 'an', 'the')."""
    return re.sub(r"\b(?:the|a|an)\b\s*", "", text, flags=re.IGNORECASE)


def normalize_number_commas(text: str) -> str:
    """Remove thousands separators in numbers (e.g., '1,250,000.50' -> '1250000.50')."""
    return re.sub(r"(?<=\d),(?=\d{3}(?:[^\d]|$))", "", text)


def normalize_text(
    text: Any,
    strip_punct: bool = True,
    strip_curr: bool = True,
    strip_pct: bool = True,
    strip_art: bool = True,
    case_fold: bool = True
) -> str:
    """
    Apply multi-stage string normalization pipeline.
    """
    if text is None:
        return ""
    s = str(text).strip()
    
    # 1. Strip markdown
    s = strip_markdown(s)

    # 2. Normalize numbers with commas
    s = normalize_number_commas(s)

    # 3. Strip currency symbols
    if strip_curr:
        for sym in CURRENCY_SYMBOLS:
            s = s.replace(sym, "")

    # 4. Strip percentage
    if strip_pct:
        s = s.replace("%", "")

    # 5. Strip English articles
    if strip_art:
        s = strip_articles(s)

    # 6. Case folding
    if case_fold:
        s = s.lower()

    # 7. Strip trailing/leading and separating punctuation
    if strip_punct:
        s = re.sub(r"[,!?:;\"'()\[\]{}]", " ", s)
        s = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", s)

    # 8. Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_boolean(val: Any) -> Optional[bool]:
    """
    Normalize boolean or boolean-like string into True / False / None.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if val == 1 or val == 1.0:
            return True
        elif val == 0 or val == 0.0:
            return False
        return None

    if isinstance(val, str):
        cleaned = val.strip().lower()
        cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned)
        if cleaned in TRUTHY_VALUES:
            return True
        elif cleaned in FALSY_VALUES:
            return False

    return None


def evaluate_exact(
    candidate: Any,
    gold: Any,
    ignore_case: bool = True,
    strip_punctuation: bool = True,
    unordered_collection: bool = True
) -> EvalResult:
    """
    Polymorphic exact and normalized matching engine supporting strings,
    booleans, collections (lists, sets, tuples), and dictionaries.
    """
    start_perf = time.perf_counter()

    # 1. Check boolean normalization
    cand_bool = normalize_boolean(candidate)
    gold_bool = normalize_boolean(gold)
    if gold_bool is not None and cand_bool is not None:
        is_match = (cand_bool == gold_bool)
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        return EvalResult(
            score=1.0 if is_match else 0.0,
            is_correct=is_match,
            eval_type="exact",
            candidate_parsed=cand_bool,
            gold_target=gold_bool,
            diagnostic_message="Boolean match passed." if is_match else f"Boolean mismatch: candidate={cand_bool}, gold={gold_bool}",
            execution_time_ms=round(elapsed_ms, 3),
            details={"type": "boolean", "cand_bool": cand_bool, "gold_bool": gold_bool}
        )

    # 2. Check collection comparison (list, tuple, set)
    if isinstance(gold, (list, tuple, set)) or isinstance(candidate, (list, tuple, set)):
        # Coerce candidate to collection if it is a comma/newline separated string
        cand_items = candidate
        if isinstance(candidate, str) and not isinstance(gold, str):
            cand_str = strip_markdown(candidate).strip()
            if cand_str.startswith("[") and cand_str.endswith("]"):
                cand_str = cand_str[1:-1]
            cand_items = [item.strip() for item in re.split(r"[,;\n]+", cand_str) if item.strip()]

        gold_list = list(gold) if isinstance(gold, (list, tuple, set)) else [gold]
        cand_list = list(cand_items) if isinstance(cand_items, (list, tuple, set)) else [cand_items]

        # Normalize items
        norm_gold = [normalize_text(x, strip_punct=strip_punctuation, case_fold=ignore_case) for x in gold_list]
        norm_cand = [normalize_text(x, strip_punct=strip_punctuation, case_fold=ignore_case) for x in cand_list]

        if unordered_collection:
            # Compare as multisets / sets
            is_match = (sorted(norm_cand) == sorted(norm_gold))
        else:
            is_match = (norm_cand == norm_gold)

        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        diag = "Collection match passed." if is_match else f"Collection mismatch: candidate={norm_cand}, gold={norm_gold}"
        return EvalResult(
            score=1.0 if is_match else 0.0,
            is_correct=is_match,
            eval_type="exact",
            candidate_parsed=cand_list,
            gold_target=gold_list,
            diagnostic_message=diag,
            execution_time_ms=round(elapsed_ms, 3),
            details={"type": "collection", "cand_normalized": norm_cand, "gold_normalized": norm_gold}
        )

    # 3. Direct primitive or normalized string comparison
    norm_cand = normalize_text(candidate, strip_punct=strip_punctuation, case_fold=ignore_case)
    norm_gold = normalize_text(gold, strip_punct=strip_punctuation, case_fold=ignore_case)

    is_match = (norm_cand == norm_gold)
    
    # Fallback check: if raw unnormalized match
    if not is_match and candidate == gold:
        is_match = True

    elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
    diag = "Exact match passed." if is_match else f"Exact match mismatch: candidate='{norm_cand}', gold='{norm_gold}'"

    return EvalResult(
        score=1.0 if is_match else 0.0,
        is_correct=is_match,
        eval_type="exact",
        candidate_parsed=norm_cand,
        gold_target=norm_gold,
        diagnostic_message=diag,
        execution_time_ms=round(elapsed_ms, 3),
        details={"cand_raw": candidate, "gold_raw": gold, "norm_cand": norm_cand, "norm_gold": norm_gold}
    )
