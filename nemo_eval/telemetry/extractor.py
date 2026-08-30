"""
nemo_eval.telemetry.extractor
------------------------------
Value-Only Target Answer Extractor.

Extracts strictly the final target scalar, fraction, or mathematical expression
from unstructured Chain-of-Thought (CoT) completions, LaTeX formulas,
JSON envelopes, anchor patterns, and raw prose.
"""

from __future__ import annotations

import json
import re
from typing import Optional


class ValueExtractor:
    """
    Multi-tier extraction engine for isolating target answers from LLM reasoning traces.

    Priority Tiers:
    1. LaTeX \\boxed{...} / \\fbox{...} with balanced curly brace matching (last occurrence).
    2. JSON envelopes {"answer": ...}, {"final_answer": ...} or markdown json blocks.
    3. Explicit natural language anchors ("####", "Final Answer:", "The answer is:").
    4. Equation tails ("x = 42", "ans = 105").
    5. Markdown, formatting, currency, and unit stripping.
    6. Rightmost numeric token, fraction, or scientific notation fallback.
    7. Expected type normalization (float_tol, set, exact, math_symbolic).
    """

    @classmethod
    def extract_value(cls, raw_text: Optional[str], expected_type: Optional[str] = None) -> str:
        """
        Extract the target scalar / value from raw_text.

        Args:
            raw_text: Raw completion string from LLM or evaluation step.
            expected_type: Optional hint ('math_symbolic', 'float_tol', 'exact', 'set', 'fraction').

        Returns:
            Extracted clean value string, or empty string if no answer found.
        """
        if raw_text is None or not isinstance(raw_text, str):
            return ""

        text = raw_text.strip()
        if not text:
            return ""

        # Tier 1: LaTeX \boxed{...} with balanced curly braces
        boxed_val = cls._extract_boxed(text)
        if boxed_val is not None:
            return cls._post_process(boxed_val, expected_type)

        # Tier 2: JSON payload / code blocks
        json_val = cls._extract_json(text)
        if json_val is not None:
            return cls._post_process(json_val, expected_type)

        # Tier 3: Explicit regex anchors ("Final Answer:", "####", "The answer is")
        anchor_val = cls._extract_anchors(text)
        if anchor_val is not None:
            return cls._post_process(anchor_val, expected_type)

        # Tier 4: Equation tail parser ("x = ...", "ans = ...")
        eq_val = cls._extract_equation_tail(text)
        if eq_val is not None:
            return cls._post_process(eq_val, expected_type)

        # If expected_type is 'set' or text looks like a set '{...}', preserve the set
        if expected_type == "set":
            cleaned_set = cls._strip_formatting_and_units(text)
            if not (cleaned_set.startswith("{") and cleaned_set.endswith("}")):
                cleaned_set = f"{{{cleaned_set}}}"
            return cls._post_process(cleaned_set, expected_type)

        if text.startswith("{") and text.endswith("}"):
            return cls._post_process(text, expected_type)

        # Tier 5 & 6: Rightmost numeric / fraction / formula fallback from lines
        num_val = cls._extract_numeric_fallback(text)
        if num_val is not None:
            return cls._post_process(num_val, expected_type)

        # Final fallback: strip prose formatting and units
        cleaned = cls._strip_formatting_and_units(text)
        return cls._post_process(cleaned, expected_type)

    # ------------------------------------------------------------------ #
    # Tier 1: Balanced LaTeX \boxed{...} Scanner
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_boxed(cls, text: str) -> Optional[str]:
        """
        Extract the content of the rightmost balanced \\boxed{...} or \\fbox{...}.
        Supports arbitrarily nested curly braces.
        """
        matches = []
        pattern = re.compile(r'\\(?:boxed|fbox)\s*\{')

        for match in pattern.finditer(text):
            start_idx = match.end()  # Right after opening brace '{'
            depth = 1
            idx = start_idx
            while idx < len(text) and depth > 0:
                char = text[idx]
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                idx += 1

            if depth == 0:
                # Successfully closed outer brace
                extracted = text[start_idx:idx - 1].strip()
                matches.append(extracted)

        if matches:
            # Pick the last / rightmost match (the final conclusion)
            return matches[-1]

        return None

    # ------------------------------------------------------------------ #
    # Tier 2: JSON Payload Extractor
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_json(cls, text: str) -> Optional[str]:
        """Extract answer from JSON code blocks or inline JSON objects."""
        # 1. Code blocks ```json { ... } ```
        code_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        for cb in reversed(code_blocks):
            try:
                data = json.loads(cb)
                if isinstance(data, dict):
                    for k in ["answer", "final_answer", "result", "solution", "value"]:
                        if k in data:
                            return str(data[k]).strip()
            except Exception:
                pass

        # 2. Inline JSON regex
        json_matches = re.findall(
            r'\{[^{}]*"(?:answer|final_answer|result|solution|value)"\s*:\s*([^,{}]+)\}',
            text
        )
        if json_matches:
            cand = json_matches[-1].strip().strip('"\'')
            if cand:
                return cand

        return None

    # ------------------------------------------------------------------ #
    # Tier 3: Anchor Delimiters
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_anchors(cls, text: str) -> Optional[str]:
        """Extract value after standard reasoning delimiters."""
        anchor_patterns = [
            r'####\s*([^\n\r]+)',
            r'(?:[Ff]inal\s+[Aa]nswer|[Aa]nswer)\s*(?:is\s*:?|:)\s*([^\n\r]+)',
            r'(?:[Tt]herefore|[Tt]hus|[Hh]ence),?\s*(?:the\s+answer\s+is|x\s*=)\s*([^\n\r]+)',
            r'(?:[Rr]esult|[Ss]olution)\s*(?:is\s*:?|:)\s*([^\n\r]+)',
        ]

        for pattern in anchor_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                cand = matches[-1].group(1).strip()
                cand = cls._strip_formatting_and_units(cand)
                if cand:
                    return cand

        return None

    # ------------------------------------------------------------------ #
    # Tier 4: Equation Tail Parser
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_equation_tail(cls, text: str) -> Optional[str]:
        """Extract value from equation assignments near the end (e.g. x = 42, ans = 105)."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        # Check last 3 non-empty lines
        for line in reversed(lines[-3:]):
            eq_match = re.search(r'(?:^|[,\s])(?:[a-zA-Z_]\w*|ans|result)\s*=\s*([-+]?\d*\.?\d+(?:/\d+)?|[a-zA-Z0-9_\^\+\-\*/\(\)\\]+)\.?$', line)
            if eq_match:
                cand = eq_match.group(1).strip()
                cand = cls._strip_formatting_and_units(cand)
                if cand:
                    return cand

        return None

    # ------------------------------------------------------------------ #
    # Tier 5 & 6: Numeric Fallback & Formatting Stripping
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_numeric_fallback(cls, text: str) -> Optional[str]:
        """Extract rightmost numeric scalar, fraction, or scientific notation."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        # Try to find numeric token in last line first, then backwards
        for line in reversed(lines):
            # Clean common markdown and currency
            cleaned_line = cls._strip_formatting_and_units(line)
            # Find all numbers / fractions / scientific tokens
            tokens = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:/\d+)?', cleaned_line)
            if tokens:
                return tokens[-1].strip()

        return None

    @classmethod
    def _strip_formatting_and_units(cls, val: str) -> str:
        """Strip formatting, Markdown, math delimiters, currency, thousands commas, and unit words."""
        s = val.strip()

        # Strip leading colons or equals signs if captured from anchor
        s = re.sub(r'^[:=\s]+', '', s).strip()

        # Strip Markdown bold, italic, code
        s = re.sub(r'\*\*(.*?)\*\*', r'\1', s).strip()
        s = re.sub(r'\*(.*?)\*', r'\1', s).strip()
        s = re.sub(r'`(.*?)`', r'\1', s).strip()

        # Strip LaTeX math delimiters $...$ or $$...$$
        s = s.strip("$").strip()

        # Strip currency symbols
        s = re.sub(r'^[\\\$£€¥]+', '', s).strip()

        # Remove thousands commas if it's a numeric quantity (e.g. 1,250,000.50 -> 1250000.50)
        # Avoid breaking sets like {1, 2, 3} or expressions
        if not (s.startswith("{") and s.endswith("}")):
            s = re.sub(r'(?<=\d),(?=\d{3}(?:\b|\D))', '', s)

        # Extract leading numeric scalar if followed by trailing words/units (e.g. "15 animals" -> "15", "100 meters." -> "100")
        lead_num = re.match(r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:/\d+)?)\s+[a-zA-Z%]+', s)
        if lead_num:
            return lead_num.group(1).strip()

        # Strip common trailing units
        s = re.sub(
            r'\s*(?:meters|meter|seconds|second|joules|joule|kg|g|cm|mm|units|%)\.?$',
            '',
            s,
            flags=re.IGNORECASE
        ).strip()

        # Strip trailing punctuation (period, colon, semicolon)
        s = re.sub(r'[.,;:]+$', '', s).strip()

        return s

    # ------------------------------------------------------------------ #
    # Tier 7: Expected Type Post-Processing
    # ------------------------------------------------------------------ #

    @classmethod
    def _post_process(cls, val: str, expected_type: Optional[str] = None) -> str:
        """Apply final clean-up and conditioning based on expected_type."""
        cleaned = val.strip()

        # Remove outer \boxed if still present
        if cleaned.startswith(r"\boxed{") and cleaned.endswith("}"):
            inner = cls._extract_boxed(cleaned)
            if inner is not None:
                cleaned = inner.strip()

        # Strip surrounding markdown quotes/backticks
        cleaned = re.sub(r'^[\'"`*]+|[\'"`*]+$', '', cleaned).strip()

        if expected_type == "set":
            # If set elements are separated by comma without braces, wrap in braces
            if not (cleaned.startswith("{") and cleaned.endswith("}")) and "," in cleaned:
                cleaned = f"{{{cleaned}}}"

        return cleaned
