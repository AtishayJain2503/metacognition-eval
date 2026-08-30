"""
nemo_eval.tools.tabular
=======================
Hermetic tabular data engine for CSV, Parquet, and JSONL inspection,
summary statistics profiling, head/tail sampling, and SQLite relational bridging.
"""

import csv
import io
import math
import os
import sqlite3
import time
from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from nemo_eval.tools.diagnostics import DiagnosticClassifier
from nemo_eval.tools.schemas import DiagnosticError, ToolResult


class TabularColumnSummary(BaseModel):
    """Statistical and distributional summary of a single tabular column."""
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    # Numeric stats
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[Any] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    max: Optional[Any] = None
    # Categorical stats
    unique_count: Optional[int] = None
    top_value: Optional[Any] = None
    top_frequency: Optional[int] = None
    sample_values: Optional[List[Any]] = None
    # Datetime stats
    min_date: Optional[str] = None
    max_date: Optional[str] = None


class TabularSchemaInfo(BaseModel):
    """Structural metadata and memory footprint of a tabular dataset."""
    file_path: str
    file_format: Literal["csv", "parquet", "jsonl", "unknown"]
    file_size_bytes: int
    memory_usage_bytes: int
    memory_usage_human: str
    shape: Dict[str, int]
    columns: List[Dict[str, Any]]


class TabularSummaryInfo(BaseModel):
    """Full statistical profile report for all columns in a dataset."""
    file_path: str
    shape: Dict[str, int]
    column_summaries: Dict[str, TabularColumnSummary]


class TabularSampleInfo(BaseModel):
    """Sample records and markdown table representation."""
    file_path: str
    action: Literal["head", "tail"]
    n_rows_requested: int
    n_rows_returned: int
    total_rows: int
    columns: List[str]
    records: List[Dict[str, Any]]
    markdown_table: str


class TabularEngine:
    """
    Hermetic tabular processing engine providing multi-format dataset ingestion,
    schema inspection, statistical profiling, head/tail sampling, and relational bridging.
    """

    @staticmethod
    def _format_bytes_human(size_bytes: int) -> str:
        """Format raw byte counts into human-readable strings."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / (1024 ** 2):.2f} MB"
        else:
            return f"{size_bytes / (1024 ** 3):.2f} GB"

    @staticmethod
    def _detect_format(file_path: str) -> Literal["csv", "parquet", "jsonl", "unknown"]:
        """Determine tabular format from file extension."""
        lower = file_path.lower()
        if lower.endswith((".csv", ".tsv", ".txt")):
            return "csv"
        elif lower.endswith((".parquet", ".pq")):
            return "parquet"
        elif lower.endswith((".jsonl", ".ndjson")):
            return "jsonl"
        return "unknown"

    @classmethod
    def load_dataset(cls, file_path: str) -> pd.DataFrame:
        """
        Load tabular dataset from local file path with auto-delimiter sniffing and encoding fallback.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tabular dataset file not found: {file_path}")

        fmt = cls._detect_format(file_path)
        if fmt == "parquet":
            return pd.read_parquet(file_path, engine="pyarrow")
        elif fmt == "jsonl":
            return pd.read_json(file_path, lines=True)
        elif fmt == "csv" or fmt == "unknown":
            # Sniff delimiter
            delimiter = ","
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    sample = f.read(16384)
                    if sample:
                        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                        delimiter = dialect.delimiter
            except Exception:
                pass

            encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
            for enc in encodings:
                try:
                    return pd.read_csv(file_path, sep=delimiter, encoding=enc, low_memory=False)
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    raise e
            raise ValueError(f"Failed to decode CSV file '{file_path}' with supported encodings.")
        else:
            raise ValueError(f"Unsupported tabular file format for: '{file_path}'")

    @classmethod
    def inspect_schema(cls, file_path: str) -> TabularSchemaInfo:
        """Extract dataset dimensions, memory footprint, column dtypes, and null counts."""
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        df = cls.load_dataset(file_path)
        mem_bytes = int(df.memory_usage(deep=True).sum())
        total_rows = len(df)

        columns_meta: List[Dict[str, Any]] = []
        for col in df.columns:
            non_null = int(df[col].count())
            null_cnt = int(df[col].isna().sum())
            null_pct = round((null_cnt / total_rows * 100.0), 2) if total_rows > 0 else 0.0
            columns_meta.append({
                "name": str(col),
                "dtype": str(df[col].dtype),
                "non_null_count": non_null,
                "null_count": null_cnt,
                "null_percentage": null_pct
            })

        return TabularSchemaInfo(
            file_path=file_path,
            file_format=cls._detect_format(file_path),
            file_size_bytes=file_size,
            memory_usage_bytes=mem_bytes,
            memory_usage_human=cls._format_bytes_human(mem_bytes),
            shape={"rows": total_rows, "columns": len(df.columns)},
            columns=columns_meta
        )

    @classmethod
    def profile_summary(cls, file_path: str) -> TabularSummaryInfo:
        """Compute comprehensive 8-point numerical and categorical statistical profile."""
        df = cls.load_dataset(file_path)
        total_rows = len(df)
        summaries: Dict[str, TabularColumnSummary] = {}

        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)
            non_null = int(series.count())
            null_cnt = int(series.isna().sum())
            null_pct = round((null_cnt / total_rows * 100.0), 2) if total_rows > 0 else 0.0

            summary = TabularColumnSummary(
                name=str(col),
                dtype=dtype_str,
                non_null_count=non_null,
                null_count=null_cnt,
                null_percentage=null_pct
            )

            if np.issubdtype(series.dtype, np.number) and not pd.api.types.is_bool_dtype(series):
                # Numerical profiling
                valid_num = series.dropna()
                if len(valid_num) > 0:
                    summary.mean = round(float(valid_num.mean()), 4)
                    summary.std = round(float(valid_num.std()), 4) if len(valid_num) > 1 else 0.0
                    summary.min = float(valid_num.min())
                    summary.p25 = round(float(valid_num.quantile(0.25)), 4)
                    summary.p50 = round(float(valid_num.median()), 4)
                    summary.p75 = round(float(valid_num.quantile(0.75)), 4)
                    summary.max = float(valid_num.max())
            elif pd.api.types.is_datetime64_any_dtype(series):
                # Datetime profiling
                valid_dates = series.dropna()
                if len(valid_dates) > 0:
                    summary.min_date = str(valid_dates.min())
                    summary.max_date = str(valid_dates.max())
            else:
                # Categorical / String / Boolean profiling
                valid_cat = series.dropna()
                summary.unique_count = int(valid_cat.nunique())
                if len(valid_cat) > 0:
                    vc = valid_cat.value_counts()
                    summary.top_value = str(vc.index[0])
                    summary.top_frequency = int(vc.iloc[0])
                    summary.sample_values = [str(x) for x in valid_cat.unique()[:5]]

            summaries[str(col)] = summary

        return TabularSummaryInfo(
            file_path=file_path,
            shape={"rows": total_rows, "columns": len(df.columns)},
            column_summaries=summaries
        )

    @classmethod
    def get_sample(
        cls, 
        file_path: str, 
        action: Literal["head", "tail"] = "head", 
        n_rows: int = 5
    ) -> TabularSampleInfo:
        """Retrieve bounded head/tail sample records with markdown table formatting."""
        df = cls.load_dataset(file_path)
        total_rows = len(df)
        clamped_n = max(1, min(n_rows, 100))

        sub_df = df.head(clamped_n) if action == "head" else df.tail(clamped_n)
        
        # Sanitize records to JSON-serializable types
        records: List[Dict[str, Any]] = []
        for _, row in sub_df.iterrows():
            rec = {}
            for col in sub_df.columns:
                val = row[col]
                if pd.isna(val):
                    rec[str(col)] = None
                elif isinstance(val, (np.floating, float)):
                    rec[str(col)] = None if math.isnan(val) or math.isinf(val) else float(val)
                elif isinstance(val, (np.integer, int)):
                    rec[str(col)] = int(val)
                elif isinstance(val, (np.bool_, bool)):
                    rec[str(col)] = bool(val)
                else:
                    rec[str(col)] = str(val)
            records.append(rec)

        # Markdown representation
        try:
            md_table = sub_df.to_markdown(index=False)
            if not md_table or "|" not in md_table:
                raise ValueError("to_markdown did not produce pipe table")
        except Exception:
            headers = [str(c) for c in sub_df.columns]
            header_row = "| " + " | ".join(headers) + " |"
            sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
            rows_md = []
            for _, row in sub_df.iterrows():
                vals = [str(v) if not pd.isna(v) else "" for v in row]
                rows_md.append("| " + " | ".join(vals) + " |")
            md_table = "\n".join([header_row, sep_row] + rows_md)

        return TabularSampleInfo(
            file_path=file_path,
            action=action,
            n_rows_requested=n_rows,
            n_rows_returned=len(records),
            total_rows=total_rows,
            columns=[str(c) for c in df.columns],
            records=records,
            markdown_table=md_table
        )

    @classmethod
    def load_to_sqlite(
        cls, 
        file_path: str, 
        table_name: str, 
        conn: sqlite3.Connection, 
        if_exists: Literal["fail", "replace", "append"] = "replace"
    ) -> int:
        """Load tabular dataset into a target SQLite connection."""
        df = cls.load_dataset(file_path)
        # Sanitize column names for SQL safety
        df.columns = [str(c).strip().replace(" ", "_").replace('"', '').replace("'", "") for c in df.columns]
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        return len(df)

    @classmethod
    def inspect_tool(
        cls,
        file_path: str,
        action: Literal["schema", "summary", "head", "tail"] = "schema",
        n_rows: int = 5
    ) -> ToolResult:
        """High-level tool wrapper returning standard ToolResult envelope."""
        start_perf = time.perf_counter()
        try:
            if action == "schema":
                res = cls.inspect_schema(file_path)
                data = res.model_dump()
            elif action == "summary":
                res = cls.profile_summary(file_path)
                data = res.model_dump()
            elif action in ("head", "tail"):
                res = cls.get_sample(file_path, action=action, n_rows=n_rows)
                data = res.model_dump()
            else:
                raise ValueError(f"Unsupported action: '{action}'. Choose from ['schema', 'summary', 'head', 'tail'].")

            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            return ToolResult(
                status="success",
                execution_time_ms=round(elapsed_ms, 3),
                data=data,
                stdout="",
                stderr=""
            )
        except BaseException as e:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            diag = DiagnosticClassifier.create_diagnostic_error(
                exc=e,
                context={"file_path": file_path, "action": action}
            )
            return ToolResult(
                status="error",
                execution_time_ms=round(elapsed_ms, 3),
                data=None,
                stdout="",
                stderr="",
                error=diag
            )
