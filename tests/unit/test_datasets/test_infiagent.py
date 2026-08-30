"""
tests.unit.test_datasets.test_infiagent
======================================
Unit tests for InfiAgent-DABench response extractors and dataset loader.
"""

import pytest

from nemo_eval.datasets.base import TaskSplit
from nemo_eval.datasets.infiagent import (
    InfiAgentLoader,
    extract_final_answer,
    extract_python_code_blocks,
)


class TestInfiAgentCodeExtraction:
    """Test extracting Python code blocks from agent response text."""

    def test_extract_single_python_block(self):
        text = """Here is the Python script to compute the result:
```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.mean())
```
Done."""
        blocks = extract_python_code_blocks(text)
        assert len(blocks) == 1
        assert "import pandas as pd" in blocks[0]
        assert "print(df.mean())" in blocks[0]

    def test_extract_multiple_code_blocks(self):
        text = """First step:
```python
x = 10
y = 20
```
Second step:
```py
result = x + y
print(result)
```"""
        blocks = extract_python_code_blocks(text)
        assert len(blocks) == 2
        assert blocks[0] == "x = 10\ny = 20"
        assert blocks[1] == "result = x + y\nprint(result)"

    def test_extract_untagged_code_block_fallback(self):
        text = """Executing code:
```
import math
ans = math.sqrt(64)
```"""
        blocks = extract_python_code_blocks(text)
        assert len(blocks) == 1
        assert "math.sqrt(64)" in blocks[0]

    def test_extract_empty_or_no_code(self):
        assert extract_python_code_blocks("") == []
        assert extract_python_code_blocks("No code here, just text.") == []


class TestInfiAgentAnswerExtraction:
    """Test extracting final answers from various delimiter formats."""

    def test_extract_final_answer_delimiters(self):
        assert extract_final_answer("Final Answer: 42.5") == "42.5"
        assert extract_final_answer("### Answer: 1200") == "1200"
        assert extract_final_answer("Answer: California") == "California"
        assert extract_final_answer("Output: 99.9%") == "99.9%"
        assert extract_final_answer("Result: True") == "True"

    def test_extract_boxed_latex_format(self):
        text = "After performing the calculation, we obtain \\boxed{25.4}."
        assert extract_final_answer(text) == "25.4"

    def test_extract_json_format(self):
        text = 'The computed response is: {"answer": "San Francisco", "confidence": 0.95}'
        assert extract_final_answer(text) == "San Francisco"

    def test_extract_fallback_last_line(self):
        text = "Step 1: Analyzed table.\nStep 2: Filtered rows.\n35.8"
        assert extract_final_answer(text) == "35.8"

    def test_extract_empty_text(self):
        assert extract_final_answer("") is None
        assert extract_final_answer("   ") is None


class TestInfiAgentLoader:
    """Test InfiAgentLoader loading from in-memory data and files."""

    def test_load_from_in_memory_data(self):
        sample_data = [
            {
                "task_id": "infi_m1",
                "instruction": "Compute the sum of columns.",
                "answer": 150.0,
                "question_type": "aggregation"
            },
            {
                "task_id": "infi_m2",
                "instruction": "Check if any row is negative.",
                "answer": False,
                "question_type": "boolean"
            }
        ]
        loader = InfiAgentLoader(tasks_data=sample_data)
        tasks = loader.load_tasks()
        assert len(tasks) == 2
        assert tasks[0].task_id == "infi_m1"
        assert tasks[0].eval_type == "float_tol"
        assert tasks[1].task_id == "infi_m2"
        assert tasks[1].eval_type == "exact"

    def test_load_from_jsonl_file(self, mock_infiagent_jsonl):
        loader = InfiAgentLoader(dataset_root=mock_infiagent_jsonl, split=TaskSplit.TEST)
        tasks = loader.load_tasks()
        assert len(tasks) == 2
        assert tasks[0].task_id == "infi_test_001"
        assert tasks[0].eval_type == "float_tol"
        assert tasks[1].task_id == "infi_test_002"
        assert tasks[1].eval_type == "dataframe_diff"

    def test_get_task_and_manifest(self, mock_infiagent_jsonl):
        loader = InfiAgentLoader(dataset_root=mock_infiagent_jsonl, split=TaskSplit.TEST)
        task = loader.get_task("infi_test_001")
        assert task.query == "Compute the average monthly charge for customers."

        manifest = loader.get_manifest()
        assert manifest["benchmark_name"] == "infiagent"
        assert manifest["total_tasks"] == 2
        assert "float_tol" in manifest["eval_types"]
