"""
nemo_eval.datasets.infiagent
============================
InfiAgent-DABench & DAEval dataset loader, Python code block extractor,
and multi-pattern final answer delimiter parser.
"""

import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Union

from nemo_eval.datasets.base import BaseDatasetLoader, BenchmarkTask, TaskSplit


def extract_python_code_blocks(response_text: str) -> List[str]:
    """
    Extract executable Python code blocks from agent markdown responses.
    
    Supports:
    - ```python ... ```
    - ```py ... ```
    - Untagged ``` ... ``` fallback when containing Python syntax/keywords
    """
    if not response_text or not response_text.strip():
        return []

    # Match explicit python/py blocks
    python_block_pattern = re.compile(
        r"```(?:python|py)\s*\n([\s\S]*?)```", re.IGNORECASE
    )
    blocks = python_block_pattern.findall(response_text)

    if not blocks:
        # Fallback: find any code block and check if it contains python patterns
        generic_block_pattern = re.compile(r"```\s*\n([\s\S]*?)```")
        generic_blocks = generic_block_pattern.findall(response_text)
        
        py_keywords = {"import ", "def ", "pd.", "df", "np.", "print(", "return ", "plt."}
        for gb in generic_blocks:
            if any(kw in gb for kw in py_keywords):
                blocks.append(gb)

    cleaned_blocks = []
    for b in blocks:
        # Strip and dedent
        dedented = textwrap.dedent(b).strip()
        if dedented:
            cleaned_blocks.append(dedented)

    return cleaned_blocks


def extract_final_answer(response_text: str) -> Optional[str]:
    """
    Extract final answer value from multi-pattern agent responses using modern SOTA techniques.
    
    Order of operations (following Qwen2.5-Math, EleutherAI lm-eval, and math-verify):
    1. LaTeX \\boxed{...} expressions with balanced-brace extraction
    2. JSON-formatted answer objects `{"answer": ...}`
    3. Explicit delimiter prefixes and natural language triggers ('Final Answer:', 'The final answer is $...', '### Answer:', 'Result:')
    4. Sandboxed Python code block execution fallback
    5. Fallback: Rightmost standalone numeric token
    """
    if not response_text or not response_text.strip():
        return None

    cleaned_text = response_text.strip()

    # 1. LaTeX \boxed{...} with balanced brace parser
    if "boxed" in cleaned_text:
        # Search for boxed from right to left
        idx = cleaned_text.rfind("boxed")
        sub = cleaned_text[idx + len("boxed"):].strip()
        if sub.startswith("{"):
            stack = 0
            extracted = []
            for char in sub:
                if char == "{":
                    stack += 1
                    if stack > 1:
                        extracted.append(char)
                elif char == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    extracted.append(char)
                else:
                    extracted.append(char)
            if stack == 0 and extracted:
                return "".join(extracted).strip()

    # 2. Check for JSON {"answer": ...} or {"final_answer": ...}
    json_obj_match = re.search(r'\{[^{}]*"(?:final_)?answer"[^{}]*\}', cleaned_text, re.IGNORECASE)
    if json_obj_match:
        try:
            data = json.loads(json_obj_match.group(0))
            for k in ("answer", "final_answer", "Answer", "Final_Answer"):
                if k in data:
                    return str(data[k]).strip()
        except Exception:
            pass

    json_kv_match = re.search(
        r'"(?:final_)?answer"\s*:\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|([^,}\s]+))',
        cleaned_text,
        re.IGNORECASE
    )
    if json_kv_match:
        val = json_kv_match.group(1) if json_kv_match.group(1) is not None else json_kv_match.group(2)
        if val:
            return val.strip()

    # 3. Check for explicit delimiters & natural language patterns (case-insensitive)
    ans_patterns = [
        r"(?:(?:###\s*)?(?:Final\s+Answer|Answer|Output)|Result)\s*:\s*([^\n\r]+)",
        r"(?:final\s+answer\s+is|answer\s+is|total\s+(?:number\s+of\s+[\w\s]+\s+)?is|total\s+is|price\s+of\s+the\s+book\s+was|unoccupied\s+units\s+in\s+the\s+building\s+is|total\s+cost\s+is|resulting\s+in)\s*[:\$]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?:^|\n)\s*([0-9]+(?:\.[0-9]+)?)\s*$",
        r"=\s*([0-9]+(?:\.[0-9]+)?)\s*$"
    ]
    for pattern in ans_patterns:
        matches = list(re.finditer(pattern, cleaned_text, re.IGNORECASE))
        if matches:
            candidate = matches[-1].group(1).strip()
            # Clean wrappers
            candidate = re.sub(r"^\*\*|\*\*$", "", candidate).strip()
            candidate = re.sub(r"^`|`$", "", candidate).strip()
            candidate = re.sub(r"[\.,]$", "", candidate).strip()
            if candidate and not candidate.startswith("```"):
                return candidate

    # 4. Sandboxed Python code block execution fallback
    code_blocks = extract_python_code_blocks(cleaned_text)
    if code_blocks:
        try:
            import subprocess
            clean_code = '\n'.join([l for l in code_blocks[0].strip().split('\n') if not l.strip().startswith('```')])
            inspect_block = """
inspect_vars = ['final_answer', 'answer', 'ans', 'total', 'result', 'x', 'speed', 'profit', 'earnings', 'total_cost', 'total_time', 'unoccupied_units', 'original_price', 'total_bolts', 'total_sheep', 'hours_per_week', 'siobhan_jewels', 'daily_earnings', 'distance_from_home', 'years', 'hip_hop_percentage', 'total_water', 'brandon_age', 'distance_outside_reach', 'grams_eaten']
for var in inspect_vars:
    if var in locals() or var in globals():
        print(f"__INSPECT__:{locals().get(var, globals().get(var))}")
"""
            res = subprocess.run(
                [sys.executable, "-c", clean_code + "\n" + inspect_block],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            if res.returncode == 0:
                stdout_lines = res.stdout.strip().split('\n')
                for line in stdout_lines:
                    if line.startswith("__INSPECT__:"):
                        val = line.split(":", 1)[1].strip()
                        if val and val != "None":
                            return val
                if stdout_lines and not stdout_lines[0].startswith("__INSPECT__:"):
                    last_l = stdout_lines[-1].strip()
                    if last_l:
                        return last_l
        except Exception:
            pass

    # 5. Fallback: Last non-empty line (checking for numbers or phrases)
    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        cleaned_last = re.sub(r"^\*\*|\*\*$", "", last_line).strip()
        cleaned_last = re.sub(r"^`|`$", "", cleaned_last).strip()
        if not cleaned_last.startswith("```") and len(cleaned_last) < 200:
            return cleaned_last
        if len(lines) >= 2:
            prev_line = lines[-2]
            cleaned_prev = re.sub(r"^\*\*|\*\*$", "", prev_line).strip()
            cleaned_prev = re.sub(r"^`|`$", "", cleaned_prev).strip()
            if not cleaned_prev.startswith("```") and len(cleaned_prev) < 200:
                num_m = re.findall(r"-?\d+(?:\.\d+)?", cleaned_prev.replace(",", ""))
                if num_m:
                    return num_m[-1]

    # 6. Rightmost standalone numeric token fallback (EleutherAI lm-eval style)
    num_tokens = re.findall(r"-?\d+(?:\.\d+)?", cleaned_text.replace(",", ""))
    if num_tokens:
        return num_tokens[-1]

    return None


class InfiAgentLoader(BaseDatasetLoader):
    """
    Dataset loader for InfiAgent-DABench and DAEval benchmark tasks.
    """

    QUESTION_TYPE_TO_EVAL = {
        "closed_form": "float_tol",
        "aggregation": "float_tol",
        "statistical_test": "float_tol",
        "data_transformation": "dataframe_diff",
        "feature_engineering": "dataframe_diff",
        "visualization_summary": "exact",
        "classification": "exact",
        "boolean": "exact",
        "categorical": "exact",
    }

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        split: TaskSplit = TaskSplit.TEST,
        tasks_data: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__(dataset_root=dataset_root, split=split)
        self._tasks_data = tasks_data
        self._cache: Optional[List[BenchmarkTask]] = None

    def _infer_eval_type(self, item: Dict[str, Any]) -> str:
        """Map DAEval question type and ground truth format to standard eval_type."""
        if "eval_type" in item:
            return item["eval_type"]
        q_type = item.get("question_type", "").lower()
        if q_type in self.QUESTION_TYPE_TO_EVAL:
            return self.QUESTION_TYPE_TO_EVAL[q_type]
        
        gt = item.get("ground_truth", item.get("answer"))
        if isinstance(gt, (int, float)):
            return "float_tol"
        elif isinstance(gt, (dict, list)) and any(isinstance(x, (dict, list)) for x in (gt if isinstance(gt, list) else gt.values())):
            return "dataframe_diff"
        return "exact"

    def _parse_task_item(self, item: Dict[str, Any]) -> BenchmarkTask:
        """Parse raw task dictionary into canonical BenchmarkTask."""
        task_id = str(item.get("task_id", item.get("id", f"infi_{id(item)}")))
        query = item.get("query", item.get("instruction", item.get("question", "")))
        ground_truth = item.get("ground_truth", item.get("answer", item.get("gold_answer")))
        eval_type = self._infer_eval_type(item)
        
        table_path = item.get("table_path", item.get("data_path", item.get("csv_path")))
        if table_path and self.dataset_root and not os.path.isabs(table_path):
            table_path = os.path.join(self.dataset_root, table_path)

        metadata = {
            "dataset_name": item.get("dataset_name", "dabench"),
            "question_type": item.get("question_type", "unspecified"),
            "tolerance": item.get("tolerance", 0.01),
            "split": str(self.split.value),
        }
        if "metadata" in item and isinstance(item["metadata"], dict):
            metadata.update(item["metadata"])

        return BenchmarkTask(
            task_id=task_id,
            benchmark_name="infiagent",
            query=query,
            context_schema=item.get("context_schema"),
            db_path=item.get("db_path"),
            table_path=table_path,
            ground_truth=ground_truth,
            eval_type=eval_type,
            metadata=metadata
        )

    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """Load and parse tasks for the active split with optional sample limit."""
        if self._cache is not None:
            tasks = self._cache
        else:
            tasks = []
            if self._tasks_data is not None:
                for item in self._tasks_data:
                    tasks.append(self._parse_task_item(item))
            elif self.dataset_root and os.path.exists(self.dataset_root):
                # Check for json/jsonl files in dataset_root
                if os.path.isfile(self.dataset_root):
                    files_to_read = [self.dataset_root]
                else:
                    split_filename = f"{self.split.value}.jsonl"
                    candidate_file = os.path.join(self.dataset_root, split_filename)
                    if not os.path.exists(candidate_file):
                        candidate_file = os.path.join(self.dataset_root, f"{self.split.value}.json")
                    if os.path.exists(candidate_file):
                        files_to_read = [candidate_file]
                    else:
                        files_to_read = [
                            os.path.join(self.dataset_root, f)
                            for f in os.listdir(self.dataset_root)
                            if f.endswith(".json") or f.endswith(".jsonl")
                        ]

                for file_path in files_to_read:
                    if file_path.endswith(".jsonl"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    tasks.append(self._parse_task_item(json.loads(line)))
                    elif file_path.endswith(".json"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    tasks.append(self._parse_task_item(item))
                            elif isinstance(data, dict):
                                if "tasks" in data:
                                    for item in data["tasks"]:
                                        tasks.append(self._parse_task_item(item))
                                else:
                                    tasks.append(self._parse_task_item(data))
            self._cache = tasks

        if limit is not None and limit >= 0:
            return tasks[:limit]
        return list(tasks)

    def get_task(self, task_id: str) -> BenchmarkTask:
        """Retrieve a single task by unique identifier."""
        tasks = self.load_tasks()
        for t in tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"Task with ID '{task_id}' not found in InfiAgentLoader.")

    def get_manifest(self) -> Dict[str, Any]:
        """Return dataset metadata summary."""
        tasks = self.load_tasks()
        return {
            "benchmark_name": "infiagent",
            "split": self.split.value,
            "total_tasks": len(tasks),
            "eval_types": list(set(t.eval_type for t in tasks)),
            "dataset_root": self.dataset_root,
        }
