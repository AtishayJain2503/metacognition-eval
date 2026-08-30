"""
Unit tests for nemo_eval.agents.planner (Milestone 4 — [T.D] Task Decomposition).
"""
import pytest
from unittest.mock import MagicMock

from nemo_eval.agents.planner import TaskPlanner, PlannerConfig, SubGoal, TaskPlan, PlanningMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_model(json_response: str):
    """Return a mock LLM client that returns json_response as content."""
    model = MagicMock()
    model.generate.return_value = MagicMock(content=json_response)
    return model


GOOD_PLAN_JSON = """
{
  "sub_goals": [
    {"id": "sg_1", "description": "Inspect schema of the database.", "tool_hint": "sqlite_schema", "depends_on": [], "expected_output_type": "dict"},
    {"id": "sg_2", "description": "Query total revenue per region.", "tool_hint": "sqlite_query", "depends_on": ["sg_1"], "expected_output_type": "sql_rows"},
    {"id": "sg_3", "description": "Compute percentage contribution per region.", "tool_hint": "python_repl", "depends_on": ["sg_2"], "expected_output_type": "dataframe"},
    {"id": "sg_4", "description": "Format and return top-3 regions.", "tool_hint": "python_repl", "depends_on": ["sg_3"], "expected_output_type": "string"}
  ]
}
"""

CYCLE_PLAN_JSON = """
{
  "sub_goals": [
    {"id": "sg_1", "description": "Step 1", "depends_on": ["sg_2"]},
    {"id": "sg_2", "description": "Step 2", "depends_on": ["sg_1"]}
  ]
}
"""

EMPTY_PLAN_JSON = "{}"
MALFORMED_JSON = "This is not JSON at all!"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTaskPlanner:

    def test_successful_decomposition(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose(task_id="t_001", query="What are the top 3 revenue regions?")

        assert isinstance(plan, TaskPlan)
        assert plan.task_id == "t_001"
        assert len(plan.sub_goals) == 4
        assert plan.execution_order == ["sg_1", "sg_2", "sg_3", "sg_4"]
        assert plan.metrics is not None

    def test_sub_goal_ids_correct(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_002", "query")
        ids = [sg.id for sg in plan.sub_goals]
        assert ids == ["sg_1", "sg_2", "sg_3", "sg_4"]

    def test_tool_hints_preserved(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_003", "query")
        gm = plan.goal_map
        assert gm["sg_1"].tool_hint == "sqlite_schema"
        assert gm["sg_2"].tool_hint == "sqlite_query"
        assert gm["sg_3"].tool_hint == "python_repl"

    def test_dependencies_respected_in_order(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_004", "query")
        order = plan.execution_order
        for sg in plan.sub_goals:
            sg_pos = order.index(sg.id)
            for dep in sg.depends_on:
                assert order.index(dep) < sg_pos

    def test_s_topo_perfect_chain(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_005", "query")
        assert plan.metrics.topological_score == 1.0

    def test_p_dep_all_valid(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_006", "query")
        assert plan.metrics.dependency_precision == 1.0

    def test_composite_score_range(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_007", "query")
        assert 0.0 <= plan.metrics.composite_score <= 1.0

    def test_cycle_detection(self):
        model = _make_mock_model(CYCLE_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_008", "query")
        assert plan.metrics.has_cycles is True

    def test_cycle_lowers_structural_score(self):
        model = _make_mock_model(CYCLE_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_009", "query")
        # Cyclic plan should have reduced S_struct
        assert plan.metrics.structural_score < 1.0

    def test_empty_plan_fallback(self):
        model = _make_mock_model(EMPTY_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_010", "query")
        assert len(plan.sub_goals) >= 1
        assert plan.execution_order  # non-empty

    def test_malformed_json_fallback(self):
        model = _make_mock_model(MALFORMED_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_011", "query")
        assert len(plan.sub_goals) == 1
        assert plan.sub_goals[0].id == "sg_1"

    def test_markdown_fenced_json(self):
        fenced = "```json\n" + GOOD_PLAN_JSON + "\n```"
        model = _make_mock_model(fenced)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_012", "query")
        assert len(plan.sub_goals) == 4

    def test_max_sub_goals_config(self):
        """Respect max_sub_goals config."""
        model = _make_mock_model(GOOD_PLAN_JSON)
        config = PlannerConfig(max_sub_goals=2)
        planner = TaskPlanner(model_client=model, config=config)
        plan = planner.decompose("t_013", "query")
        assert len(plan.sub_goals) <= 2

    def test_planning_duration_positive(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_014", "query")
        assert plan.planning_duration_ms >= 0.0

    def test_ordered_sub_goals_returns_list(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_015", "query")
        ordered = plan.ordered_sub_goals()
        assert len(ordered) == len(plan.sub_goals)
        assert all(isinstance(sg, SubGoal) for sg in ordered)

    def test_node_edge_count(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_016", "query")
        assert plan.metrics.node_count == 4
        assert plan.metrics.edge_count == 3  # sg2→sg1, sg3→sg2, sg4→sg3

    def test_max_depth_linear_chain(self):
        model = _make_mock_model(GOOD_PLAN_JSON)
        planner = TaskPlanner(model_client=model)
        plan = planner.decompose("t_017", "query")
        assert plan.metrics.max_depth == 3  # depth 0,1,2,3


class TestSubGoal:
    def test_has_dependencies(self):
        sg = SubGoal(id="sg_1", description="test", depends_on=["sg_0"])
        assert sg.has_dependencies is True

    def test_no_dependencies(self):
        sg = SubGoal(id="sg_1", description="test", depends_on=[])
        assert sg.has_dependencies is False
