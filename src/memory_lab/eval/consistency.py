"""
Multi-turn conversation consistency check for the memory-lab agent.

Runs scripted conversations through the real LangGraph memory agent (short-
term memory via the graph's checkpointer, long-term memory via ChromaDB)
and checks whether later "recall" turns correctly reference a fact stated
several turns earlier in the *same* conversation -- i.e. whether the agent
stays consistent with its own conversation history, not whether any single
response looks plausible on its own.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
from langchain_core.messages import HumanMessage
from llmops_common.logging.mlflow_logger import MLflowLogger
from llmops_common.stats.proportions import (
    format_proportion_with_ci,
    wilson_score_interval,
)
from pydantic import BaseModel

from memory_lab.agents.graph import build_memory_graph


class ConsistencyTurn(BaseModel):
    """One message in a scripted conversation.

    `expected_facts` marks this as a *recall* turn: the response is checked
    for at least one of these strings (case-insensitive). Turns without
    `expected_facts` are filler/state turns and aren't scored.
    """

    message: str
    expected_facts: list[str] | None = None


class ConsistencyScenario(BaseModel):
    """A full scripted conversation: a name, a user_id (facts are namespaced
    by user_id in long-term memory), and an ordered list of turns."""

    name: str
    user_id: str
    turns: list[ConsistencyTurn]


class TurnResult(BaseModel):
    """The real response for one turn, and whether it was consistent (only
    meaningful for recall turns -- `None` for filler/state turns)."""

    scenario_name: str
    turn_index: int
    message: str
    response: str
    expected_facts: list[str] | None
    consistent: bool | None


class ConsistencyReport(BaseModel):
    turn_results: list[TurnResult]
    consistent_count: int
    total_recall_turns: int
    consistency_rate: float
    consistency_rate_ci_lower: float
    consistency_rate_ci_upper: float


def load_scenarios(path: Path) -> list[ConsistencyScenario]:
    if not path.exists():
        raise FileNotFoundError(f"Consistency scenarios not found at: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ConsistencyScenario(**item) for item in raw]


def run_scenario(graph: Any, scenario: ConsistencyScenario) -> list[TurnResult]:
    """Runs one scenario's turns in order, on a single conversation thread,
    so later turns can only recall facts via the agent's own memory (short-
    or long-term) -- never via the caller re-stating them."""
    config = {"configurable": {"thread_id": f"consistency_{scenario.name}"}}
    results = []

    for index, turn in enumerate(scenario.turns):
        state = {
            "messages": [HumanMessage(content=turn.message)],
            "user_id": scenario.user_id,
        }
        outcome = graph.invoke(state, config=config)
        response = outcome["messages"][-1].content

        consistent = None
        if turn.expected_facts:
            lowered_response = response.lower()
            consistent = any(
                fact.lower() in lowered_response for fact in turn.expected_facts
            )

        results.append(
            TurnResult(
                scenario_name=scenario.name,
                turn_index=index,
                message=turn.message,
                response=response,
                expected_facts=turn.expected_facts,
                consistent=consistent,
            )
        )

    return results


def run_consistency_suite(
    graph: Any, scenarios: Sequence[ConsistencyScenario]
) -> ConsistencyReport:
    """Runs every scenario and reports a Wilson-CI consistency rate over all
    recall turns combined.

    Raises:
        ValueError: if no scenario contains a recall turn (nothing to score).
    """
    all_results: list[TurnResult] = []
    for scenario in scenarios:
        all_results.extend(run_scenario(graph, scenario))

    recall_results = [r for r in all_results if r.consistent is not None]
    total_recall_turns = len(recall_results)
    if total_recall_turns == 0:
        raise ValueError(
            "No recall turns (turns with expected_facts) found across scenarios"
        )
    consistent_count = sum(1 for r in recall_results if r.consistent)

    ci = wilson_score_interval(consistent_count, total_recall_turns)

    return ConsistencyReport(
        turn_results=all_results,
        consistent_count=consistent_count,
        total_recall_turns=total_recall_turns,
        consistency_rate=consistent_count / total_recall_turns,
        consistency_rate_ci_lower=ci.lower,
        consistency_rate_ci_upper=ci.upper,
    )


class ConsistencyRunner:
    """Orchestrates the multi-turn consistency suite against the real
    memory-lab agent and logs a Wilson-CI consistency-rate metric to
    MLflow, mirroring the other labs' batch-run reporting pattern."""

    def __init__(self, graph: Any | None = None, data_path: Path | None = None):
        self.graph = graph or build_memory_graph()
        if data_path is None:
            self.data_path = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "consistency"
                / "scenarios.json"
            )
        else:
            self.data_path = Path(data_path)

        self.mlflow_logger = MLflowLogger(experiment_name="memory_lab_consistency")

    def load_scenarios(self) -> list[ConsistencyScenario]:
        return load_scenarios(self.data_path)

    def run(self, run_name: str = "multi_turn_consistency") -> ConsistencyReport:
        scenarios = self.load_scenarios()

        self.mlflow_logger.start_trace(run_name=run_name)
        mlflow.log_param("scenario_count", len(scenarios))

        report = run_consistency_suite(self.graph, scenarios)

        mlflow.log_metrics(
            {
                "consistency_rate": report.consistency_rate,
                "consistency_rate_ci_lower": report.consistency_rate_ci_lower,
                "consistency_rate_ci_upper": report.consistency_rate_ci_upper,
                "consistent_count": report.consistent_count,
                "total_recall_turns": report.total_recall_turns,
            }
        )
        self.mlflow_logger.end_trace()

        print(
            "Multi-turn consistency: "
            + format_proportion_with_ci(
                report.consistent_count, report.total_recall_turns
            )
        )
        return report


if __name__ == "__main__":
    ConsistencyRunner().run()
