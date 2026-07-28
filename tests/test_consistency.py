import json
from unittest.mock import MagicMock, patch

import pytest

from memory_lab.eval.consistency import (
    ConsistencyRunner,
    ConsistencyScenario,
    ConsistencyTurn,
    load_scenarios,
    run_consistency_suite,
    run_scenario,
)


def _ai_message(content: str):
    """The graph's real return shape only needs `.content` off the last
    message -- a lightweight stand-in, same pattern as test_client.py."""
    return type("Msg", (), {"content": content})()


def _mock_graph(responses: list[str]) -> MagicMock:
    graph = MagicMock()
    graph.invoke.side_effect = [{"messages": [_ai_message(r)]} for r in responses]
    return graph


class TestRunScenario:
    def test_recall_turn_marked_consistent_when_fact_present(self):
        scenario = ConsistencyScenario(
            name="name_recall",
            user_id="u1",
            turns=[
                ConsistencyTurn(message="My name is Elena."),
                ConsistencyTurn(message="What's the weather?"),
                ConsistencyTurn(message="What is my name?", expected_facts=["Elena"]),
            ],
        )
        graph = _mock_graph(
            [
                "Nice to meet you, Elena!",
                "I don't have real-time weather data.",
                "Your name is Elena.",
            ]
        )

        results = run_scenario(graph, scenario)

        assert results[0].consistent is None  # filler/state turn, not scored
        assert results[1].consistent is None
        assert results[2].consistent is True

    def test_recall_turn_marked_inconsistent_when_fact_absent(self):
        scenario = ConsistencyScenario(
            name="name_recall",
            user_id="u1",
            turns=[
                ConsistencyTurn(message="My name is Elena."),
                ConsistencyTurn(message="What is my name?", expected_facts=["Elena"]),
            ],
        )
        graph = _mock_graph(["Nice to meet you!", "I'm not sure, remind me?"])

        results = run_scenario(graph, scenario)

        assert results[1].consistent is False

    def test_fact_match_is_case_insensitive(self):
        scenario = ConsistencyScenario(
            name="location_recall",
            user_id="u1",
            turns=[
                ConsistencyTurn(message="Where do I live?", expected_facts=["Kyoto"]),
            ],
        )
        graph = _mock_graph(["You mentioned you live in KYOTO."])

        results = run_scenario(graph, scenario)

        assert results[0].consistent is True

    def test_all_turns_use_the_same_thread_id(self):
        scenario = ConsistencyScenario(
            name="name_recall",
            user_id="u1",
            turns=[
                ConsistencyTurn(message="Hi"),
                ConsistencyTurn(message="What's my name?", expected_facts=["x"]),
            ],
        )
        graph = _mock_graph(["Hello!", "I don't know."])

        run_scenario(graph, scenario)

        thread_ids = {
            call.kwargs["config"]["configurable"]["thread_id"]
            for call in graph.invoke.call_args_list
        }
        assert thread_ids == {"consistency_name_recall"}


class TestRunConsistencySuite:
    def test_aggregates_wilson_ci_across_scenarios(self):
        scenario_a = ConsistencyScenario(
            name="a",
            user_id="u1",
            turns=[ConsistencyTurn(message="recall a", expected_facts=["yes"])],
        )
        scenario_b = ConsistencyScenario(
            name="b",
            user_id="u2",
            turns=[ConsistencyTurn(message="recall b", expected_facts=["yes"])],
        )
        # 1 of 2 recall turns consistent.
        graph = _mock_graph(["contains yes", "does not match"])

        report = run_consistency_suite(graph, [scenario_a, scenario_b])

        assert report.total_recall_turns == 2
        assert report.consistent_count == 1
        assert report.consistency_rate == pytest.approx(0.5)
        assert report.consistency_rate_ci_lower < report.consistency_rate
        assert report.consistency_rate_ci_upper > report.consistency_rate

    def test_raises_when_no_recall_turns_exist(self):
        scenario = ConsistencyScenario(
            name="a", user_id="u1", turns=[ConsistencyTurn(message="just chatting")]
        )
        graph = _mock_graph(["sure, how can I help?"])

        with pytest.raises(ValueError, match="No recall turns"):
            run_consistency_suite(graph, [scenario])


class TestLoadScenarios:
    def test_reads_json_into_scenario_objects(self, tmp_path):
        data = [
            {
                "name": "s1",
                "user_id": "u1",
                "turns": [
                    {"message": "hi"},
                    {"message": "what's my name?", "expected_facts": ["Sam"]},
                ],
            }
        ]
        path = tmp_path / "scenarios.json"
        path.write_text(json.dumps(data))

        scenarios = load_scenarios(path)

        assert len(scenarios) == 1
        assert scenarios[0].name == "s1"
        assert scenarios[0].turns[1].expected_facts == ["Sam"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scenarios(tmp_path / "nope.json")


class TestConsistencyRunner:
    @patch("memory_lab.eval.consistency.mlflow")
    @patch("memory_lab.eval.consistency.MLflowLogger")
    def test_run_logs_consistency_metrics_to_mlflow(
        self, mock_mlflow_logger_cls, mock_mlflow, tmp_path
    ):
        data = [
            {
                "name": "s1",
                "user_id": "u1",
                "turns": [{"message": "what is my name?", "expected_facts": ["Sam"]}],
            }
        ]
        scenarios_path = tmp_path / "scenarios.json"
        scenarios_path.write_text(json.dumps(data))

        graph = _mock_graph(["Your name is Sam."])
        runner = ConsistencyRunner(graph=graph, data_path=scenarios_path)

        report = runner.run(run_name="test_run")

        runner.mlflow_logger.start_trace.assert_called_once_with(run_name="test_run")
        runner.mlflow_logger.end_trace.assert_called_once()
        mock_mlflow.log_param.assert_called_once_with("scenario_count", 1)
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert logged_metrics["consistency_rate"] == 1.0
        assert logged_metrics["total_recall_turns"] == 1
        assert report.consistency_rate == 1.0
