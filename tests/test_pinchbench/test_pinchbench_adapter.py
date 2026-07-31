from __future__ import annotations

import json

import pytest

from pinchbench.adapter import (
    BenchmarkInfrastructureError,
    PinchBenchRunConfig,
    build_docker_command,
    inspect_score_from_result,
    model_alias,
    parse_pinchbench_results,
    resolve_model_config,
    suite_for_mode,
)


def test_suite_for_mode_defaults() -> None:
    assert suite_for_mode("smoke", None) == "task_sanity"
    assert suite_for_mode("subset", None) == "task_sanity,task_calendar,task_weather"
    assert suite_for_mode("full", None) == "all"
    assert suite_for_mode("full", "custom_task") == "custom_task"


def test_model_alias_removes_provider_separator() -> None:
    assert model_alias("provider/model name") == "provider-model-name"


def test_resolve_model_config_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PINCHBENCH_MODEL_BASE_URL", "https://example.invalid/v1/")
    monkeypatch.setenv("PINCHBENCH_MODEL", "provider/model")
    monkeypatch.delenv("PINCHBENCH_API_KEY", raising=False)
    assert resolve_model_config(PinchBenchRunConfig()) == (
        "https://example.invalid/v1",
        "provider/model",
        None,
    )


def test_resolve_model_config_requires_endpoint_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PINCHBENCH_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("PINCHBENCH_MODEL", raising=False)
    with pytest.raises(BenchmarkInfrastructureError):
        resolve_model_config(PinchBenchRunConfig())


def test_inspect_score_from_result() -> None:
    value, explanation, metadata = inspect_score_from_result(
        {
            "status": "success",
            "score": 0.5,
            "tasks": [{"task_id": "task_sanity"}],
            "mode": "smoke",
        }
    )
    assert value == pytest.approx(0.5)
    assert "PinchBench smoke" in explanation
    assert metadata["score"] == 0.5


def test_parse_pinchbench_results_uses_category_aggregate(tmp_path) -> None:
    """The native summary reports scores per category, not as a top-level field."""
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "category_scores": {
                    "PRODUCTIVITY": {"score": 1.0, "max_score": 1.0},
                    "CODING": {"score": 1.0, "max_score": 3.0},
                },
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )

    assert parse_pinchbench_results(result_path)["score"] == pytest.approx(0.5)


def test_parse_pinchbench_results_falls_back_to_task_grading(tmp_path) -> None:
    """Per-task scores live under "grading", never under a "score" key."""
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "task_sanity",
                        "grading": {
                            "runs": [
                                {"score": 1.0, "max_score": 1.0},
                                {"score": 0.0, "max_score": 1.0},
                            ],
                            "mean": 0.5,
                        },
                    },
                    {"task_id": "task_calendar", "grading": {"runs": [], "mean": 1.0}},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert parse_pinchbench_results(result_path)["score"] == pytest.approx(0.75)


def test_parse_pinchbench_results_scores_zero_without_any_scores(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"tasks": []}), encoding="utf-8")

    assert parse_pinchbench_results(result_path)["score"] == 0.0


def test_build_docker_command_always_forwards_api_key_env(tmp_path) -> None:
    """OpenClaw resolves ${OPENAI_API_KEY} inside the container.

    The variable has to be forwarded even when the endpoint is unauthenticated,
    otherwise provider auth fails and the agent issues no model requests.
    """
    command = build_docker_command(
        PinchBenchRunConfig(),
        tmp_path / "benchmark",
        tmp_path / "run",
        "task_sanity",
        "https://example.invalid/v1",
        "provider/model",
        None,
    )

    assert "OPENAI_API_KEY" in command
    assert "OPENAI_COMPATIBLE_API_KEY" in command
    # Forwarded by name only, so no key value can leak into the process list.
    assert not any(arg.startswith("OPENAI_API_KEY=") for arg in command)


def test_build_docker_command_selects_proxy_port_at_runtime(tmp_path) -> None:
    command = build_docker_command(
        PinchBenchRunConfig(),
        tmp_path / "benchmark",
        tmp_path / "run",
        "task_sanity",
        "https://example.invalid/v1",
        "provider/model",
        None,
    )

    inner = command[-1]
    assert "proxy_port=$(" in inner
    assert "http://127.0.0.1:${proxy_port}/v1" in inner
    assert "http://127.0.0.1:0/v1" not in inner
