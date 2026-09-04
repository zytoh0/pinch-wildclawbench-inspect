from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinchbench.adapter import (
    BenchmarkInfrastructureError,
    PinchBenchRunConfig,
    build_docker_command,
    container_api_key,
    container_name,
    inspect_score_from_result,
    list_task_specs,
    model_alias,
    parse_pinchbench_results,
    record_grading_type,
    resolve_judge_model,
    resolve_model_config,
    suite_for_mode,
    task_points,
    task_score,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "pinchbench_skill"


def test_suite_for_mode_defaults() -> None:
    assert suite_for_mode("smoke", None) == "task_sanity"
    assert suite_for_mode("subset", None) == "core"
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


def test_resolve_judge_model_defaults_to_evaluated_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PINCHBENCH_JUDGE_MODEL", raising=False)
    assert (
        resolve_judge_model(PinchBenchRunConfig(), "provider/model") == "provider/model"
    )
    monkeypatch.setenv("PINCHBENCH_JUDGE_MODEL", "provider/judge")
    assert (
        resolve_judge_model(PinchBenchRunConfig(), "provider/model") == "provider/judge"
    )
    assert (
        resolve_judge_model(
            PinchBenchRunConfig(judge_model="explicit"), "provider/model"
        )
        == "explicit"
    )


def test_list_task_specs_follows_manifest_order_for_full_mode() -> None:
    specs = list_task_specs(FIXTURE_ROOT, "full", None)
    assert [spec.task_id for spec in specs] == [
        "task_sanity",
        "task_calendar",
        "task_email",
    ]
    email = specs[-1]
    assert email.category == "writing"
    assert email.grading_type == "llm_judge"
    assert email.timeout_seconds == 180
    assert email.prompt.startswith("Write a professional email")


def test_list_task_specs_mode_presets_and_suite_syntax() -> None:
    assert [s.task_id for s in list_task_specs(FIXTURE_ROOT, "smoke", None)] == [
        "task_sanity"
    ]
    assert [s.task_id for s in list_task_specs(FIXTURE_ROOT, "subset", None)] == [
        "task_sanity",
        "task_email",
    ]
    assert [s.task_id for s in list_task_specs(FIXTURE_ROOT, "full", "writing")] == [
        "task_email"
    ]
    assert [
        s.task_id for s in list_task_specs(FIXTURE_ROOT, "full", "productivity+writing")
    ] == ["task_sanity", "task_calendar", "task_email"]
    assert [
        s.task_id
        for s in list_task_specs(FIXTURE_ROOT, "full", "task_email,task_sanity")
    ] == ["task_email", "task_sanity"]
    assert [
        s.task_id for s in list_task_specs(FIXTURE_ROOT, "full", "automated-only")
    ] == ["task_sanity", "task_calendar"]


def test_list_task_specs_rejects_unknown_task() -> None:
    with pytest.raises(BenchmarkInfrastructureError, match="unknown PinchBench task"):
        list_task_specs(FIXTURE_ROOT, "full", "task_missing")


def test_task_points_normalises_multi_point_tasks() -> None:
    task = {"grading": {"runs": [{"score": 2.0, "max_score": 4.0}], "mean": 0.5}}
    assert task_points(task) == (2.0, 4.0)
    assert task_score(task) == pytest.approx(0.5)
    assert task_points({"grading": {"runs": [], "mean": 0.25}}) == (0.25, 1.0)
    assert task_points({"status": "error"}) is None


def test_inspect_score_from_result() -> None:
    value, explanation, metadata = inspect_score_from_result(
        {
            "status": "success",
            "task_id": "task_email",
            "grading_type": "llm_judge",
            "score": 0.5,
            "score_points": 0.5,
            "max_points": 1.0,
            "task_status": "success",
            "notes": "Judge: partial credit",
            "run_dir": "/tmp/run",
        }
    )
    assert value == pytest.approx(0.5)
    assert "task_email" in explanation
    assert "Judge: partial credit" in explanation
    assert metadata["score_points"] == 0.5


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


def test_build_docker_command_routes_judge_through_proxy(tmp_path) -> None:
    """The judge must use the configured endpoint, not upstream's OpenRouter default.

    Upstream grades 122/147 tasks with an LLM judge and defaults to a Claude model
    on OpenRouter, so without this every judged task fails with
    "OPENROUTER_API_KEY not set" and scores 0.
    """
    command = build_docker_command(
        PinchBenchRunConfig(),
        tmp_path / "benchmark",
        tmp_path / "run",
        "task_email",
        "https://example.invalid/v1",
        "provider/model",
        None,
        judge_model="provider/judge",
        name="pinchbench-run-task_email",
    )

    inner = command[-1]
    assert "--judge openai/provider/judge" in inner
    assert 'export OPENAI_BASE_URL="http://127.0.0.1:${proxy_port}/v1"' in inner
    assert inner.index("export OPENAI_BASE_URL") < inner.index("scripts/benchmark.py")
    assert command[command.index("--name") + 1] == "pinchbench-run-task_email"


def test_container_name_is_docker_safe() -> None:
    name = container_name("full_20260903T000000Z_abc123", "task/with spaces")
    assert name == "pinchbench-full_20260903T000000Z_abc123-task-with-spaces"


def test_suite_accepts_list_from_inspect_task_args() -> None:
    """``-T suite=a,b`` reaches the task as a list because Inspect parses it as YAML."""
    assert (
        suite_for_mode("full", ["task_email", "task_sanity"])
        == "task_email,task_sanity"
    )
    assert [
        s.task_id
        for s in list_task_specs(FIXTURE_ROOT, "full", ["task_email", "task_sanity"])
    ] == ["task_email", "task_sanity"]


def test_build_docker_command_mounts_adapter_scripts_from_checkout(tmp_path) -> None:
    """Container-side scripts come from this checkout, not from the image.

    The compatibility patch is what routes the judge to the configured endpoint;
    baking it into the image meant a stale image silently ran the unpatched
    upstream code and every judged task failed with an OpenAI 401.
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
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "-v"]
    compat = [
        m
        for m in mounts
        if m.endswith("/usr/local/bin/pinchbench_openclaw_compat.py:ro")
    ]
    proxy = [
        m for m in mounts if m.endswith("/usr/local/bin/pinchbench_model_proxy.py:ro")
    ]
    assert compat and Path(compat[0].split(":")[0]).name == "openclaw_compat.py"
    assert proxy and Path(proxy[0].split(":")[0]).name == "_openai_compatible_proxy.py"


def test_record_grading_type_reads_the_runs() -> None:
    record = {
        "grading": {"mean": 0.0, "runs": [{"score": 0.0, "grading_type": "llm_judge"}]}
    }
    assert record_grading_type(record) == "llm_judge"
    assert record_grading_type({"grading": {"mean": 1.0, "runs": []}}) is None


def test_compat_patch_applies_to_pinned_upstream_snippets(tmp_path) -> None:
    """Every snippet the patch expects must exist once in the runner; here we
    exercise the patcher on a stand-in containing the pinned upstream lines."""
    from pinchbench.openclaw_compat import apply_pinchbench_openclaw_compat

    runner = tmp_path / "scripts" / "lib_agent.py"
    runner.parent.mkdir()
    runner.write_text(
        "\n".join(
            [
                "use_local = fws_env is not None",
                '                "--model",\n                model_id,',
                'bench_agent_dir = _get_agent_store_dir(agent_id) / "agent"',
                '            "apiKey": key_ref,',
                '"https://api.openai.com/v1/chat/completions",',
                '"max_completion_tokens": 2048,',
                'bench_models.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")',
            ]
        ),
        encoding="utf-8",
    )
    apply_pinchbench_openclaw_compat(tmp_path)
    patched = runner.read_text(encoding="utf-8")
    assert 'os.environ.get("OPENAI_BASE_URL"' in patched
    assert "PINCHBENCH_JUDGE_MAX_TOKENS" in patched
    assert "PINCHBENCH_PROVIDER_TIMEOUT_SECONDS" in patched
    assert "use_local = True" in patched


def test_container_api_key_never_falls_back_to_host_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent container runs model-produced commands with host networking, so
    only the key configured for the benchmarked endpoint may be forwarded."""
    monkeypatch.setenv("OPENAI_API_KEY", "host-key-for-another-provider")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "host-key-for-another-provider")
    assert container_api_key(None) == "EMPTY"
    assert container_api_key("endpoint-key") == "endpoint-key"
