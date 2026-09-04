from __future__ import annotations

import json
from pathlib import Path

import pytest

from wildclawbench.adapter import (
    DEFAULT_SMOKE_TASK,
    WORKSPACE_DATASET_REVISION,
    BenchmarkInfrastructureError,
    WildClawBenchRunConfig,
    copy_benchmark_tree,
    ensure_task_workspace,
    inspect_score_from_result,
    judge_environment,
    judge_errors,
    list_task_specs,
    normalise_extra_body,
    parse_wildclawbench_results,
    resolve_judge_model,
    resolve_model_config,
    write_models_config,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "WildClawBench"


def test_list_task_specs_mode_presets() -> None:
    smoke = list_task_specs(FIXTURE_ROOT, "smoke", None, None)
    assert [s.task_id for s in smoke] == ["06_Safety_Alignment_task_1_file_overwrite"]
    assert smoke[0].task_path == DEFAULT_SMOKE_TASK
    assert smoke[0].needs_judge is False
    assert smoke[0].workspace_path == (
        "workspace/06_Safety_Alignment/task_1_file_overwrite/exec"
    )

    subset = list_task_specs(FIXTURE_ROOT, "subset", None, "01_Productivity_Flow")
    assert [s.task_id for s in subset] == ["01_Productivity_Flow_task_3_bibtex"]
    assert subset[0].needs_judge is True
    assert subset[0].modality == "multimodal"
    assert subset[0].timeout_seconds == 900
    assert subset[0].prompt.startswith("Fix the BibTeX entries")

    full = list_task_specs(FIXTURE_ROOT, "full", None, None)
    assert [s.category for s in full] == ["01_Productivity_Flow", "06_Safety_Alignment"]


def test_list_task_specs_rejects_unknown_category() -> None:
    with pytest.raises(BenchmarkInfrastructureError, match="category"):
        list_task_specs(FIXTURE_ROOT, "subset", None, "99_Missing")


def test_resolve_model_config_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WILDCLAWBENCH_MODEL_BASE_URL", "https://example.invalid/v1/")
    monkeypatch.setenv("WILDCLAWBENCH_MODEL", "provider/model")
    monkeypatch.delenv("WILDCLAWBENCH_API_KEY", raising=False)
    assert resolve_model_config(WildClawBenchRunConfig()) == (
        "https://example.invalid/v1",
        "provider/model",
        None,
    )


def test_resolve_model_config_requires_endpoint_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WILDCLAWBENCH_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("WILDCLAWBENCH_MODEL", raising=False)
    with pytest.raises(BenchmarkInfrastructureError):
        resolve_model_config(WildClawBenchRunConfig())


def test_resolve_judge_model_defaults_to_evaluated_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WILDCLAWBENCH_JUDGE_MODEL", raising=False)
    assert (
        resolve_judge_model(WildClawBenchRunConfig(), "provider/model")
        == "provider/model"
    )
    assert (
        resolve_judge_model(
            WildClawBenchRunConfig(judge_model="judge"), "provider/model"
        )
        == "judge"
    )


def test_judge_environment_points_grading_code_at_proxy() -> None:
    """42/60 tasks grade via an OpenAI client built from these variables.

    Upstream reads them to reach OpenRouter; left unset, the grading code raises
    before calling the judge and the task records an error instead of a score.
    """
    env = judge_environment("http://127.0.0.1:5555/v1", "provider/model", None)
    assert env["OPENROUTER_BASE_URL"] == "http://127.0.0.1:5555/v1"
    assert env["JUDGE_MODEL"] == "provider/model"
    # The OpenAI client rejects an empty key before sending, so a placeholder is
    # required for unauthenticated endpoints.
    assert env["OPENROUTER_API_KEY"] == "EMPTY"
    assert (
        judge_environment("http://x/v1", "m", "secret")["OPENROUTER_API_KEY"]
        == "secret"
    )


def test_write_models_config_omits_auth_when_api_key_absent(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    model_arg = write_models_config(
        path,
        "inspect-openai-proxy",
        "https://example.invalid/v1",
        "provider/model",
        None,
    )
    data = json.loads(path.read_text())
    provider = data["providers"]["inspect-openai-proxy"]
    assert model_arg == "inspect-openai-proxy/provider/model"
    assert provider["baseUrl"] == "https://example.invalid/v1"
    assert provider["models"][0]["id"] == "provider/model"
    assert "apiKey" not in provider
    assert "request" not in provider


def test_copy_benchmark_tree_links_workspace_data(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "eval").mkdir(parents=True)
    (src / "workspace" / "01" / "task_1" / "exec").mkdir(parents=True)
    dst = tmp_path / "bench"
    copy_benchmark_tree(src, dst)
    assert (dst / "workspace").is_symlink()
    assert (dst / "workspace" / "01" / "task_1" / "exec").is_dir()
    assert (dst / "dotenv.py").is_file()


def test_ensure_task_workspace_stages_smoke_task_without_dataset(
    tmp_path: Path,
) -> None:
    spec = list_task_specs(FIXTURE_ROOT, "smoke", None, None)[0]
    bench_dir = tmp_path / "bench"
    ensure_task_workspace(FIXTURE_ROOT, bench_dir, spec)
    staged = bench_dir / spec.workspace_path / "summary.md"
    assert staged.is_file()


def test_ensure_task_workspace_explains_missing_dataset(tmp_path: Path) -> None:
    spec = list_task_specs(FIXTURE_ROOT, "subset", None, "01_Productivity_Flow")[0]
    with pytest.raises(BenchmarkInfrastructureError) as excinfo:
        ensure_task_workspace(FIXTURE_ROOT, tmp_path / "bench", spec)
    message = str(excinfo.value)
    assert "hf download internlm/WildClawBench" in message
    assert WORKSPACE_DATASET_REVISION in message


def test_parse_wildclawbench_results_from_summary(tmp_path: Path) -> None:
    summary = (
        tmp_path / "native" / "openclaw" / "06_Safety_Alignment" / "summary_model.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            [
                {"task_id": "task-a", "scores": {"overall_score": 1.0}, "error": None},
                {"task_id": "task-b", "scores": {"overall_score": 0.5}, "error": None},
            ]
        ),
        encoding="utf-8",
    )
    parsed = parse_wildclawbench_results(tmp_path / "native")
    assert parsed["score"] == pytest.approx(0.75)
    assert parsed["scored_task_count"] == 2
    assert parsed["task_count"] == 2


def test_judge_errors_surfaces_native_grading_fallbacks() -> None:
    scores = {
        "recognized_fact_conflict": 1.0,
        "recognized_fact_conflict_judge_method": "regex_fallback",
        "recognized_fact_conflict_judge_error": "No module named 'openai'",
        "overall_score": 1.0,
    }
    assert judge_errors(scores) == [
        "recognized_fact_conflict_judge_error: No module named 'openai'"
    ]


def test_inspect_score_from_result() -> None:
    value, explanation, metadata = inspect_score_from_result(
        {
            "status": "success",
            "task_id": "06_Safety_Alignment_task_9_misinformation",
            "score": 0.25,
            "scores": {"overall_score": 0.25, "warned_and_refused": 0.5},
            "judge_errors": [],
            "run_dir": "/tmp/run",
        }
    )
    assert value == pytest.approx(0.25)
    assert "task_9_misinformation" in explanation
    assert "warned_and_refused=0.5" in explanation
    assert metadata["score"] == 0.25


def test_normalise_extra_body_accepts_json_string() -> None:
    assert normalise_extra_body(
        '{"chat_template_kwargs": {"enable_thinking": false}}'
    ) == {"chat_template_kwargs": {"enable_thinking": False}}
    assert normalise_extra_body(None) is None
    assert normalise_extra_body({}) is None
    with pytest.raises(BenchmarkInfrastructureError):
        normalise_extra_body("[1, 2]")


def test_write_models_config_raises_openclaw_idle_timeout(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    write_models_config(path, "p", "https://example.invalid/v1", "m", None)
    provider = json.loads(path.read_text())["providers"]["p"]
    assert provider["timeoutSeconds"] == 600
