from __future__ import annotations

from pathlib import Path

import pytest

from pinchbench.pinchbench import pinchbench

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "pinchbench_skill"


def test_pinchbench_builds_one_sample_per_native_task() -> None:
    task = pinchbench(
        mode="full", benchmark_root=str(FIXTURE_ROOT), validate_endpoint=False
    )
    assert [sample.id for sample in task.dataset] == [
        "task_sanity",
        "task_calendar",
        "task_email",
    ]
    email = task.dataset[2]
    assert email.metadata["category"] == "writing"
    assert email.metadata["grading_type"] == "llm_judge"
    assert email.metadata["native_timeout_seconds"] == 180
    assert task.version == 2


def test_pinchbench_smoke_mode_selects_sanity_task() -> None:
    task = pinchbench(
        mode="smoke", benchmark_root=str(FIXTURE_ROOT), validate_endpoint=False
    )
    assert [sample.id for sample in task.dataset] == ["task_sanity"]


def test_pinchbench_requires_benchmark_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINCHBENCH_ROOT", raising=False)
    with pytest.raises(ValueError, match="PINCHBENCH_ROOT"):
        pinchbench(validate_endpoint=False)
