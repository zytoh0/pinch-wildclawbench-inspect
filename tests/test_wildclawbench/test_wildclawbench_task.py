from __future__ import annotations

from pathlib import Path

import pytest

from wildclawbench.wildclawbench import wildclawbench

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "WildClawBench"


def test_wildclawbench_builds_one_sample_per_native_task() -> None:
    task = wildclawbench(
        mode="full", benchmark_root=str(FIXTURE_ROOT), validate_endpoint=False
    )
    assert [sample.id for sample in task.dataset] == [
        "01_Productivity_Flow_task_3_bibtex",
        "06_Safety_Alignment_task_1_file_overwrite",
    ]
    bibtex = task.dataset[0]
    assert bibtex.metadata["category"] == "01_Productivity_Flow"
    assert bibtex.metadata["needs_judge"] is True
    assert bibtex.metadata["modality"] == "multimodal"
    assert task.version == 2


def test_wildclawbench_smoke_mode_selects_file_overwrite_task() -> None:
    task = wildclawbench(
        mode="smoke", benchmark_root=str(FIXTURE_ROOT), validate_endpoint=False
    )
    assert [sample.id for sample in task.dataset] == [
        "06_Safety_Alignment_task_1_file_overwrite"
    ]


def test_wildclawbench_requires_benchmark_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WILDCLAWBENCH_ROOT", raising=False)
    with pytest.raises(ValueError, match="WILDCLAWBENCH_ROOT"):
        wildclawbench(validate_endpoint=False)
