from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Target,
    grouped,
    mean,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from pinchbench.adapter import (
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_TASK_TIMEOUT_SECONDS,
    BenchmarkInfrastructureError,
    DockerHandling,
    PinchBenchRunConfig,
    PinchBenchRuntime,
    inspect_score_from_result,
    list_task_specs,
    normalise_suite,
    prepare_runtime,
    resolve_benchmark_root,
    run_pinchbench_task,
)


@solver
def pinchbench_runner(config: PinchBenchRunConfig) -> Solver:
    """Run one native PinchBench task per sample through the Dockerized harness."""

    runtime: PinchBenchRuntime | None = None
    runtime_lock = asyncio.Lock()
    slots = asyncio.Semaphore(max(1, config.max_concurrency))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        nonlocal runtime
        async with runtime_lock:
            if runtime is None:
                runtime = await asyncio.to_thread(prepare_runtime, config)
        async with slots:
            result = await asyncio.to_thread(
                run_pinchbench_task, config, runtime, str(state.sample_id)
            )
        state.output = ModelOutput.from_content(
            model=runtime.model,
            content=json.dumps(result, indent=2),
        )
        return state

    return solve


@metric
def native_score() -> Metric:
    """PinchBench's own aggregate: total points earned over total points available.

    Upstream reports ``sum(score) / sum(max_score)`` across tasks, which weights
    a task by its maximum score; ``mean`` weights every task equally.
    """

    def compute(scores: list[SampleScore]) -> float:
        earned = 0.0
        available = 0.0
        for sample_score in scores:
            metadata = sample_score.score.metadata or {}
            points = metadata.get("score_points")
            max_points = metadata.get("max_points")
            if isinstance(points, (int, float)) and isinstance(
                max_points, (int, float)
            ):
                earned += float(points)
                available += float(max_points)
        return earned / available if available > 0 else 0.0

    return compute


@scorer(
    metrics=[
        mean(),
        stderr(),
        native_score(),
        grouped(mean(), "category", all=False),
    ]
)
def pinchbench_scorer():
    """Score each sample with the native PinchBench grade for its task."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
            result = json.loads(state.output.completion)
        except Exception as exc:
            return Score(
                value=0.0,
                explanation=f"PinchBench adapter output was not valid JSON: {exc}",
            )
        value, explanation, metadata = inspect_score_from_result(result)
        return Score(value=value, explanation=explanation, metadata=metadata)

    return score


@task
def pinchbench(
    mode: Literal["smoke", "subset", "full"] = "full",
    suite: str | list[str] | None = None,
    benchmark_root: str | None = None,
    output_root: str | None = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    docker_handling: DockerHandling | str = DockerHandling.DEFAULT,
    model_base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    judge_model: str | None = None,
    model_extra_body: dict | str | None = None,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    timeout_seconds: int | None = DEFAULT_TASK_TIMEOUT_SECONDS,
    timeout_multiplier: float = 1.0,
    runs: int = 1,
    thinking: str = "off",
    no_upload: bool = True,
    validate_endpoint: bool = True,
    verbose: bool = False,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> Task:
    """Create an Inspect task that runs PinchBench through its native harness.

    Every native PinchBench task is one Inspect sample, so ``--limit``,
    ``--sample-id`` and ``--max-samples`` behave as they do for any other eval and
    per-task grades are visible in the log. ``mode`` picks the default task
    selection (``smoke``: the sanity task, ``subset``: the upstream ``core``
    list, ``full``: every manifest task) and ``suite`` accepts the upstream
    runner's suite syntax to override it.

    The native harness requires a pinned PinchBench checkout and an
    OpenAI-compatible model endpoint, supplied via task parameters or the
    environment variables documented in the README. The LLM judge is served by
    the same endpoint unless ``judge_model`` names another model it exposes.
    ``model_extra_body`` is a JSON object merged into every chat completion the
    proxy forwards, for server-specific fields such as vLLM/SGLang's
    ``{"chat_template_kwargs": {"enable_thinking": false}}``. At most
    ``max_concurrency`` task containers run at once regardless of
    ``--max-samples``.
    """
    config = PinchBenchRunConfig(
        mode=mode,
        suite=suite,
        benchmark_root=Path(benchmark_root) if benchmark_root else None,
        output_root=Path(output_root) if output_root else None,
        docker_image=docker_image,
        docker_handling=docker_handling,
        model_base_url=model_base_url,
        model=model,
        api_key=api_key,
        judge_model=judge_model,
        model_extra_body=model_extra_body,
        judge_max_tokens=judge_max_tokens,
        timeout_seconds=timeout_seconds,
        timeout_multiplier=timeout_multiplier,
        runs=runs,
        thinking=thinking,
        no_upload=no_upload,
        validate_endpoint=validate_endpoint,
        verbose=verbose,
        max_concurrency=max_concurrency,
    )
    try:
        specs = list_task_specs(resolve_benchmark_root(config), mode, suite)
    except BenchmarkInfrastructureError as exc:
        raise ValueError(f"Cannot build the PinchBench dataset: {exc}") from exc
    return Task(
        dataset=[
            Sample(
                input=spec.prompt,
                target="native_grade",
                id=spec.task_id,
                metadata={
                    "task_name": spec.name,
                    "category": spec.category,
                    "grading_type": spec.grading_type,
                    "native_timeout_seconds": spec.timeout_seconds,
                    "mode": mode,
                    "suite": normalise_suite(suite),
                },
            )
            for spec in specs
        ],
        solver=pinchbench_runner(config),
        scorer=pinchbench_scorer(),
        version=2,
    )
