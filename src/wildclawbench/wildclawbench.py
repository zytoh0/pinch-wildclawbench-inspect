from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Target, grouped, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver

from wildclawbench.adapter import (
    DEFAULT_AGENT_BACKEND,
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_TASK_TIMEOUT_SECONDS,
    BenchmarkInfrastructureError,
    DockerHandling,
    WildClawBenchRunConfig,
    WildClawBenchRuntime,
    WildClawBenchTaskSpec,
    inspect_score_from_result,
    list_task_specs,
    prepare_runtime,
    resolve_benchmark_root,
    run_wildclawbench_task,
)


@solver
def wildclawbench_runner(
    config: WildClawBenchRunConfig, specs: dict[str, WildClawBenchTaskSpec]
) -> Solver:
    """Run one native WildClawBench task per sample through the batch runner."""

    runtime: WildClawBenchRuntime | None = None
    runtime_lock = asyncio.Lock()
    slots = asyncio.Semaphore(max(1, config.max_concurrency))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        nonlocal runtime
        async with runtime_lock:
            if runtime is None:
                runtime = await asyncio.to_thread(prepare_runtime, config)
        async with slots:
            result = await asyncio.to_thread(
                run_wildclawbench_task, config, runtime, specs[str(state.sample_id)]
            )
        state.output = ModelOutput.from_content(
            model=runtime.model,
            content=json.dumps(result, indent=2),
        )
        return state

    return solve


@scorer(metrics=[mean(), stderr(), grouped(mean(), "category", all=False)])
def wildclawbench_scorer():
    """Score each sample with the native WildClawBench overall_score for its task."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
            result = json.loads(state.output.completion)
        except Exception as exc:
            return Score(
                value=0.0,
                explanation=f"WildClawBench adapter output was not valid JSON: {exc}",
            )
        value, explanation, metadata = inspect_score_from_result(result)
        return Score(value=value, explanation=explanation, metadata=metadata)

    return score


@task
def wildclawbench(
    mode: Literal["smoke", "subset", "full"] = "full",
    task: str | None = None,
    category: str | None = None,
    benchmark_root: str | None = None,
    output_root: str | None = None,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    docker_handling: DockerHandling | str = DockerHandling.DEFAULT,
    agent_backend: str = DEFAULT_AGENT_BACKEND,
    model_provider: str = DEFAULT_MODEL_PROVIDER,
    model_base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    judge_model: str | None = None,
    model_extra_body: dict | str | None = None,
    timeout_seconds: int | None = DEFAULT_TASK_TIMEOUT_SECONDS,
    thinking: str | None = "off",
    validate_endpoint: bool = True,
    verbose: bool = False,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> Task:
    """Create an Inspect task that runs WildClawBench through its native harness.

    Every native WildClawBench task is one Inspect sample, so ``--limit``,
    ``--sample-id`` and ``--max-samples`` behave as they do for any other eval and
    per-task grades are visible in the log. ``mode`` picks the default task
    selection (``smoke``: one safety task, ``subset``: one category, ``full``:
    all 60 tasks); ``task`` (a task file path) or ``category`` override it.

    The native harness requires a pinned WildClawBench checkout with the task
    data downloaded into it, and an OpenAI-compatible model endpoint. The LLM/VLM
    judge is served by the same endpoint unless ``judge_model`` names another
    model it exposes. ``model_extra_body`` is a JSON object merged into every
    chat completion the proxy forwards, for server-specific fields such as
    vLLM/SGLang's ``{"chat_template_kwargs": {"enable_thinking": false}}``; the
    native judge snippets allow as few as 128 completion tokens, which a
    reasoning model otherwise spends thinking. At most ``max_concurrency``
    task containers run at once regardless of ``--max-samples``.
    """
    config = WildClawBenchRunConfig(
        mode=mode,
        task=task,
        category=category,
        benchmark_root=Path(benchmark_root) if benchmark_root else None,
        output_root=Path(output_root) if output_root else None,
        docker_image=docker_image,
        docker_handling=docker_handling,
        agent_backend=agent_backend,
        model_provider=model_provider,
        model_base_url=model_base_url,
        model=model,
        api_key=api_key,
        judge_model=judge_model,
        model_extra_body=model_extra_body,
        timeout_seconds=timeout_seconds,
        thinking=thinking,
        validate_endpoint=validate_endpoint,
        verbose=verbose,
        max_concurrency=max_concurrency,
    )
    try:
        specs = list_task_specs(resolve_benchmark_root(config), mode, task, category)
    except BenchmarkInfrastructureError as exc:
        raise ValueError(f"Cannot build the WildClawBench dataset: {exc}") from exc
    return Task(
        dataset=[
            Sample(
                input=spec.prompt,
                target="native_grade",
                id=spec.task_id,
                metadata={
                    "task_name": spec.title,
                    "category": spec.category,
                    "modality": spec.modality,
                    "needs_judge": spec.needs_judge,
                    "native_timeout_seconds": spec.timeout_seconds,
                    "task_path": spec.task_path,
                    "mode": mode,
                },
            )
            for spec in specs
        ],
        solver=wildclawbench_runner(config, {spec.task_id: spec for spec in specs}),
        scorer=wildclawbench_scorer(),
        version=2,
    )
