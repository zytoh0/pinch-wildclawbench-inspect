from __future__ import annotations

from enum import Enum

import json
import os
import random
import re
import shlex
import shutil
import string
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.request import Request, urlopen

import yaml


class DockerHandling(str, Enum):
    DEFAULT = "default"
    FORCE_BUILD = "force_build"
    FORCE_PULL = "force_pull"


BenchmarkMode = Literal["smoke", "subset", "full"]

PINCHBENCH_ROOT_ENV = "PINCHBENCH_ROOT"
PINCHBENCH_MODEL_BASE_URL_ENV = "PINCHBENCH_MODEL_BASE_URL"
PINCHBENCH_MODEL_ENV = "PINCHBENCH_MODEL"
PINCHBENCH_API_KEY_ENV = "PINCHBENCH_API_KEY"
PINCHBENCH_JUDGE_MODEL_ENV = "PINCHBENCH_JUDGE_MODEL"
INSPECT_EVALS_ARTIFACTS_DIR_ENV = "INSPECT_EVALS_ARTIFACTS_DIR"

DEFAULT_DOCKER_IMAGE = "pinch-wildclawbench-inspect-pinchbench:local"
DEFAULT_SMOKE_SUITE = "task_sanity"
# The upstream manifest's ``core`` list: the maintainers' own representative
# subset (at least one task per category, mixed grading types), run upstream
# with ``benchmark.py --core``.
DEFAULT_SUBSET_SUITE = "core"
DEFAULT_TASK_TIMEOUT_SECONDS = 1800
# Inspect's default --max-samples is 10; each sample here is an agent container
# plus its model traffic, so the solver bounds concurrency independently.
DEFAULT_MAX_CONCURRENCY = 4
# Upstream caps judge completions at 2048 tokens, which reasoning models spend on
# thinking before they answer; the compatibility patch makes the cap configurable.
DEFAULT_JUDGE_MAX_TOKENS = 8192

# The upstream runner writes the literal "${OPENAI_API_KEY}" into OpenClaw's
# provider config when no API key is configured, and OpenClaw refuses to call a
# provider whose key cannot be resolved. Unauthenticated OpenAI-compatible
# endpoints (vLLM, SGLang, Ollama) therefore need a non-empty placeholder in the
# container environment, otherwise the agent issues no model requests at all and
# the eval silently reports a zero score.
UNAUTHENTICATED_API_KEY_PLACEHOLDER = "EMPTY"


class BenchmarkInfrastructureError(RuntimeError):
    """Raised when external benchmark infrastructure is missing or misconfigured."""


@dataclass(frozen=True)
class PinchBenchRunConfig:
    """Configuration shared by every PinchBench task run in one eval."""

    mode: BenchmarkMode = "full"
    suite: str | list[str] | None = None
    benchmark_root: Path | None = None
    output_root: Path | None = None
    docker_image: str = DEFAULT_DOCKER_IMAGE
    docker_handling: DockerHandling | str = DockerHandling.DEFAULT
    model_base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    judge_model: str | None = None
    model_extra_body: dict[str, Any] | str | None = None
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    timeout_seconds: int | None = DEFAULT_TASK_TIMEOUT_SECONDS
    timeout_multiplier: float = 1.0
    runs: int = 1
    thinking: str = "off"
    no_upload: bool = True
    validate_endpoint: bool = True
    verbose: bool = False
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY


@dataclass(frozen=True)
class PinchBenchTaskSpec:
    """One native PinchBench task, which becomes one Inspect sample."""

    task_id: str
    name: str
    category: str
    grading_type: str
    timeout_seconds: int | None
    prompt: str


def model_alias(model: str) -> str:
    """Return a slash-free model alias accepted by OpenClaw's custom provider."""
    alias = re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-.").lower()
    return alias or "model"


def make_run_id(mode: str) -> str:
    """Create a human-readable, collision-resistant run directory name."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(6)
    )
    return f"{mode}_{stamp}_{suffix}"


def normalise_suite(suite: str | list[str] | tuple[str, ...] | None) -> str | None:
    """Accept the suite as a string or a list.

    ``inspect eval -T suite=task_a,task_b`` hands the task a list because Inspect
    parses comma-separated ``-T`` values as YAML sequences.
    """
    if suite is None:
        return None
    if isinstance(suite, (list, tuple)):
        return ",".join(str(item).strip() for item in suite if str(item).strip())
    return str(suite).strip() or None


def suite_for_mode(mode: BenchmarkMode, suite: str | list[str] | None) -> str:
    """Resolve the PinchBench suite selected by the Inspect task mode."""
    suite = normalise_suite(suite)
    if suite:
        return suite
    if mode == "smoke":
        return DEFAULT_SMOKE_SUITE
    if mode == "subset":
        return DEFAULT_SUBSET_SUITE
    if mode == "full":
        return "all"
    raise ValueError(f"Unsupported PinchBench mode: {mode}")


def _path_from_env(env_var: str) -> Path | None:
    value = os.environ.get(env_var)
    return Path(value) if value else None


def resolve_benchmark_root(config: PinchBenchRunConfig) -> Path:
    """Resolve the pinned upstream PinchBench checkout path."""
    root = config.benchmark_root or _path_from_env(PINCHBENCH_ROOT_ENV)
    if root is None:
        raise BenchmarkInfrastructureError(
            f"Set {PINCHBENCH_ROOT_ENV} or pass benchmark_root to the task."
        )
    root = root.expanduser().resolve()
    if not (root / "scripts" / "benchmark.py").is_file():
        raise BenchmarkInfrastructureError(
            f"PinchBench runner not found at {root / 'scripts' / 'benchmark.py'}"
        )
    return root


def resolve_output_root(config: PinchBenchRunConfig) -> Path:
    """Resolve where adapter artifacts should be written."""
    if config.output_root is not None:
        return config.output_root.expanduser().resolve()
    base = _path_from_env(INSPECT_EVALS_ARTIFACTS_DIR_ENV)
    if base is not None:
        return (base / "pinchbench").expanduser().resolve()
    return (Path.cwd() / "pinchbench_artifacts" / "pinchbench").resolve()


def resolve_model_config(config: PinchBenchRunConfig) -> tuple[str, str, str | None]:
    """Resolve the OpenAI-compatible endpoint, model id, and optional API key."""
    base_url = config.model_base_url or os.environ.get(PINCHBENCH_MODEL_BASE_URL_ENV)
    model = config.model or os.environ.get(PINCHBENCH_MODEL_ENV)
    api_key = config.api_key or os.environ.get(PINCHBENCH_API_KEY_ENV)
    missing = []
    if not base_url:
        missing.append(PINCHBENCH_MODEL_BASE_URL_ENV)
    if not model:
        missing.append(PINCHBENCH_MODEL_ENV)
    if missing:
        raise BenchmarkInfrastructureError(
            "Missing model configuration. Set "
            + ", ".join(missing)
            + " or pass task parameters."
        )
    return base_url.rstrip("/"), model, api_key or None


def normalise_extra_body(
    extra_body: dict[str, Any] | str | None,
) -> dict[str, Any] | None:
    """Accept ``model_extra_body`` as a dict or a JSON string.

    ``inspect eval -T model_extra_body='{"a": 1}'`` may hand the task either,
    depending on how Inspect parses the value.
    """
    if extra_body is None or extra_body == "":
        return None
    if isinstance(extra_body, str):
        extra_body = json.loads(extra_body)
    if not isinstance(extra_body, dict):
        raise BenchmarkInfrastructureError("model_extra_body must be a JSON object")
    return extra_body or None


def resolve_judge_model(config: PinchBenchRunConfig, model: str) -> str:
    """Resolve the judge model id, defaulting to the model under evaluation.

    PinchBench grades 122 of its 147 tasks with an LLM judge. Upstream defaults
    to a Claude model on OpenRouter, which is unreachable from the sandboxed
    run; the judge is instead served by the same OpenAI-compatible endpoint as
    the agent (through the per-run proxy), so every task can be graded.
    """
    return config.judge_model or os.environ.get(PINCHBENCH_JUDGE_MODEL_ENV) or model


def validate_model_endpoint(
    base_url: str, model: str, api_key: str | None, timeout: int = 10
) -> dict[str, Any]:
    """Validate that the configured OpenAI-compatible endpoint exposes the model."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    models_url = base_url.rstrip("/") + "/models"
    try:
        request = Request(models_url, headers=headers)
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise BenchmarkInfrastructureError(
            f"Unable to reach model endpoint {models_url}: {exc}"
        ) from exc
    models = [
        item.get("id") for item in payload.get("data", []) if isinstance(item, dict)
    ]
    if model not in models:
        raise BenchmarkInfrastructureError(
            f"Model {model!r} was not present in {models_url}; exposed models: {models}"
        )
    return {"models_url": models_url, "models": models}


def check_docker_available() -> str:
    """Return the Docker server version or raise an infrastructure error."""
    docker = shutil.which("docker")
    if not docker:
        raise BenchmarkInfrastructureError("Docker CLI was not found on PATH")
    result = subprocess.run(
        [docker, "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise BenchmarkInfrastructureError(
            f"Docker is not available: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def prepare_docker_image(image: str, docker_handling: DockerHandling | str) -> None:
    """Prepare the adapter Docker image using inspect-evals DockerHandling values."""
    handling = DockerHandling(docker_handling)
    dockerfile = Path(__file__).with_name("Dockerfile")
    context = Path(__file__).parents[2]
    if handling is DockerHandling.FORCE_PULL:
        subprocess.run(["docker", "pull", image], check=True)
        return
    image_exists = (
        subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )
    if handling is DockerHandling.FORCE_BUILD or not image_exists:
        subprocess.run(
            ["docker", "build", "-t", image, "-f", str(dockerfile), str(context)],
            check=True,
        )


def read_task_markdown(path: Path) -> tuple[dict[str, Any], str]:
    """Split a PinchBench task file into its YAML frontmatter and body."""
    content = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        raise BenchmarkInfrastructureError(f"No YAML frontmatter found in {path}")
    frontmatter = yaml.safe_load(match.group(1)) or {}
    if not isinstance(frontmatter, dict):
        raise BenchmarkInfrastructureError(f"Malformed frontmatter in {path}")
    return frontmatter, match.group(2)


def task_prompt(body: str) -> str:
    """Return the text of the ``## Prompt`` section of a task file."""
    match = re.search(
        r"^## Prompt\s*\n(.*?)(?=^## |\Z)", body, re.DOTALL | re.MULTILINE
    )
    return (match.group(1) if match else body).strip()


def load_manifest(benchmark_root: Path) -> dict[str, Any]:
    """Load ``tasks/manifest.yaml``, the upstream single source of truth for tasks."""
    manifest_path = benchmark_root / "tasks" / "manifest.yaml"
    if not manifest_path.is_file():
        raise BenchmarkInfrastructureError(
            f"PinchBench task manifest not found at {manifest_path}"
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    categories = manifest.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise BenchmarkInfrastructureError(
            f"PinchBench manifest {manifest_path} declares no task categories"
        )
    return manifest


def manifest_task_ids(manifest: dict[str, Any]) -> list[str]:
    """Return every manifest task id in the upstream's canonical order."""
    return [
        task_id
        for tasks in manifest["categories"].values()
        for task_id in (tasks or [])
    ]


def select_task_ids(manifest: dict[str, Any], suite: str) -> list[str]:
    """Resolve a suite string the way the upstream runner does.

    ``all`` selects every manifest task, ``core`` the manifest's quick subset,
    ``automated-only`` the tasks without an LLM judge, a ``+``-joined list of
    category names selects those categories, and anything else is treated as a
    comma-separated list of task ids.
    """
    categories: dict[str, list[str]] = {
        name: list(tasks or []) for name, tasks in manifest["categories"].items()
    }
    if suite == "all":
        return manifest_task_ids(manifest)
    if suite == "core":
        core = manifest.get("core") or []
        if not core:
            raise BenchmarkInfrastructureError(
                "PinchBench manifest defines no core tasks; pass an explicit suite"
            )
        return list(core)
    if suite == "automated-only":
        # Grading types live in the task files; ``list_task_specs`` filters.
        return manifest_task_ids(manifest)
    requested = [part.strip() for part in suite.split("+") if part.strip()]
    if requested and all(part in categories for part in requested):
        return [task_id for part in requested for task_id in categories[part]]
    return [task_id.strip() for task_id in suite.split(",") if task_id.strip()]


def list_task_specs(
    benchmark_root: Path, mode: BenchmarkMode, suite: str | list[str] | None
) -> list[PinchBenchTaskSpec]:
    """Build the per-task specs that become the Inspect dataset."""
    manifest = load_manifest(benchmark_root)
    category_of = {
        task_id: category
        for category, tasks in manifest["categories"].items()
        for task_id in (tasks or [])
    }
    resolved_suite = suite_for_mode(mode, suite)
    task_ids = select_task_ids(manifest, resolved_suite)
    automated_only = resolved_suite == "automated-only"
    specs: list[PinchBenchTaskSpec] = []
    for task_id in task_ids:
        path = benchmark_root / "tasks" / f"{task_id}.md"
        if not path.is_file():
            raise BenchmarkInfrastructureError(
                f"Suite {resolved_suite!r} names unknown PinchBench task {task_id!r}"
            )
        frontmatter, body = read_task_markdown(path)
        grading_type = str(frontmatter.get("grading_type", "unknown"))
        if automated_only and grading_type != "automated":
            continue
        timeout = frontmatter.get("timeout_seconds")
        specs.append(
            PinchBenchTaskSpec(
                task_id=task_id,
                name=str(frontmatter.get("name", task_id)),
                category=category_of.get(task_id, str(frontmatter.get("category", ""))),
                grading_type=grading_type,
                timeout_seconds=int(timeout)
                if isinstance(timeout, (int, float))
                else None,
                prompt=task_prompt(body),
            )
        )
    if not specs:
        raise BenchmarkInfrastructureError(
            f"Suite {resolved_suite!r} selected no PinchBench tasks"
        )
    return specs


def container_name(run_id: str, task_id: str) -> str:
    """Return a Docker container name unique to one task run."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"pinchbench-{run_id}-{task_id}")
    return safe[:128]


def host_owner() -> str:
    """Return ``uid:gid`` of the invoking user for chown inside the container."""
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    return f"{getuid()}:{getgid()}" if getuid and getgid else "0:0"


def build_docker_command(
    config: PinchBenchRunConfig,
    benchmark_root: Path,
    run_dir: Path,
    suite: str,
    base_url: str,
    model: str,
    api_key: str | None,
    judge_model: str | None = None,
    name: str | None = None,
) -> list[str]:
    """Build the Docker command that executes the native PinchBench runner."""
    alias = model_alias(model)
    proxy_port_arg = "$proxy_port"
    proxy_url_arg = '"http://127.0.0.1:${proxy_port}/v1"'
    script_args = [
        "python3",
        "scripts/benchmark.py",
        "--model",
        alias,
        "--base-url",
        proxy_url_arg,
        "--suite",
        suite,
        "--runs",
        str(config.runs),
        "--timeout-multiplier",
        str(config.timeout_multiplier),
        "--thinking",
        config.thinking,
        "--output-dir",
        "/outputs/native",
        "--no-parallel-judge",
        "--no-judge-cache",
    ]
    if judge_model:
        # ``openai/<model>`` selects the upstream runner's direct OpenAI-compatible
        # judge backend; the compatibility patch points it at OPENAI_BASE_URL,
        # which the container sets to the per-run proxy below.
        script_args.extend(["--judge", f"openai/{judge_model}"])
    if config.no_upload:
        script_args.append("--no-upload")
    if config.verbose:
        script_args.append("--verbose")

    proxy_args = [
        "python3",
        "/usr/local/bin/pinchbench_model_proxy.py",
        "--listen-host",
        "127.0.0.1",
        "--port",
        proxy_port_arg,
        "--target-base-url",
        base_url,
        "--alias-model",
        alias,
        "--actual-model",
        model,
    ]
    extra_body = normalise_extra_body(config.model_extra_body)
    if extra_body:
        proxy_args.extend(["--extra-body", json.dumps(extra_body)])
    raw_shell_args = {proxy_port_arg, proxy_url_arg}
    script_cmd = " ".join(
        arg if arg in raw_shell_args else shlex.quote(arg) for arg in script_args
    )
    proxy_cmd = " ".join(
        arg if arg in raw_shell_args else shlex.quote(arg) for arg in proxy_args
    )
    inner = (
        "cp -a /benchmark/pinchbench_skill /tmp/pinchbench_skill "
        "&& chmod -R u+w /tmp/pinchbench_skill "
        "&& cd /tmp/pinchbench_skill "
        "&& python3 /usr/local/bin/pinchbench_openclaw_compat.py "
        "|| exit $?; "
        "proxy_port=$(python3 - <<'PYPORT'\n"
        "import socket\n"
        "with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:\n"
        "    sock.bind(('127.0.0.1', 0))\n"
        "    print(sock.getsockname()[1])\n"
        "PYPORT\n"
        "); "
        f"({proxy_cmd} >/outputs/model_proxy.log 2>&1) & "
        "proxy_pid=$!; "
        # Files the harness writes to /outputs are owned by the container's root
        # user; hand them back to the invoking user so the artifacts can be
        # inspected and removed without sudo.
        f"trap 'kill $proxy_pid 2>/dev/null || true; chown -R {host_owner()} /outputs 2>/dev/null || true' EXIT; "
        "for _ in $(seq 1 100); do curl -fsS http://127.0.0.1:${proxy_port}/v1/models >/dev/null && break; sleep 0.1; done; "
        'export OPENAI_BASE_URL="http://127.0.0.1:${proxy_port}/v1"; '
        f"export PINCHBENCH_JUDGE_MAX_TOKENS={int(config.judge_max_tokens)}; "
        f"{script_cmd}"
    )
    env_args = [
        "-e",
        "HOME=/tmp/pinchbench-home",
        "-e",
        f"PINCHBENCH_MODEL_ALIAS={alias}",
    ]
    # Forwarded by name so the value is taken from the adapter environment
    # instead of appearing in the container's argument list.
    env_args.extend(["-e", "OPENAI_API_KEY", "-e", "OPENAI_COMPATIBLE_API_KEY"])
    name_args = ["--name", name] if name else []
    # The compatibility patch and proxy are mounted from this checkout so the
    # container always runs the same version as the adapter, even when an
    # older image is reused.
    compat_script = Path(__file__).with_name("openclaw_compat.py").resolve()
    proxy_script = (Path(__file__).parents[1] / "_openai_compatible_proxy.py").resolve()
    return [
        "docker",
        "run",
        "--rm",
        *name_args,
        "--network",
        "host",
        *env_args,
        "-v",
        f"{benchmark_root}:/benchmark/pinchbench_skill:ro",
        "-v",
        f"{run_dir}:/outputs",
        "-v",
        f"{compat_script}:/usr/local/bin/pinchbench_openclaw_compat.py:ro",
        "-v",
        f"{proxy_script}:/usr/local/bin/pinchbench_model_proxy.py:ro",
        config.docker_image,
        "bash",
        "-lc",
        inner,
    ]


def latest_native_result(native_dir: Path) -> Path | None:
    """Return the newest native PinchBench JSON result, if one exists."""
    candidates = [p for p in native_dir.glob("*.json") if p.is_file()]
    return max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else None


def category_aggregate_score(data: dict[str, Any]) -> float | None:
    """Aggregate the native per-category totals the way PinchBench reports them.

    The native summary has no top-level score field; it reports
    ``category_scores`` as ``{CATEGORY: {"score": x, "max_score": y}}`` and
    prints ``sum(score) / sum(max_score)`` as its overall score.
    """
    categories = data.get("category_scores")
    if not isinstance(categories, dict):
        return None
    earned = 0.0
    available = 0.0
    for category in categories.values():
        if not isinstance(category, dict):
            continue
        score = category.get("score")
        max_score = category.get("max_score")
        if isinstance(score, (int, float)) and isinstance(max_score, (int, float)):
            earned += float(score)
            available += float(max_score)
    if available <= 0:
        return None
    return earned / available


def task_points(task: dict[str, Any]) -> tuple[float, float] | None:
    """Return a task's native ``(score, max_score)`` points summed over its runs."""
    grading = task.get("grading")
    if isinstance(grading, dict):
        runs = grading.get("runs")
        if isinstance(runs, list):
            earned = 0.0
            available = 0.0
            counted = False
            for run in runs:
                if not isinstance(run, dict):
                    continue
                score = run.get("score")
                if not isinstance(score, (int, float)):
                    continue
                max_score = run.get("max_score")
                earned += float(score)
                available += (
                    float(max_score)
                    if isinstance(max_score, (int, float)) and max_score > 0
                    else 1.0
                )
                counted = True
            if counted:
                return earned, available
        mean = grading.get("mean")
        if isinstance(mean, (int, float)):
            return float(mean), 1.0
    score = task.get("score")
    return (float(score), 1.0) if isinstance(score, (int, float)) else None


def task_score(task: dict[str, Any]) -> float | None:
    """Return a single task's native score normalised to the 0-1 range."""
    points = task_points(task)
    if points is None:
        return None
    earned, available = points
    return earned / available if available > 0 else earned


def parse_pinchbench_results(result_path: Path) -> dict[str, Any]:
    """Parse the native PinchBench JSON result into Inspect scorer metadata."""
    data = json.loads(result_path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    score = data.get("score")
    if score is None:
        score = category_aggregate_score(data)
    if score is None:
        # The native records keep each task's score under "grading", so fall
        # back to the mean of the per-task scores rather than assuming a
        # top-level "score" key exists on each task.
        numeric_scores = [
            value
            for value in (task_score(task) for task in tasks if isinstance(task, dict))
            if value is not None
        ]
        score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
    return {
        "native_result": str(result_path),
        "score": float(score),
        "tasks": tasks,
        "summary": data,
    }


def find_task_record(tasks: list[Any], task_id: str) -> dict[str, Any] | None:
    """Return the native record for ``task_id`` from a result's task list."""
    for task in tasks:
        if isinstance(task, dict) and task.get("task_id") == task_id:
            return task
    return None


def record_grading_type(task: dict[str, Any]) -> str | None:
    """Return how the native harness graded a task (``automated``/``llm_judge``/``hybrid``)."""
    grading = task.get("grading")
    if not isinstance(grading, dict):
        return None
    if grading.get("grading_type"):
        return str(grading["grading_type"])
    for run in grading.get("runs", []) or []:
        if isinstance(run, dict) and run.get("grading_type"):
            return str(run["grading_type"])
    return None


def grading_notes(task: dict[str, Any]) -> str:
    """Collect the native grader notes for one task."""
    grading = task.get("grading")
    if not isinstance(grading, dict):
        return ""
    notes = [
        str(run.get("notes"))
        for run in grading.get("runs", []) or []
        if isinstance(run, dict) and run.get("notes")
    ]
    return " | ".join(notes)


def kill_container(name: str) -> None:
    """Stop a container left behind after the adapter timed out waiting for it."""
    subprocess.run(
        ["docker", "kill", name], capture_output=True, text=True, check=False
    )


@dataclass(frozen=True)
class PinchBenchRuntime:
    """Infrastructure resolved once per eval and shared by every task run."""

    benchmark_root: Path
    output_root: Path
    base_url: str
    model: str
    api_key: str | None
    judge_model: str
    endpoint: dict[str, Any]
    run_id: str


def prepare_runtime(config: PinchBenchRunConfig) -> PinchBenchRuntime:
    """Validate the infrastructure a PinchBench run depends on."""
    benchmark_root = resolve_benchmark_root(config)
    output_root = resolve_output_root(config)
    base_url, model, api_key = resolve_model_config(config)
    check_docker_available()
    prepare_docker_image(config.docker_image, config.docker_handling)
    endpoint = (
        validate_model_endpoint(base_url, model, api_key)
        if config.validate_endpoint
        else {"models_url": None, "models": []}
    )
    return PinchBenchRuntime(
        benchmark_root=benchmark_root,
        output_root=output_root,
        base_url=base_url,
        model=model,
        api_key=api_key,
        judge_model=resolve_judge_model(config, model),
        endpoint=endpoint,
        run_id=make_run_id(config.mode),
    )


def run_pinchbench_task(
    config: PinchBenchRunConfig, runtime: PinchBenchRuntime, task_id: str
) -> dict[str, Any]:
    """Run one native PinchBench task and return a JSON-serialisable result."""
    run_dir = runtime.output_root / config.mode / runtime.run_id / task_id
    native_dir = run_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=False)
    name = container_name(runtime.run_id, task_id)
    command = build_docker_command(
        config,
        runtime.benchmark_root,
        run_dir,
        task_id,
        runtime.base_url,
        runtime.model,
        runtime.api_key,
        judge_model=runtime.judge_model,
        name=name,
    )
    command_record = {
        "benchmark": "pinchbench",
        "mode": config.mode,
        "task_id": task_id,
        "run_dir": str(run_dir),
        "container": name,
        "command": command,
        "docker_image": config.docker_image,
        "api_key_provided": bool(runtime.api_key),
        "judge_model": runtime.judge_model,
        "endpoint": runtime.endpoint,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "command.json").write_text(
        json.dumps(command_record, indent=2), encoding="utf-8"
    )
    env = os.environ.copy()
    container_api_key = (
        runtime.api_key
        or env.get("OPENAI_API_KEY")
        or env.get("OPENAI_COMPATIBLE_API_KEY")
        or UNAUTHENTICATED_API_KEY_PLACEHOLDER
    )
    env["OPENAI_API_KEY"] = container_api_key
    env["OPENAI_COMPATIBLE_API_KEY"] = container_api_key
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds or None,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        # Killing the docker client does not stop the container; kill it by
        # name so an expired task cannot keep hitting the model endpoint.
        kill_container(name)
        returncode = 124
        stdout = (
            exc.stdout
            if isinstance(exc.stdout, str)
            else (exc.stdout or b"").decode("utf-8", "replace")
        )
        stderr = (
            exc.stderr
            if isinstance(exc.stderr, str)
            else (exc.stderr or b"").decode("utf-8", "replace")
        )
        stderr += f"\nAdapter timeout after {config.timeout_seconds} seconds"
        timed_out = True
    duration = time.monotonic() - started
    (run_dir / "stdout.log").write_text(stdout or "", encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr or "", encoding="utf-8")
    native_result = latest_native_result(native_dir)
    record: dict[str, Any] | None = None
    summary: dict[str, Any] = {}
    if native_result is not None:
        parsed = parse_pinchbench_results(native_result)
        summary = parsed["summary"]
        record = find_task_record(parsed["tasks"], task_id)
    infrastructure_error = None
    if timed_out:
        infrastructure_error = (
            f"PinchBench task {task_id} exceeded the adapter timeout of "
            f"{config.timeout_seconds} seconds; see {run_dir / 'stderr.log'}"
        )
    elif record is None:
        infrastructure_error = (
            f"PinchBench runner exited {returncode} without a native result for "
            f"{task_id}; see {run_dir / 'stdout.log'} and {run_dir / 'stderr.log'}"
        )
    points = task_points(record) if record else None
    result = {
        "benchmark": "pinchbench",
        "status": "success" if infrastructure_error is None else "error",
        "mode": config.mode,
        "task_id": task_id,
        "run_dir": str(run_dir),
        "stdout_log": str(run_dir / "stdout.log"),
        "stderr_log": str(run_dir / "stderr.log"),
        "command_json": str(run_dir / "command.json"),
        "native_result": str(native_result) if native_result else None,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "infrastructure_error": infrastructure_error,
        "score": task_score(record) if record else None,
        "score_points": points[0] if points else None,
        "max_points": points[1] if points else None,
        "task_status": record.get("status") if record else None,
        "grading_type": record_grading_type(record) if record else None,
        "notes": grading_notes(record) if record else "",
        "usage": record.get("usage") if record else None,
        "task": record,
        "benchmark_version": summary.get("benchmark_version"),
        "judge_model": runtime.judge_model,
    }
    (run_dir / "inspect_adapter.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if infrastructure_error:
        raise BenchmarkInfrastructureError(infrastructure_error)
    return result


def inspect_score_from_result(
    result: dict[str, Any],
) -> tuple[float, str, dict[str, Any]]:
    """Convert adapter JSON into Inspect's Score fields."""
    value = float(result.get("score", 0.0) or 0.0)
    points = result.get("score_points")
    max_points = result.get("max_points")
    explanation = (
        f"PinchBench {result.get('task_id')} ({result.get('grading_type')}) "
        f"scored {points}/{max_points} = {value:.3f}; native status "
        f"{result.get('task_status')}."
    )
    if result.get("notes"):
        explanation += f" Notes: {result['notes']}"
    explanation += f" Artifacts: {result.get('run_dir')}"
    return value, explanation, dict(result)
