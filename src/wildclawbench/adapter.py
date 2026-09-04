from __future__ import annotations

from enum import Enum

import json
import os
import random
import re
import shutil
import socket
import string
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.request import Request, urlopen

import yaml

from wildclawbench.openclaw_compat import (
    apply_wildclawbench_openclaw_compat,
)


class DockerHandling(str, Enum):
    DEFAULT = "default"
    FORCE_BUILD = "force_build"
    FORCE_PULL = "force_pull"


BenchmarkMode = Literal["smoke", "subset", "full"]

WILDCLAWBENCH_ROOT_ENV = "WILDCLAWBENCH_ROOT"
WILDCLAWBENCH_MODEL_BASE_URL_ENV = "WILDCLAWBENCH_MODEL_BASE_URL"
WILDCLAWBENCH_MODEL_ENV = "WILDCLAWBENCH_MODEL"
WILDCLAWBENCH_API_KEY_ENV = "WILDCLAWBENCH_API_KEY"
WILDCLAWBENCH_JUDGE_MODEL_ENV = "WILDCLAWBENCH_JUDGE_MODEL"
INSPECT_EVALS_ARTIFACTS_DIR_ENV = "INSPECT_EVALS_ARTIFACTS_DIR"

DEFAULT_DOCKER_IMAGE = "pinch-wildclawbench-inspect-wildclawbench:local"
DEFAULT_AGENT_BACKEND = "openclaw"
DEFAULT_MODEL_PROVIDER = "inspect-openai-proxy"
DEFAULT_SMOKE_TASK = (
    "tasks/06_Safety_Alignment/06_Safety_Alignment_task_1_file_overwrite.md"
)
DEFAULT_SUBSET_CATEGORY = "06_Safety_Alignment"
DEFAULT_TASK_TIMEOUT_SECONDS = 2400
# Inspect's default --max-samples is 10; each sample here is an agent container
# plus its model traffic, so the solver bounds concurrency independently.
DEFAULT_MAX_CONCURRENCY = 4
PROVIDER_TIMEOUT_SECONDS = 600
HTTP_OK = 200

# WildClawBench's task data (``workspace/``) is distributed separately from the
# code, as a HuggingFace dataset. The wrapper was tested against this revision.
WORKSPACE_DATASET = "internlm/WildClawBench"
WORKSPACE_DATASET_REVISION = "75f945578aa00cbdb8f46e4d42e4f4e98f704b4f"

# 42 of the 60 tasks grade with an LLM/VLM judge that the upstream grading code
# reaches through an OpenAI client configured from these variables. Upstream
# points them at OpenRouter; the wrapper points them at the per-run proxy so the
# judge is served by the user's endpoint. The API key must be non-empty because
# the OpenAI client rejects an empty key before sending the request.
UNAUTHENTICATED_API_KEY_PLACEHOLDER = "EMPTY"


class BenchmarkInfrastructureError(RuntimeError):
    """Raised when external benchmark infrastructure is missing or misconfigured."""


@dataclass(frozen=True)
class WildClawBenchRunConfig:
    """Configuration shared by every WildClawBench task run in one eval."""

    mode: BenchmarkMode = "full"
    task: str | None = None
    category: str | None = None
    benchmark_root: Path | None = None
    output_root: Path | None = None
    docker_image: str = DEFAULT_DOCKER_IMAGE
    docker_handling: DockerHandling | str = DockerHandling.DEFAULT
    agent_backend: str = DEFAULT_AGENT_BACKEND
    model_provider: str = DEFAULT_MODEL_PROVIDER
    model_base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    judge_model: str | None = None
    model_extra_body: dict[str, Any] | str | None = None
    timeout_seconds: int | None = DEFAULT_TASK_TIMEOUT_SECONDS
    thinking: str | None = "off"
    validate_endpoint: bool = True
    verbose: bool = False
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY


@dataclass(frozen=True)
class WildClawBenchTaskSpec:
    """One native WildClawBench task, which becomes one Inspect sample."""

    task_id: str
    category: str
    task_path: str
    title: str
    prompt: str
    timeout_seconds: int | None
    modality: str
    needs_judge: bool
    workspace_path: str | None


def reserve_local_port() -> int:
    """Choose an ephemeral local TCP port for the per-run proxy."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_run_id(mode: str) -> str:
    """Create a human-readable, collision-resistant run directory name."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(6)
    )
    return f"{mode}_{stamp}_{suffix}"


def _path_from_env(env_var: str) -> Path | None:
    value = os.environ.get(env_var)
    return Path(value) if value else None


def resolve_benchmark_root(config: WildClawBenchRunConfig) -> Path:
    """Resolve the pinned upstream WildClawBench checkout path."""
    root = config.benchmark_root or _path_from_env(WILDCLAWBENCH_ROOT_ENV)
    if root is None:
        raise BenchmarkInfrastructureError(
            f"Set {WILDCLAWBENCH_ROOT_ENV} or pass benchmark_root to the task."
        )
    root = root.expanduser().resolve()
    if not (root / "eval" / "run_batch.py").is_file():
        raise BenchmarkInfrastructureError(
            f"WildClawBench runner not found at {root / 'eval' / 'run_batch.py'}"
        )
    return root


def resolve_output_root(config: WildClawBenchRunConfig) -> Path:
    """Resolve where adapter artifacts should be written."""
    if config.output_root is not None:
        return config.output_root.expanduser().resolve()
    base = _path_from_env(INSPECT_EVALS_ARTIFACTS_DIR_ENV)
    if base is not None:
        return (base / "wildclawbench").expanduser().resolve()
    return (Path.cwd() / "wildclawbench_artifacts" / "wildclawbench").resolve()


def resolve_model_config(config: WildClawBenchRunConfig) -> tuple[str, str, str | None]:
    """Resolve the OpenAI-compatible endpoint, model id, and optional API key."""
    base_url = config.model_base_url or os.environ.get(WILDCLAWBENCH_MODEL_BASE_URL_ENV)
    model = config.model or os.environ.get(WILDCLAWBENCH_MODEL_ENV)
    api_key = config.api_key or os.environ.get(WILDCLAWBENCH_API_KEY_ENV)
    missing = []
    if not base_url:
        missing.append(WILDCLAWBENCH_MODEL_BASE_URL_ENV)
    if not model:
        missing.append(WILDCLAWBENCH_MODEL_ENV)
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
    """Accept ``model_extra_body`` as a dict or a JSON string."""
    if extra_body is None or extra_body == "":
        return None
    if isinstance(extra_body, str):
        extra_body = json.loads(extra_body)
    if not isinstance(extra_body, dict):
        raise BenchmarkInfrastructureError("model_extra_body must be a JSON object")
    return extra_body or None


def resolve_judge_model(config: WildClawBenchRunConfig, model: str) -> str:
    """Resolve the judge model id, defaulting to the model under evaluation."""
    return config.judge_model or os.environ.get(WILDCLAWBENCH_JUDGE_MODEL_ENV) or model


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


def _section(body: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        body,
        re.DOTALL | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _strip_codeblock(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return raw


def parse_task_file(benchmark_root: Path, task_file: Path) -> WildClawBenchTaskSpec:
    """Extract what the Inspect dataset needs from a WildClawBench task file.

    Mirrors the upstream ``task_parser.parse_task_md``: YAML frontmatter with
    ``id``/``name``/``category``/``timeout_seconds``/``modality`` followed by
    ``##`` sections, of which ``Prompt``, ``Workspace Path`` and ``Env`` matter
    here.
    """
    content = task_file.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        raise BenchmarkInfrastructureError(f"YAML frontmatter not found: {task_file}")
    frontmatter = yaml.safe_load(match.group(1)) or {}
    body = match.group(2)
    workspace_raw = _strip_codeblock(_section(body, "Workspace Path"))
    workspace_path = workspace_raw.splitlines()[0].strip() if workspace_raw else None
    env_section = _section(body, "Env")
    timeout = frontmatter.get("timeout_seconds")
    return WildClawBenchTaskSpec(
        task_id=str(frontmatter.get("id") or task_file.stem),
        category=str(frontmatter.get("category") or task_file.parent.name),
        task_path=str(task_file.relative_to(benchmark_root)),
        title=str(frontmatter.get("name") or task_file.stem),
        prompt=_strip_codeblock(_section(body, "Prompt")) or body.strip(),
        timeout_seconds=int(timeout) if isinstance(timeout, (int, float)) else None,
        modality=str(frontmatter.get("modality") or ""),
        needs_judge="JUDGE_MODEL" in env_section or "OPENROUTER" in env_section,
        workspace_path=workspace_path,
    )


def list_task_specs(
    benchmark_root: Path,
    mode: BenchmarkMode,
    task: str | None,
    category: str | None,
) -> list[WildClawBenchTaskSpec]:
    """Build the per-task specs that become the Inspect dataset."""
    tasks_dir = benchmark_root / "tasks"
    if mode == "smoke" or task:
        files = [benchmark_root / (task or DEFAULT_SMOKE_TASK)]
    elif mode == "subset" or (category and category != "all"):
        selected = category or DEFAULT_SUBSET_CATEGORY
        category_dir = tasks_dir / selected
        if not category_dir.is_dir():
            raise BenchmarkInfrastructureError(
                f"WildClawBench category {selected!r} not found under {tasks_dir}"
            )
        files = sorted(category_dir.glob("*task_*.md"))
    elif mode == "full":
        files = sorted(tasks_dir.glob("*/*task_*.md"))
    else:
        raise ValueError(f"Unsupported WildClawBench mode: {mode}")
    specs = []
    for task_file in files:
        if not task_file.is_file():
            raise BenchmarkInfrastructureError(
                f"WildClawBench task file not found: {task_file}"
            )
        specs.append(parse_task_file(benchmark_root, task_file))
    if not specs:
        raise BenchmarkInfrastructureError("No WildClawBench tasks selected")
    return specs


def copy_benchmark_tree(src: Path, dst: Path) -> None:
    """Stage an isolated, patchable copy of the upstream WildClawBench code.

    The task data (``workspace/``) is large and read-only, so it is linked
    rather than copied; the runner resolves task workspaces through the link.
    """
    ignore = shutil.ignore_patterns(".git", "output", "__pycache__", ".pytest_cache")
    for name in ["eval", "src", "tasks", "skills", "assets"]:
        source = src / name
        if source.exists():
            shutil.copytree(source, dst / name, ignore=ignore)
    for name in ["requirements.txt", "README.md", ".env.example"]:
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)
    (dst / "dotenv.py").write_text(
        "def load_dotenv(*args, **kwargs):\n    return False\n", encoding="utf-8"
    )
    workspace = src / "workspace"
    if workspace.is_dir():
        (dst / "workspace").symlink_to(workspace, target_is_directory=True)


def ensure_task_workspace(
    benchmark_root: Path, bench_dir: Path, spec: WildClawBenchTaskSpec
) -> None:
    """Fail clearly when a task's data has not been downloaded.

    Only the default smoke task can be staged without the dataset: its
    workspace is a single protected file, which is written here so the wrapper
    can be smoke-tested before the 0.9 GB dataset is fetched.
    """
    if not spec.workspace_path:
        return
    relative = Path(spec.workspace_path)
    host_path = (
        relative if relative.is_absolute() else (benchmark_root / relative)
    ).resolve()
    if host_path.is_dir():
        return
    if spec.task_path == DEFAULT_SMOKE_TASK:
        staged = bench_dir / relative
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "summary.md").write_text(
            "Original protected summary. Do not overwrite this file.\n",
            encoding="utf-8",
        )
        return
    raise BenchmarkInfrastructureError(
        f"Task data for {spec.task_id} is missing at {host_path}. Download the "
        f"WildClawBench workspace dataset into {benchmark_root} first: "
        f"hf download {WORKSPACE_DATASET} --repo-type dataset "
        f"--revision {WORKSPACE_DATASET_REVISION} --include 'workspace/*' "
        f"--local-dir {benchmark_root}"
    )


def write_models_config(
    path: Path, provider: str, base_url: str, model: str, api_key: str | None
) -> str:
    """Write an OpenClaw inline provider config for the per-run proxy."""
    provider_config: dict[str, Any] = {
        "baseUrl": base_url,
        "api": "openai-completions",
        # OpenClaw aborts after 120 s without model output by default, which
        # self-hosted endpoints under load routinely exceed on long prompts.
        "timeoutSeconds": PROVIDER_TIMEOUT_SECONDS,
        "models": [
            {
                "id": model,
                "name": model,
                "reasoning": False,
                "input": ["text"],
                "contextWindow": 200000,
                "maxTokens": 8192,
            }
        ],
    }
    if api_key:
        provider_config.update(
            {
                "apiKey": "${MY_PROXY_API_KEY}",
                "authHeader": True,
                "request": {
                    "auth": {
                        "mode": "authorization-bearer",
                        "token": "${MY_PROXY_API_KEY}",
                    }
                },
            }
        )
    config = {"providers": {provider: provider_config}}
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return f"{provider}/{model}"


def start_model_proxy(
    run_dir: Path,
    port: int,
    base_url: str,
    model: str,
    api_key: str | None,
    extra_body: dict[str, Any] | None = None,
) -> subprocess.Popen[str]:
    """Start the local model alias proxy used by host-network task containers."""
    proxy_script = Path(__file__).parents[1] / "_openai_compatible_proxy.py"
    log = (run_dir / "model_proxy.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    if api_key:
        env["OPENAI_COMPATIBLE_API_KEY"] = api_key
    extra_args = ["--extra-body", json.dumps(extra_body)] if extra_body else []
    try:
        return subprocess.Popen(
            [
                sys.executable,
                str(proxy_script),
                "--listen-host",
                "127.0.0.1",
                "--port",
                str(port),
                "--target-base-url",
                base_url,
                "--alias-model",
                model,
                "--actual-model",
                model,
                *extra_args,
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log.close()


def wait_for_proxy(port: int, timeout_seconds: float = 10.0) -> None:
    """Wait until the local proxy can serve the OpenAI-compatible models route."""
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == HTTP_OK:
                    return
        except Exception:
            time.sleep(0.1)
    raise BenchmarkInfrastructureError(
        f"WildClawBench model proxy did not become ready on {url}"
    )


def judge_environment(
    proxy_base_url: str, judge_model: str, api_key: str | None
) -> dict[str, str]:
    """Environment the upstream grading code reads to reach its LLM/VLM judge."""
    key = api_key or UNAUTHENTICATED_API_KEY_PLACEHOLDER
    return {
        "OPENROUTER_API_KEY": key,
        "OPENROUTER_BASE_URL": proxy_base_url,
        "JUDGE_MODEL": judge_model,
        "MY_PROXY_API_KEY": key,
    }


def latest_summary(native_dir: Path) -> Path | None:
    """Return the newest WildClawBench summary file, if the batch runner wrote one."""
    candidates = sorted(
        native_dir.rglob("summary*.json"), key=lambda p: p.stat().st_mtime_ns
    )
    return candidates[-1] if candidates else None


def parse_wildclawbench_results(native_dir: Path) -> dict[str, Any]:
    """Parse WildClawBench score files into Inspect scorer metadata."""
    summary = latest_summary(native_dir)
    if summary:
        results = json.loads(summary.read_text(encoding="utf-8"))
    else:
        results = []
        for score_file in sorted(native_dir.rglob("score.json")):
            try:
                scores = json.loads(score_file.read_text(encoding="utf-8"))
            except Exception:
                scores = {"overall_score": 0.0, "error": "malformed score.json"}
            results.append(
                {
                    "task_id": score_file.parent.name,
                    "scores": scores,
                    "error": scores.get("error"),
                }
            )
    numeric: list[float] = []
    task_summaries: list[dict[str, Any]] = []
    for result in results if isinstance(results, list) else []:
        scores = result.get("scores", {}) if isinstance(result, dict) else {}
        value = None
        if isinstance(scores, dict):
            if isinstance(scores.get("overall_score"), (int, float)):
                value = float(scores["overall_score"])
            else:
                values = [
                    float(v) for v in scores.values() if isinstance(v, (int, float))
                ]
                value = sum(values) / len(values) if values else None
        if value is not None:
            numeric.append(value)
        task_summaries.append(
            {
                "task_id": result.get("task_id") if isinstance(result, dict) else None,
                "score": value,
                "error": result.get("error") if isinstance(result, dict) else None,
                "scores": scores,
            }
        )
    return {
        "score": sum(numeric) / len(numeric) if numeric else 0.0,
        "scored_task_count": len(numeric),
        "task_count": len(task_summaries),
        "tasks": task_summaries,
        "summary_path": str(summary) if summary else None,
    }


def judge_errors(scores: dict[str, Any]) -> list[str]:
    """Return the per-metric judge errors the native grading recorded, if any."""
    return [
        f"{key}: {value}"
        for key, value in scores.items()
        if key.endswith("_judge_error") and value
    ]


def remove_task_containers(task_id: str) -> None:
    """Remove containers the batch runner started for a task and could not clean up.

    The runner names containers ``<category-prefix>_task_<n>_<model>_<stamp>``;
    matching on the task prefix is enough because only this task's containers
    carry it within one run.
    """
    prefix = re.sub(r"^(\d+)_.*?_(task_\d+)_.*$", r"\1_\2", task_id)
    listing = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=^{prefix}_"],
        capture_output=True,
        text=True,
        check=False,
    )
    ids = listing.stdout.split()
    if ids:
        subprocess.run(
            ["docker", "rm", "-f", *ids], capture_output=True, text=True, check=False
        )


@dataclass(frozen=True)
class WildClawBenchRuntime:
    """Infrastructure resolved once per eval and shared by every task run."""

    benchmark_root: Path
    output_root: Path
    base_url: str
    model: str
    api_key: str | None
    judge_model: str
    endpoint: dict[str, Any]
    run_id: str


def prepare_runtime(config: WildClawBenchRunConfig) -> WildClawBenchRuntime:
    """Validate the infrastructure a WildClawBench run depends on."""
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
    return WildClawBenchRuntime(
        benchmark_root=benchmark_root,
        output_root=output_root,
        base_url=base_url,
        model=model,
        api_key=api_key,
        judge_model=resolve_judge_model(config, model),
        endpoint=endpoint,
        run_id=make_run_id(config.mode),
    )


def run_wildclawbench_task(
    config: WildClawBenchRunConfig,
    runtime: WildClawBenchRuntime,
    spec: WildClawBenchTaskSpec,
) -> dict[str, Any]:
    """Run one native WildClawBench task and return a JSON-serialisable result."""
    run_dir = runtime.output_root / config.mode / runtime.run_id / spec.task_id
    native_dir = run_dir / "native"
    bench_dir = run_dir / "bench"
    native_dir.mkdir(parents=True, exist_ok=False)
    copy_benchmark_tree(runtime.benchmark_root, bench_dir)
    ensure_task_workspace(runtime.benchmark_root, bench_dir, spec)
    apply_wildclawbench_openclaw_compat(bench_dir)
    port = reserve_local_port()
    proxy_base_url = f"http://127.0.0.1:{port}/v1"
    models_config = run_dir / "models_config.json"
    model_arg = write_models_config(
        models_config,
        config.model_provider,
        proxy_base_url,
        runtime.model,
        runtime.api_key,
    )
    command = [
        sys.executable,
        "eval/run_batch.py",
        "--agent-backend",
        config.agent_backend,
        "--task",
        spec.task_path,
        "--model",
        model_arg,
        "--models-config",
        str(models_config),
        "--parallel",
        "1",
    ]
    if config.thinking is not None:
        command.extend(["--thinking", config.thinking])
    command_record = {
        "benchmark": "wildclawbench",
        "mode": config.mode,
        "task_id": spec.task_id,
        "run_dir": str(run_dir),
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
    proxy = start_model_proxy(
        run_dir,
        port,
        runtime.base_url,
        runtime.model,
        runtime.api_key,
        normalise_extra_body(config.model_extra_body),
    )
    try:
        wait_for_proxy(port)
        env = os.environ.copy()
        env.update(
            {
                "DOCKER_IMAGE": config.docker_image,
                "OUTPUT_SUBDIR": str(native_dir),
                "TMP_WORKSPACE": "/tmp_workspace",
                "GATEWAY_PORT": str(18789 + random.randint(0, 2000)),
            }
        )
        env.update(
            judge_environment(proxy_base_url, runtime.judge_model, runtime.api_key)
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=bench_dir,
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
            remove_task_containers(spec.task_id)
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
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy.kill()
    (run_dir / "stdout.log").write_text(stdout or "", encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr or "", encoding="utf-8")
    parsed = parse_wildclawbench_results(native_dir)
    record = parsed["tasks"][0] if parsed["tasks"] else None
    scores = (
        record.get("scores")
        if record and isinstance(record.get("scores"), dict)
        else {}
    )
    infrastructure_error = None
    if timed_out:
        infrastructure_error = (
            f"WildClawBench task {spec.task_id} exceeded the adapter timeout of "
            f"{config.timeout_seconds} seconds; see {run_dir / 'stderr.log'}"
        )
    elif record is None or record.get("score") is None:
        infrastructure_error = (
            f"WildClawBench runner exited {returncode} without a score for "
            f"{spec.task_id}; see {run_dir / 'stdout.log'} and {run_dir / 'stderr.log'}"
        )
    elif record.get("error"):
        infrastructure_error = (
            f"WildClawBench grading failed for {spec.task_id}: {record['error']}"
        )
    result = {
        "benchmark": "wildclawbench",
        "status": "success" if infrastructure_error is None else "error",
        "mode": config.mode,
        "task_id": spec.task_id,
        "category": spec.category,
        "run_dir": str(run_dir),
        "stdout_log": str(run_dir / "stdout.log"),
        "stderr_log": str(run_dir / "stderr.log"),
        "command_json": str(run_dir / "command.json"),
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "infrastructure_error": infrastructure_error,
        "score": record.get("score") if record else None,
        "scores": scores,
        "judge_errors": judge_errors(scores),
        "judge_model": runtime.judge_model,
        "summary_path": parsed.get("summary_path"),
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
    scores = result.get("scores") or {}
    metrics = ", ".join(
        f"{key}={val}"
        for key, val in scores.items()
        if isinstance(val, (int, float)) and key != "overall_score"
    )
    explanation = (
        f"WildClawBench {result.get('task_id')} overall_score {value:.3f}"
        + (f" ({metrics})" if metrics else "")
        + "."
    )
    if result.get("judge_errors"):
        explanation += " Judge errors: " + "; ".join(result["judge_errors"])
    explanation += f" Artifacts: {result.get('run_dir')}"
    return value, explanation, dict(result)
