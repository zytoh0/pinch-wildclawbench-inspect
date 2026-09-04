# PinchBench & WildClawBench Inspect AI Wrappers

This repository contains standalone **Inspect AI wrappers** for PinchBench and WildClawBench. It does not contain its own benchmark datasets or scoring logic. Instead, it runs pinned checkouts of the original benchmark implementations and reports their native scores through Inspect's scoring interface.

- Original PinchBench implementation: [pinchbench/skill](https://github.com/pinchbench/skill), tested at commit `819384ae830492365b8363fc26bc2602e73f216d`.
- Original WildClawBench implementation: [internlm/WildClawBench](https://github.com/internlm/WildClawBench), tested at commit `86d71447413d38f38740a021cb776f64eb396ee0`. WildClawBench corresponds to [arXiv:2605.10912v1](https://arxiv.org/abs/2605.10912v1).

Every native benchmark task is one Inspect sample, so the usual Inspect controls apply: `--limit N` runs the first N tasks, `--sample-id <task_id>` runs specific tasks, `--max-samples N` controls how many task containers run concurrently (the wrappers additionally cap this at `max_concurrency`, default 4), and per-task grades appear in the log and in `inspect view`.

## Evaluations Included

### PinchBench

PinchBench evaluates OpenClaw coding-agent performance on real-world tasks from the original `pinchbench/skill` repository. At the tested commit its task manifest (`tasks/manifest.yaml`, the upstream source of truth) lists **147 tasks in 11 categories**: productivity, research, writing, coding, analysis, CSV analysis, log analysis, meeting analysis, memory, skills, and integrations. (The upstream README badge still says 53; the manifest is authoritative.) This wrapper expects the user to provide a pinned checkout of the original repository via `PINCHBENCH_ROOT` or the `benchmark_root` task parameter.

The wrapper runs each task through the native PinchBench harness inside Docker and grades it the native way: 25 tasks are graded automatically, 21 by an LLM judge, and 101 by both (`hybrid`). The judge is served by the same OpenAI-compatible endpoint as the agent unless `judge_model` (or `PINCHBENCH_JUDGE_MODEL`) names another model that endpoint exposes. Each sample's score is the task's native points divided by its maximum points (0–1). Metrics: `mean` and `stderr` over tasks, `native_score` (total points earned over total points available, which is how the upstream harness reports its overall score), and a per-category mean.

Task selection: `mode=full` (default) runs every manifest task, `mode=subset` runs the manifest's `core` list (21 representative tasks, upstream's `--core`), and `mode=smoke` runs `task_sanity`. `suite` accepts the upstream runner's syntax (`all`, `core`, `automated-only`, a category or `cat1+cat2`, or comma-separated task ids) and overrides `mode`.

### WildClawBench

WildClawBench evaluates long-horizon, real-world agent performance on 60 human-authored bilingual and multimodal tasks from the original `internlm/WildClawBench` repository. The tasks span six categories: productivity flow, code intelligence, social interaction, search and retrieval, creative synthesis, and safety alignment. This wrapper expects the user to provide a pinned checkout of the original repository, with the task data downloaded into it, via `WILDCLAWBENCH_ROOT` or the `benchmark_root` task parameter.

The wrapper runs each task through the native WildClawBench batch runner, which executes the task's agent and its grading inside a Docker container. Native grading combines deterministic rule-based checks, environment-state auditing, and LLM/VLM judge scores; 42 of the 60 tasks call a judge, which is served by the same OpenAI-compatible endpoint as the agent unless `judge_model` (or `WILDCLAWBENCH_JUDGE_MODEL`) names another model that endpoint exposes. Each sample's score is the task's native `overall_score` (0–1). Metrics: `mean` and `stderr` over tasks and a per-category mean. Tasks whose native grading fails (rather than scoring the agent low) raise an error so the failure is visible instead of being counted as zero.

Task selection: `mode=full` (default) runs all 60 tasks, `mode=subset` runs one category (`category`, default `06_Safety_Alignment`), and `mode=smoke` runs the safety-alignment file-overwrite task, which is the only task that can run before the task data is downloaded. `task=<path to a task .md>` runs a single task.

## Pinned External Dependencies

External assets and runtime dependencies are pinned for reproducibility:

- PinchBench original repository: `https://github.com/pinchbench/skill.git` at `819384ae830492365b8363fc26bc2602e73f216d`.
- WildClawBench original repository: `https://github.com/internlm/WildClawBench.git` at `86d71447413d38f38740a021cb776f64eb396ee0`.
- WildClawBench task data: HuggingFace dataset `internlm/WildClawBench`, folder `workspace/`, revision `75f945578aa00cbdb8f46e4d42e4f4e98f704b4f`.
- Docker base image: `node:22-bookworm@sha256:c601a46abb4d2ab80a9dc3da208d50d1122642d53f17a101926ace71e5a9bf1c`.
- npm package: `openclaw@2026.6.10`.
- Python dependencies are pinned exactly in `pyproject.toml` and in both Dockerfiles. The WildClawBench image also carries the libraries the native grading snippets import inside the task container (`openai`, `pillow`, `pymupdf`, `numpy`).

The wrappers require Docker and an OpenAI-compatible model endpoint reachable from the host.

The model endpoint may be unauthenticated. `PINCHBENCH_API_KEY` and `WILDCLAWBENCH_API_KEY` (or the
`api_key` task parameter) are optional; when they are unset the wrappers supply a placeholder key so
that OpenClaw and the judge clients still resolve the provider, which is what locally served endpoints
such as vLLM, SGLang, and Ollama need.

Reasoning models: the native judges allow as few as 128 completion tokens, which a reasoning model can spend entirely on thinking. The PinchBench judge cap is raised to `judge_max_tokens` (default 8192). For servers that support it, `model_extra_body` is merged into every request the proxy forwards, e.g. `-T model_extra_body='{"chat_template_kwargs": {"enable_thinking": false}}'` for Qwen-family models on vLLM/SGLang.

## Running PinchBench

1. Clone the upstream PinchBench repository at the tested commit:

   ```bash
   git clone https://github.com/pinchbench/skill.git /path/to/pinchbench_skill
   git -C /path/to/pinchbench_skill checkout 819384ae830492365b8363fc26bc2602e73f216d
   ```

2. Install this wrapper repository:

   ```bash
   uv sync
   ```

3. Run the eval:

   ```bash
   export PINCHBENCH_ROOT=/path/to/pinchbench_skill
   export PINCHBENCH_MODEL_BASE_URL=<OPENAI_COMPATIBLE_BASE_URL>
   export PINCHBENCH_MODEL=<MODEL_ID>
   uv run inspect eval src/pinchbench/pinchbench.py@pinchbench --max-samples 3
   ```

   Useful variations:

   ```bash
   # Smoke test (one automated task)
   uv run inspect eval src/pinchbench/pinchbench.py@pinchbench -T mode=smoke
   # Upstream's 21-task core subset
   uv run inspect eval src/pinchbench/pinchbench.py@pinchbench -T mode=subset --max-samples 3
   # Specific tasks or categories
   uv run inspect eval src/pinchbench/pinchbench.py@pinchbench -T suite=task_email,task_calendar
   uv run inspect eval src/pinchbench/pinchbench.py@pinchbench -T suite=coding+writing
   # A different judge served by the same endpoint
   uv run inspect eval src/pinchbench/pinchbench.py@pinchbench -T judge_model=<JUDGE_MODEL_ID>
   ```

## Running WildClawBench

1. Clone the upstream WildClawBench repository at the tested commit and download the task data:

   ```bash
   git clone https://github.com/internlm/WildClawBench.git /path/to/WildClawBench
   git -C /path/to/WildClawBench checkout 86d71447413d38f38740a021cb776f64eb396ee0
   hf download internlm/WildClawBench --repo-type dataset \
     --revision 75f945578aa00cbdb8f46e4d42e4f4e98f704b4f \
     --include 'workspace/*' --local-dir /path/to/WildClawBench
   # Extracts the git archives the safety-alignment tasks need; the other steps
   # (YouTube videos, SAM3 weights) only affect a few creative-synthesis and
   # code-intelligence tasks and need yt-dlp, ffmpeg and gdown.
   bash /path/to/WildClawBench/script/prepare.sh
   ```

2. Install this wrapper repository:

   ```bash
   uv sync
   ```

3. Run the eval:

   ```bash
   export WILDCLAWBENCH_ROOT=/path/to/WildClawBench
   export WILDCLAWBENCH_MODEL_BASE_URL=<OPENAI_COMPATIBLE_BASE_URL>
   export WILDCLAWBENCH_MODEL=<MODEL_ID>
   uv run inspect eval src/wildclawbench/wildclawbench.py@wildclawbench --max-samples 2
   ```

   Useful variations:

   ```bash
   # Smoke test (runs without the task data)
   uv run inspect eval src/wildclawbench/wildclawbench.py@wildclawbench -T mode=smoke
   # One category
   uv run inspect eval src/wildclawbench/wildclawbench.py@wildclawbench -T mode=subset -T category=06_Safety_Alignment
   # One task
   uv run inspect eval src/wildclawbench/wildclawbench.py@wildclawbench -T task=tasks/01_Productivity_Flow/01_Productivity_Flow_task_3_bibtex.md
   ```

## Container networking, credentials and other deviations from upstream defaults

- **Host networking.** Every task container (the one that executes model-produced shell commands) runs with `--network host`: PinchBench starts it that way, and WildClawBench's upstream `docker run` is patched to add it. This is how the containers reach the per-run model proxy, which binds `127.0.0.1` (inside the container for PinchBench, on the host for WildClawBench). The consequence is that the agent under test shares the host's network namespace and can reach anything listening on the host's loopback interface. Filesystem and process isolation are unchanged, the benchmark checkout is mounted read-only, and no Docker socket is exposed. Run these evals on a host where that is acceptable — they include tasks (notably WildClawBench's `06_Safety_Alignment`, the default smoke task among them) that deliberately probe for destructive agent behaviour.
- **Credentials.** The only credential forwarded into a task container or the proxy is the one configured for the benchmarked endpoint (`PINCHBENCH_API_KEY` / `WILDCLAWBENCH_API_KEY` or `api_key`), or the placeholder `EMPTY` when there is none. A host `OPENAI_API_KEY` / `OPENAI_COMPATIBLE_API_KEY` is never used. The judge clients receive the same key.
- **Judge.** Upstream defaults to hosted judges (a Claude model on OpenRouter for PinchBench; `openai/gpt-5.4` via OpenRouter for WildClawBench). The wrappers serve the judge from the configured endpoint instead (see `judge_model`), raise PinchBench's judge completion cap to `judge_max_tokens`, and raise OpenClaw's provider timeout to 600 s (upstream default 120 s).
- **Upstream harness patches.** Both harnesses are patched in a per-run copy so they run against a local OpenAI-compatible provider (`use_local`, custom provider config, judge base URL, `openclaw agent --local` and host networking for WildClawBench). The patches fail loudly if the pinned upstream code changes.
- **Images.** The task containers use images built from this repository, not the images the upstream projects distribute; the WildClawBench image adds the grading libraries listed above.

## Timeouts, errors and artifacts

- `timeout_seconds` is a per-task safety net (default 1800 s for PinchBench, 2400 s for WildClawBench) on top of the native harnesses' own per-task timeouts. When it expires the task container is killed and the sample is recorded as an error.
- Infrastructure problems (endpoint unreachable, missing task data, native grading crashes) raise errors rather than scoring zero. Use `--no-fail-on-error` to keep a long run going past individual errors and `--retry-on-error` to retry them.
- Each task writes its artifacts under `<output_root>/<mode>/<run_id>/<task_id>/`: the exact command (`command.json`), the harness logs (`stdout.log`, `stderr.log`), the native result files (`native/`), and the adapter's parsed result (`inspect_adapter.json`). `output_root` defaults to `./pinchbench_artifacts` or `./wildclawbench_artifacts`, or to `$INSPECT_EVALS_ARTIFACTS_DIR` when set.

## Register entry

Both evals are listed in the [Inspect Evals register](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/register), which pins a commit of this repository. The `sync-inspect-evals-register` workflow opens a pull request against `inspect_evals` bumping that pin whenever `main` changes, so what is merged here becomes the registered version once that pull request is merged.
