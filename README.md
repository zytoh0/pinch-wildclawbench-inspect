# PinchBench & WildClawBench Inspect AI Wrappers

This repository contains standalone **Inspect AI wrappers** for PinchBench and WildClawBench. It does not contain its own benchmark datasets or scoring logic. Instead, it runs pinned checkouts of the original benchmark implementations and reports their native scores through Inspect's scoring interface.

- Original PinchBench implementation: [pinchbench/skill](https://github.com/pinchbench/skill), tested at commit `819384ae830492365b8363fc26bc2602e73f216d`.
- Original WildClawBench implementation: [internlm/WildClawBench](https://github.com/internlm/WildClawBench), tested at commit `86d71447413d38f38740a021cb776f64eb396ee0`. WildClawBench corresponds to [arXiv:2605.10912v1](https://arxiv.org/abs/2605.10912v1).

## Evaluations Included

### PinchBench

PinchBench evaluates OpenClaw coding-agent performance on 53 real-world tasks from the original `pinchbench/skill` repository. The task set covers productivity, research, writing, coding, analysis, email, memory, and skill-discovery categories. This wrapper expects the user to provide a pinned checkout of the original repository via `PINCHBENCH_ROOT` or the `benchmark_root` task parameter.

The wrapper runs the native PinchBench harness inside Docker. The native harness grades each task automatically, with an LLM judge, or both, and writes JSON results. The Inspect scorer reports the native aggregate score, or the mean of per-task numeric scores when the native output does not include an aggregate, using Inspect `mean()` and `stderr()` metrics.

### WildClawBench

WildClawBench evaluates long-horizon, real-world agent performance on 60 human-authored bilingual and multimodal tasks from the original `internlm/WildClawBench` repository. The tasks span six categories: productivity flow, code intelligence, social interaction, search and retrieval, creative synthesis, and safety alignment. This wrapper expects the user to provide a pinned checkout of the original repository via `WILDCLAWBENCH_ROOT` or the `benchmark_root` task parameter.

The wrapper runs the native WildClawBench batch runner inside Docker. Native grading combines deterministic rule-based checks, environment-state auditing, and LLM/VLM judge scores. The Inspect scorer parses native `summary*.json` or per-task `score.json` files and reports the mean native `overall_score` across scored tasks, using Inspect `mean()` and `stderr()` metrics.

## Pinned External Dependencies

External assets and runtime dependencies are pinned for reproducibility:

- PinchBench original repository: `https://github.com/pinchbench/skill.git` at `819384ae830492365b8363fc26bc2602e73f216d`.
- WildClawBench original repository: `https://github.com/internlm/WildClawBench.git` at `86d71447413d38f38740a021cb776f64eb396ee0`.
- Docker base image: `node:22-bookworm@sha256:c601a46abb4d2ab80a9dc3da208d50d1122642d53f17a101926ace71e5a9bf1c`.
- npm package: `openclaw@2026.6.10`.
- Python dependencies are pinned exactly in `pyproject.toml` and in both Dockerfiles.

The wrappers require Docker and an OpenAI-compatible model endpoint reachable from the host.

The model endpoint may be unauthenticated. `PINCHBENCH_API_KEY` and `WILDCLAWBENCH_API_KEY` (or the
`api_key` task parameter) are optional; when they are unset the wrappers supply a placeholder key so
that OpenClaw still resolves the provider, which is what locally served endpoints such as vLLM,
SGLang, and Ollama need.

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
   uv run inspect eval src/pinchbench/pinchbench.py@pinchbench
   ```

## Running WildClawBench

1. Clone the upstream WildClawBench repository at the tested commit:

   ```bash
   git clone https://github.com/internlm/WildClawBench.git /path/to/WildClawBench
   git -C /path/to/WildClawBench checkout 86d71447413d38f38740a021cb776f64eb396ee0
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
   uv run inspect eval src/wildclawbench/wildclawbench.py@wildclawbench
   ```
