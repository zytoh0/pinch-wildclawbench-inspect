from __future__ import annotations

from pathlib import Path


class CompatibilityPatchError(RuntimeError):
    """Raised when the pinned upstream runner no longer matches expected code."""


def _replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise CompatibilityPatchError(
            f"Expected upstream snippet not found: {old[:80]!r}"
        )
    return source.replace(old, new, 1)


def apply_pinchbench_openclaw_compat(benchmark_dir: Path) -> None:
    """Patch a copied PinchBench checkout for OpenAI-compatible local execution.

    PinchBench delegates to OpenClaw. The upstream runner assumes OpenClaw gateway
    credentials for non-local runs and stores custom model metadata in a location
    that recent OpenClaw versions do not discover consistently. Inspect evals run
    this patch only against the per-run copy of the pinned upstream checkout.
    """
    runner = benchmark_dir / "scripts" / "lib_agent.py"
    source = runner.read_text(encoding="utf-8")
    source = _replace_once(
        source, "use_local = fws_env is not None", "use_local = True"
    )
    source = _replace_once(
        source,
        '"--model",\n                model_id,',
        '"--model",\n                f"custom/{model_id}" if base_url and "/" not in model_id else model_id,',
    )
    source = _replace_once(
        source,
        'bench_agent_dir = _get_agent_store_dir(agent_id) / "agent"',
        'bench_store_dir = _get_agent_store_dir(agent_id)\n    bench_agent_dir = bench_store_dir / "agent"',
    )
    # OpenClaw aborts a request after 120 s without model output by default;
    # self-hosted endpoints under load routinely exceed that on long prompts.
    source = _replace_once(
        source,
        '"apiKey": key_ref,',
        '"apiKey": key_ref,\n            "timeoutSeconds": int(os.environ.get("PINCHBENCH_PROVIDER_TIMEOUT_SECONDS", "600")),\n            **({"authHeader": True, "request": {"auth": {"mode": "authorization-bearer", "token": key_ref}}} if key_ref else {}),',
    )
    # The direct ``openai/<model>`` judge backend hardcodes api.openai.com. Honour
    # OPENAI_BASE_URL (the OpenAI SDK convention) so the judge can be served by
    # the same OpenAI-compatible endpoint as the agent via the per-run proxy.
    source = _replace_once(
        source,
        '"https://api.openai.com/v1/chat/completions",',
        'os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions",',
    )
    # Reasoning models spend the judge's 2048-token completion budget thinking
    # and return an empty answer; let the adapter raise the cap.
    source = _replace_once(
        source,
        '"max_completion_tokens": 2048,',
        '"max_completion_tokens": int(os.environ.get("PINCHBENCH_JUDGE_MAX_TOKENS", "2048")),',
    )
    source = _replace_once(
        source,
        'bench_models.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")',
        "models_payload = json.dumps(data, indent=2, ensure_ascii=False)\n"
        '        bench_models.write_text(models_payload, "utf-8")\n'
        '        (bench_store_dir / "models.json").write_text(models_payload, "utf-8")\n'
        '        global_config = Path.home() / ".openclaw" / "openclaw.json"\n'
        "        global_config.parent.mkdir(parents=True, exist_ok=True)\n"
        "        try:\n"
        '            global_data = json.loads(global_config.read_text("utf-8-sig")) if global_config.exists() else {}\n'
        "        except Exception:\n"
        "            global_data = {}\n"
        '        global_providers = global_data.setdefault("models", {}).setdefault("providers", {})\n'
        '        global_providers["custom"] = providers["custom"]\n'
        '        global_config.write_text(json.dumps(global_data, indent=2, ensure_ascii=False), "utf-8")',
    )
    runner.write_text(source, encoding="utf-8")


def main() -> None:
    apply_pinchbench_openclaw_compat(Path.cwd())


if __name__ == "__main__":
    main()
