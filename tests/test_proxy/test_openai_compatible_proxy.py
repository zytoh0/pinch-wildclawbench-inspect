from __future__ import annotations

import importlib.util
from pathlib import Path

PROXY_PATH = Path(__file__).parents[2] / "src" / "_openai_compatible_proxy.py"
spec = importlib.util.spec_from_file_location("openai_compatible_proxy", PROXY_PATH)
proxy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(proxy)


def test_merge_extra_body_merges_nested_dicts_and_overrides_scalars() -> None:
    payload = {
        "model": "m",
        "max_tokens": 128,
        "chat_template_kwargs": {"foo": 1},
    }
    merged = proxy.merge_extra_body(
        payload, {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0}
    )
    assert merged["chat_template_kwargs"] == {"foo": 1, "enable_thinking": False}
    assert merged["temperature"] == 0
    assert merged["max_tokens"] == 128
    # The caller's payload is left untouched.
    assert payload["chat_template_kwargs"] == {"foo": 1}
