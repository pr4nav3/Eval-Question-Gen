#!/usr/bin/env python3
"""Small chat-completions JSON client for Eval-Question-Gen."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_LLM_MODEL = "private-large"


@dataclass(frozen=True)
class LLMConfig:
    url: str
    model: str
    api_key: str
    timeout_seconds: int = 300
    retries: int = 2


class LLMEmptyContentError(ValueError):
    """Raised when the endpoint responds but omits usable text content."""


class LLMInvalidJSONError(ValueError):
    """Raised when the endpoint returns text that cannot be parsed as JSON."""


def chat_completions_url_from_base(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def litellm_chat_completions_url_from_base(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def load_env_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path:
        return values
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        values[key] = value
    return values


def apply_env_file(path: str, *, override: bool = False) -> dict[str, str]:
    values = load_env_file(path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    if "JUSPAY_API_KEY" not in os.environ and os.environ.get("LITELLM_API_KEY"):
        os.environ["JUSPAY_API_KEY"] = os.environ["LITELLM_API_KEY"]
    return values


def default_llm_url() -> str:
    explicit = os.environ.get("LLM_URL") or os.environ.get("OPENAI_CHAT_COMPLETIONS_URL")
    if explicit:
        return explicit
    litellm_base = os.environ.get("LITELLM_BASE_URL")
    if litellm_base:
        return litellm_chat_completions_url_from_base(litellm_base)
    base = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    )
    return chat_completions_url_from_base(base)


def default_llm_api_key() -> str:
    return (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("LITELLM_API_KEY")
        or os.environ.get("JUSPAY_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )


def default_llm_model() -> str:
    return (
        os.environ.get("EVAL_LLM_MODEL")
        or os.environ.get("LLM_MODEL")
        or DEFAULT_LLM_MODEL
    )


def parse_json_response(text: str) -> Any:
    content = (text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = content.find(opener)
            end = content.rfind(closer)
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    pass
    raise ValueError("LLM response did not contain valid JSON")


def content_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        if isinstance(part.get("text"), str):
            return part["text"]
        if isinstance(part.get("content"), str):
            return part["content"]
    return ""


def extract_choice_content(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMEmptyContentError("response did not contain choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message")
    content: Any = None
    if isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        joined = "".join(content_part_text(part) for part in content).strip()
        if joined:
            return joined
    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return text
    message_keys = sorted(message.keys()) if isinstance(message, dict) else []
    raise LLMEmptyContentError(
        "response did not contain usable message content "
        f"(finish_reason={choice.get('finish_reason')!r}, message_keys={message_keys})"
    )


def retry_delay_seconds(attempt: int, *, retry_after: str | None = None) -> float:
    base_delay = min(2 ** attempt, 30)
    if retry_after:
        try:
            return min(max(float(retry_after), base_delay), 60)
        except ValueError:
            pass
    return base_delay


def call_llm_json(
    *,
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    temperature: float = 0.1,
    max_tokens: int = 3000,
) -> Any:
    if not config.url:
        raise ValueError("LLM URL is not configured")
    payload = json.dumps(
        {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    last_error: Exception | None = None
    for attempt in range(max(1, config.retries + 1)):
        request = urllib.request.Request(config.url, data=payload, headers=headers)
        started_at = time.time()
        try:
            with urllib.request.urlopen(
                request,
                timeout=config.timeout_seconds,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = extract_choice_content(body)
            try:
                parsed = parse_json_response(content)
            except ValueError as exc:
                excerpt = content[:800].replace("\n", "\\n")
                raise LLMInvalidJSONError(
                    f"{exc}; raw_content_excerpt={excerpt!r}"
                ) from exc
            return {
                "parsed": parsed,
                "raw_content": content,
                "duration_ms": round((time.time() - started_at) * 1000),
                "attempt": attempt + 1,
            }
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
            time.sleep(retry_delay_seconds(attempt, retry_after=exc.headers.get("Retry-After")))
        except LLMEmptyContentError as exc:
            last_error = exc
            time.sleep(retry_delay_seconds(attempt))
        except LLMInvalidJSONError as exc:
            last_error = exc
            time.sleep(retry_delay_seconds(attempt))
        except Exception as exc:
            last_error = exc
            break
    raise RuntimeError(f"LLM call failed: {last_error}") from last_error
