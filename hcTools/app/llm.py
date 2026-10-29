"""OpenAI 兼容网关的异步最小客户端：带超时与重试，支持 native function calling。

metadata：可选的业务上下文，原样放进 OpenAI-format 请求体的 `metadata` 字段
（对齐 hcAgent/app/llm.py）。网关多数实现会忽略它，但 hcProxy 会据此把调用分流到
`data/captures/<metadata.app>/` 并做链路审计。调用方用 app/metadata.py 的
`build_llm_metadata()` 构造统一 schema；**不传则请求体里不出现 metadata**。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from .config import get_settings

logger = logging.getLogger("hcTools.llm")

# 网关调用的固定参数
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 120.0
MAX_RETRY = 3


class LLMError(Exception):
    """网关调用失败或返回无法解析。"""


async def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """调用 /chat/completions，返回 choices[0].message。失败抛 LLMError。

    metadata 透传到请求体的 metadata 字段（链路审计/计费标签，hcProxy 依赖它分流）；
    不改写 messages / tools，也不参与任何决策。空 dict 视同不传。
    """
    settings = get_settings()
    if not settings.api_base:
        raise LLMError("HC_API_BASE 未配置（网关地址）")

    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
    }
    if metadata:
        payload["metadata"] = metadata
    if not settings.model.startswith("gpt-5"):
        payload["temperature"] = TEMPERATURE
        # 参考 hcAgent / compare：禁用 thinking。baseline 不开会在 content 里只回 " thinking"。
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    last_error = ""
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for attempt in range(MAX_RETRY):
            try:
                resp = await client.post(settings.chat_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                logger.info(
                    "LLM req %s | %s | md=%s\n"
                    "  >> IN  %s\n"
                    "  << OUT %s  (%.0fms)",
                    settings.model,
                    " ".join((m.get("role", "?") for m in messages)),
                    json.dumps(metadata, ensure_ascii=False) if metadata else "-",
                    json.dumps(messages, ensure_ascii=False),
                    json.dumps(message, ensure_ascii=False),
                    (time.perf_counter() - started) * 1000,
                )
                return message
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                last_error = f"HTTP {code}: {exc.response.text[:200]}"
                # 4xx 里除 429 外都是请求本身问题，重试无意义
                if code != 429 and 400 <= code < 500:
                    break
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.5 * (attempt + 1))

    raise LLMError(last_error or "unknown gateway error")


def first_tool_call(message: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """从 message 里取出（工具名, 参数）。兼容模型把 JSON 塞进 content 的情况。"""
    calls = message.get("tool_calls") or []
    if calls:
        function = calls[0].get("function") or {}
        name = function.get("name", "")
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = None
        return name, args if isinstance(args, dict) else None

    # 兜底：content 里可能是 ```json ... ``` 或裸 JSON
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        args = _parse_json_block(content)
        if isinstance(args, dict):
            return args.get("tool", "") or args.get("name", ""), args.get("params") or args
    return "", None


def _parse_json_block(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1] if "```" in stripped[3:] else stripped[3:]
        stripped = stripped.lstrip("json").lstrip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None
