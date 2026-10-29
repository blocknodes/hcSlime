"""LLM metadata 透传：hcTools 每次调 LLM 都带 app=hcTools，hcProxy 才能分流捕获。

不连网：用 httpx.MockTransport 拦下请求，直接检查发出的 payload。
（本仓没装 pytest-asyncio，故用 asyncio.run 驱动协程。）
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from app import llm  # noqa: E402
from app.metadata import (  # noqa: E402
    APP,
    PURPOSE_FILL,
    PURPOSE_ONECALL,
    PURPOSE_SELECT,
    build_llm_metadata,
)


#: 真·httpx.AsyncClient。`llm.httpx` 就是 httpx 模块本身，monkeypatch 它会连测试里的
#: 引用一起改掉，故必须先留一份原类再用来造 MockTransport 客户端。
_REAL_ASYNC_CLIENT = httpx.AsyncClient


class _Capture:
    """收集 chat() 实际发出的 payload，并回一个最小可用的 OpenAI 响应。"""

    def __init__(self):
        self.payloads: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )


def _patch_client(monkeypatch, handler):
    """把 chat() 里的 httpx.AsyncClient 换成走 MockTransport 的客户端。"""
    monkeypatch.setattr(
        llm.httpx, "AsyncClient",
        lambda **kw: _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setattr(llm.get_settings(), "api_base", "http://proxy.test/v1", raising=False)


@pytest.fixture
def captured(monkeypatch):
    """把 llm 的 httpx.AsyncClient 换成 MockTransport，并保证 api_base 已配置。"""
    cap = _Capture()
    _patch_client(monkeypatch, cap.handler)
    return cap


def _chat(messages, **kw):
    return asyncio.run(llm.chat(messages, **kw))


# ---- build_llm_metadata 契约 ----

def test_metadata_app_is_always_hctools():
    """app 恒为 hcTools——hcProxy 的 app_tag() 只认这个键，缺了会落进 _raw/。"""
    md = build_llm_metadata(purpose=PURPOSE_SELECT)
    assert md["app"] == APP == "hcTools"
    assert md["purpose"] == "select"


def test_metadata_omits_empty_keys():
    """空值不写 key：网关 metadata 索引对空值不友好，缺 key 更能表达「本调用无此上下文」。"""
    md = build_llm_metadata(purpose=PURPOSE_FILL, domain="music", tool="music_song_search")
    assert md == {"app": "hcTools", "purpose": "fill",
                  "domain": "music", "tool": "music_song_search"}
    assert "stage" not in md and "attempt" not in md


def test_metadata_attempt_only_when_retrying():
    """attempt 从 1 计；首次不写，重试才写——便于统计「首次输出不合法」的比例。"""
    assert "attempt" not in build_llm_metadata(purpose=PURPOSE_ONECALL, attempt=1)
    assert build_llm_metadata(purpose=PURPOSE_ONECALL, attempt=2)["attempt"] == "2"


def test_metadata_extra_flattened_to_str():
    md = build_llm_metadata(purpose=PURPOSE_ONECALL, extra={"n_shots": 8, "cached": False})
    assert md["n_shots"] == "8" and md["cached"] == "False"


# ---- chat() 透传 ----

def test_chat_sends_metadata(captured):
    _chat([{"role": "user", "content": "放首歌"}],
          metadata=build_llm_metadata(purpose=PURPOSE_ONECALL, domain="music"))
    sent = captured.payloads[-1]
    assert sent["metadata"] == {"app": "hcTools", "purpose": "onecall", "domain": "music"}


def test_chat_omits_metadata_when_absent(captured):
    """不传 metadata → 请求体里不能出现该 key（空 dict 也视同不传）。"""
    _chat([{"role": "user", "content": "放首歌"}])
    assert "metadata" not in captured.payloads[-1]
    _chat([{"role": "user", "content": "放首歌"}], metadata={})
    assert "metadata" not in captured.payloads[-1]


def test_chat_metadata_does_not_touch_messages(captured):
    """metadata 是搭车字段，不得改写 messages——否则会污染决策。"""
    original = [{"role": "user", "content": "放首歌"}]
    _chat(original, metadata=build_llm_metadata(purpose=PURPOSE_FILL, domain="music"))
    sent = captured.payloads[-1]
    assert sent["messages"] == original
    assert "metadata" not in original[0]


def test_chat_metadata_survives_retry(monkeypatch):
    """重试要沿用同一份 metadata（失败重发的仍是同一次业务调用）。"""
    attempts: list[dict] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(json.loads(request.content.decode("utf-8")))
        if len(attempts) == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    _patch_client(monkeypatch, flaky)
    monkeypatch.setattr(llm, "REQUEST_TIMEOUT", 1.0)
    monkeypatch.setattr(llm.asyncio, "sleep", lambda *_: _noop())

    md = build_llm_metadata(purpose=PURPOSE_SELECT, domain="audio")
    msg = _chat([{"role": "user", "content": "打开收音机"}], metadata=md)
    assert len(attempts) == 2
    assert all(p["metadata"] == md for p in attempts)
    assert msg.get("content") == "ok"


async def _noop():
    return None
