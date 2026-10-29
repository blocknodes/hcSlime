"""LLM 调用的业务 metadata 构造（OpenAI-format 请求体 `metadata` 字段）。

对齐 hcAgent 的做法（`hcAgent/app/prompts.py::build_llm_metadata`）：调 LLM 时把业务上下文
搭车在请求体里，网关/代理可采集审计。hcProxy 按 `metadata.app` 分流捕获文件
（`data/captures/<app>/YYYY-MM-DD.jsonl`），故 `app` 恒为 "hcTools"。

与 hcAgent 的区别（有意为之）：
- hcTools 是无状态单轮服务，没有 device/trace/same_turn 这些跨轮概念，不硬造；
- 只标「这一次调用在做什么」：purpose（select/fill/onecall）+ domain（+ tool/attempt）。

注意与 `PredictRequest.metadata` 区分：那个字段是**外部辅助信号**（分词/维表命中），
由 `app/engine.py::_metadata_hint()` 拼进 prompt；本模块是**链路审计标签**，只进请求体
metadata、绝不进 prompt。两者语义不同，不要合并。
"""
from __future__ import annotations

from typing import Any

#: 捕获分流用的 app 标记，hcProxy 依赖它把本服务落到 captures/hcTools/
APP = "hcTools"

#: 本服务里一次 LLM 调用的业务类型
PURPOSE_SELECT = "select"      # 两段式：选工具名（engine.select_tool）
PURPOSE_FILL = "fill"          # 两段式：按 schema 填参数（engine.fill_params）
PURPOSE_ONECALL = "onecall"    # 内核：单次 LLM 同时定 tool+params（pipeline_kernel）


def build_llm_metadata(
    *,
    purpose: str,
    domain: str = "",
    tool: str = "",
    stage: str = "",
    attempt: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """构造一次 LLM 调用的业务 metadata（扁平 dict[str, str]）。

    - purpose: select / fill / onecall（见上方常量）
    - domain:  域 key（vod/audio/music/…），便于按域统计 LLM 调用量
    - tool:    已选定的工具名（fill 阶段）；select/onecall 阶段未知则省略
    - stage:   可选流程阶段标记（如语料来源 source="bench"）
    - attempt: 第几次尝试（>1 才写入，用于统计「首次输出不合法」比例）
    - extra:   额外扁平键值，值统一 str() 化

    值全部为字符串、空值不写 key——网关的 metadata 索引对空值不友好，
    且「key 不存在」比「key 为空串」更好表达「本次调用没有这个上下文」。
    """
    md: dict[str, str] = {"app": APP, "purpose": purpose}
    if domain:
        md["domain"] = domain
    if tool:
        md["tool"] = tool
    if stage:
        md["stage"] = stage
    if attempt and attempt > 1:
        md["attempt"] = str(attempt)
    if extra:
        md.update({k: str(v) for k, v in extra.items()})
    return md
