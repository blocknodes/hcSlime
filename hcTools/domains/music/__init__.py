"""音乐（music）域：LLM 为中心，无规则层。

对外暴露唯一的 Domain 实例。工具定义硬编码在 pipeline.py（_TOOL_SPECS），
不再依赖 schema.json；决策 = 单次 LLM 出 {tool, params}，失败兜底 music_song_search。
"""

from __future__ import annotations

from app.domain import Domain, Tool

from .pipeline import _TOOL_SPECS, pipeline as _pipeline

_SPECS_TO_TOOLS = [Tool(name=s["name"], domain_key="music", domain="音乐",
                        description=s["description"],
                        parameters={"type": "object", "required": list(s["required"])})
                   for s in _TOOL_SPECS]

domain: Domain = Domain(
    key="music",
    name="音乐",
    tools=_SPECS_TO_TOOLS,
    pipeline=_pipeline,
)