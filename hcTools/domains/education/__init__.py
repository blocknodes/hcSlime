"""教育（education）域：LLM 为中心，无规则层。

对外暴露唯一的 Domain 实例。工具定义硬编码在 pipeline.py（_TOOL_SPECS），
不再读 schema.json；决策是单次 LLM 出 {tool, params}，失败兜底 edu_fuzzy_search。
"""

from __future__ import annotations

from app.domain import Domain, Tool

from .pipeline import _TOOL_SPECS, pipeline as _pipeline

_SPECS_TO_TOOLS = [Tool(name=s["name"], domain_key="education", domain="教育",
                        description=s["description"],
                        parameters={"type": "object", "required": list(s["required"])})
                   for s in _TOOL_SPECS]

domain: Domain = Domain(
    key="education",
    name="教育",
    tools=_SPECS_TO_TOOLS,
    pipeline=_pipeline,
)