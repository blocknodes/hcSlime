"""设备（device）域：LLM 为中心，无规则层。

对外暴露唯一的 Domain 实例。工具定义硬编码在 pipeline.py（_TOOL_SPECS），
不再依赖 schema.json；决策 = 单次 LLM 出 {tool, params}，后处理只做 object 的
canonical 词表归一（协议层），失败兜底 common_control。
"""

from __future__ import annotations

from app.domain import Domain, Tool

from .pipeline import _TOOL_SPECS, pipeline as _pipeline

_SPECS_TO_TOOLS = [Tool(name=s["name"], domain_key="device", domain="智能设备",
                        description=s["description"],
                        parameters={"type": "object", "required": list(s["required"])})
                   for s in _TOOL_SPECS]

domain: Domain = Domain(
    key="device",
    name="智能设备",
    tools=_SPECS_TO_TOOLS,
    pipeline=_pipeline,
)