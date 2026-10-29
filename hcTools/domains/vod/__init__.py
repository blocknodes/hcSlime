"""影视（vod）域：LLM 为中心，无规则层。

决策全部在 `pipeline.py`：BM25 fewshot + 单次 LLM 出 {tool, params}，LLM 失败才整句兜底到
vod_fuzzy_search。没有 badcase / 精确缓存 / 规则 / 后处理改写。

加载前提：hcTools 在 sys.path 上（`uvicorn app.main:app` 或从 hcTools 目录启动 python）。
"""

from __future__ import annotations

from app.domain import Domain
from app.domain import Tool

from .pipeline import _TOOL_SPECS
from .pipeline import pipeline as _pipeline

# 只保留协议层需要的参数信息（retext 补填白名单）；其余参数校验全在 LLM prompt 里。
_SPECS_TO_TOOLS = [Tool(name=s["name"], domain_key="vod", domain="影视",
                        description=s["description"],
                        parameters={"type": "object", "required": list(s["required"])})
                   for s in _TOOL_SPECS]

domain: Domain = Domain(
    key="vod",
    name="影视",
    tools=_SPECS_TO_TOOLS,
    pipeline=_pipeline,
)
