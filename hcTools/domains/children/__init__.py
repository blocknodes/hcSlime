"""少儿（children）域：LLM 为中心，无规则层。

决策全部在 `pipeline.py`：一次 LLM 调用直接出 {tool, params}；失败才整句兜底到
educ_fuzzy_search。没有 badcase / 精确缓存 / 规则 / fewshot / 后处理改写。

加载前提：hcTools 在 sys.path 上（`uvicorn app.main:app` 或从 hcTools 目录启动 python）。
"""

from __future__ import annotations

from app.domain import Domain, Tool

from .pipeline import _TOOL_SPECS, pipeline as _pipeline

_SPECS_TO_TOOLS = [Tool(name=s["name"], domain_key="children", domain="少儿",
                        description=s["description"],
                        parameters={"type": "object", "required": list(s["required"])})
                   for s in _TOOL_SPECS]

domain: Domain = Domain(
    key="children",
    name="少儿",
    tools=_SPECS_TO_TOOLS,
    pipeline=_pipeline,
)