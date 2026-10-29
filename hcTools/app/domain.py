"""域契约与域加载器。

每个域（domains/<key>/）由一个 __init__.py 导出满足统一契约的 Domain 实例。
内核只依赖这个契约，不关心域内部如何实现——这是域隔离的关键。

内核提供默认 select_prompt / fill_prompt，域可覆盖；缺省前后处理为恒等。
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("hcTools.domain")

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"

# 网关 grammar 编译器常不支持的 JSON Schema 关键字，注入前裁掉（只是校验性约束）。
UNSUPPORTED_KEYS = {
    "uniqueItems", "minItems", "maxItems", "minLength", "maxLength",
    "minProperties", "maxProperties", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "allOf", "if", "then", "else", "default", "examples", "format",
}
META_KEYS = {"$schema", "name", "title"}

DEFAULT_SELECT_PROMPT = """你是电视语音助手的工具选择器。根据用户的话，从候选工具中选出最合适的一个。

原则：
- 严格按每个工具描述的适用场景匹配，不要凭工具名猜测。
- 优先参考“已标注好的同类问句”（它们是最可靠的路由先例）：从这些样例里挑与当前用户的话最像的，
  跟随它对应的工具。
- 只输出工具名本身，不要输出解释、不要包 JSON 对象、不要其他内容。
"""

DEFAULT_FILL_PROMPT = """你是电视语音助手的参数抽取器。已经为你选定了唯一工具，请根据用户的话填好它的参数。

原则：
- 通过 function calling 调用给定的这个工具，不要输出解释文字。
- 只抽取用户明确表达的信息，不臆造、不补充用户没说的条件；用户没提的可选字段留空。
- 严格按参数 schema 里每个字段的 description / enum 写的口径取值。
- 如果有 retext 字段，原样填写用户完整原话，不改写、不截断。
"""


@dataclass
class Tool:
    name: str
    domain_key: str       # 英文域标识，如 "vod"
    domain: str           # 内部中文域名，如 "影视"
    description: str
    parameters: dict[str, Any]

    def as_openai_function(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": " ".join(self.description.split())[:1024],
                "parameters": _prune(self.parameters, top=True),
            },
        }


@dataclass
class Domain:
    key: str                          # 域标识，如 "vod"
    name: str                         # 中文名，如 "影视"
    tools: list[Tool] = field(default_factory=list)

    # ---- 可选（缺省用内核默认）----
    select_prompt: str = DEFAULT_SELECT_PROMPT
    fill_prompt: str = DEFAULT_FILL_PROMPT
    fewshot: list[dict[str, Any]] = field(default_factory=list)
    example_bank: Any = None          # 动态 few-shot 检索池（app.examples.ExampleBank）

    # ---- 可选钩子（缺省为恒等）----
    preprocess: Callable | None = None      # (req) -> req
    postprocess: Callable | None = None    # (tool_name, params) -> params
    rule_select: Callable | None = None    # (query) -> (tool, params|None)|None；命中则跳过 LLM select
    badcase_lookup: Callable | None = None  # (query) -> (tool, params)|None；L2 最高优先，允许覆盖一切
    fallback: Callable | None = None        # (query) -> (tool, params)|None；L3 有序兜底，保证非空
    # 整段流水线接管：async (req, domain) -> Prediction|None。
    # 非 None 即直接作为最终结果（engine 不再介入）；None/异常则落回上述三层流水线。
    # 供「不走 select→fill 两段式」的域自定义决策主轴（如 vod 的 fewshot 缓存 + 单次 LLM）。
    pipeline: Callable | None = None

    @property
    def tools_by_name(self) -> dict[str, Tool]:
        return {t.name: t for t in self.tools}


def _prune(node: Any, top: bool = False) -> Any:
    """递归去掉网关不支持的校验关键字；properties/definitions 的下一层是字段名空间，不删。"""
    if isinstance(node, list):
        return [_prune(item) for item in node]
    if not isinstance(node, dict):
        return node
    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in UNSUPPORTED_KEYS or (top and key in META_KEYS):
            continue
        if key in {"properties", "definitions", "$defs", "patternProperties"} and isinstance(value, dict):
            result[key] = {name: _prune(spec) for name, spec in value.items()}
        else:
            result[key] = _prune(value)
    result.setdefault("type", "object") if top else None
    return result


def _unwrap(schema: Any) -> Any:
    """有的 schema 把入参包在 parameters 里（实为完整接口描述），剥一层取真入参。"""
    if not isinstance(schema, dict) or isinstance(schema.get("properties"), dict):
        return schema
    inner = schema.get("parameters")
    return inner if isinstance(inner, dict) else schema


def _load_schema_json(dirpath: Path) -> list[Tool]:
    """从 domains/<key>/schema.json 加载工具定义。"""
    schema_path = dirpath / "schema.json"
    if not schema_path.exists():
        return []
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    domain = payload.get("domain", "")
    domain_key = payload.get("domain_key", "")
    tools: list[Tool] = []
    for item in payload.get("tools", []):
        raw = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
        params = _unwrap(raw)
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        tools.append(Tool(
            name=item["tool_name"],
            domain_key=domain_key,
            domain=domain,
            description=item.get("description", "") or item.get("title", ""),
            parameters=params,
        ))
    return tools


def load_domain(key: str) -> Domain | None:
    """加载单个域。先尝试 import domains/<key>（导出 Domain 实例），失败再回退到 schema.json。失败均返回 None。"""
    dirpath = DOMAINS_DIR / key
    if not dirpath.is_dir():
        logger.warning("域目录不存在：%s", key)
        return None
    try:
        module = importlib.import_module(f"domains.{key}")
        domain = getattr(module, "domain", None)
        if isinstance(domain, Domain):
            return domain
    except Exception as exc:  # noqa: BLE001 单域失败只影响本域
        logger.error("域 %s __init__ 加载失败：%s", key, exc)
        return None
    try:
        tools = _load_schema_json(dirpath)
        if not tools:
            logger.warning("域 %s 无可用工具，跳过", key)
            return None
        return Domain(key=key, name=tools[0].domain or key, tools=tools)
    except Exception as exc:  # noqa: BLE001
        logger.error("域 %s schema 加载失败：%s", key, exc)
        return None


_domains_cache: dict[str, Domain] | None = None


def load_domains() -> dict[str, Domain]:
    """扫描 domains/* 独立加载；单域失败只禁用该域，其余正常。"""
    global _domains_cache
    if _domains_cache is not None:
        return _domains_cache
    domains: dict[str, Domain] = {}
    if DOMAINS_DIR.is_dir():
        for dirpath in sorted(DOMAINS_DIR.iterdir()):
            if not dirpath.is_dir() or dirpath.name.startswith("_"):
                continue
            domain = load_domain(dirpath.name)
            if domain is not None:
                domains[domain.key] = domain
    _domains_cache = domains
    return domains