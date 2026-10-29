"""教育（education）域：LLM 为中心，无规则层、无 fewshot、无 badcase。

决策全部在 `pipeline.py`：一次 LLM 调用直接出 {tool, params}；LLM 失败/输出不合法
→ edu_fuzzy_search 整句兜底。没有 badcase / 精确缓存 / 规则 / 后处理改写。
"""

from __future__ import annotations

import logging
import re
from datetime import date

from app.llm import _parse_json_block, chat
from app.metadata import PURPOSE_ONECALL, build_llm_metadata
from app.models import Prediction, PredictRequest

logger = logging.getLogger("hcTools.domains.education")

FALLBACK_TOOL = "edu_fuzzy_search"
MAX_ATTEMPTS = 2

# 2 个工具的完整定义（来自原 domains/education/schema.json，全量内联，不再依赖 schema.json）。
_TOOL_SPECS = [
    {
        'name': 'edu_search',
        'description': '教育媒资组合式搜索工具(edu_search)。当用户表达搜索、查找、推荐K12教育内容的意图时触发，支持按 7个维度筛选：6个精确匹配维度(title/grade/first_subject/figures/material_version/semester)、1 个状态维度(is_fee 0=免费|1=付费)。同一维度多值通过 values 数组+operator 字段指定逻辑(or任一匹配/and全部匹配, 默认or)；不同维度间通过 BoolNode 组合：and(与)、or(或)、not(非)，支持任意嵌套',
        'required': ['query'],
    },
    {
        'name': 'edu_fuzzy_search',
        'description': '教育媒资非明确的复杂语义搜索。用于处理用户以自然语言、模糊描述、组合条件或问答等方式表达的教育内容查找意图。\\n适用场景：1. 拓展课媒资检索：如“三年级习题辅导”、“二年级下册数学试卷”、“考研帮科学实验视频”；2. 素质课搜索：如“音乐课”、“美术课”、“我想学围棋课”；3. 知识点媒资检索：如“一元二次方程解题知识点课程”、“初中英语语法知识点辅导”、“文言文知识点”；\\n不适用场景：简单明确的查询不要使用该工具，如“一年语文课程”、“打开英语课程”、“看初中语文课程”等直接指定了明确内容名称或固定分类的请求，应使用明确的检索工具。',
        'required': ['retext'],
    },
    {
        'name': 'fan_knowledge_agent',
        'description': '教育知识问答（fan_knowledge_agent）。学生/家长关于学习内容的任何疑问：课本/知识点疑问、解题思路、学习方法、百科讲解、课程/老师/教材相关问答，都走教育知识问答链路（education_qa 场景）。不做课程媒资检索（那是 edu_search/edu_fuzzy_search）。\\n参数：{"messages":"用户原话整体传入（不要改写）"}。',
        'required': ['messages'],
    },
]

_TOOL_REQUIRED = {s["name"]: list(s["required"]) for s in _TOOL_SPECS}
_RETEXT_TOOLS = {"edu_fuzzy_search"}

# 第二层兜底：本域知识问答（education_qa 场景）。
from app.fanparams import (  # noqa: E402
    domain_fan_params,
    looks_like_qa as _looks_like_qa,
    rebuild_fan_params,
)

_FIELD_CATALOG = """- grade 年级（如 一年级/七年级/高三）
- figures 学段（小学/初中/高中）
- first_subject 科目（语文/数学/英语/物理/化学…）
- material_version 教材版本（人教版/北师大版/湘教版…）
- semester 学期（上半学期/下半学期）
- is_fee 免付费（0=免费，1=付费）
- k12_category 课程类型（默认取值 同步课 / 同步拓展课，多值用 values+operator=or）"""

_SYSTEM_TMPL = """你是教育内容意图解析器。读用户的话，输出该调用的工具名和参数。

【工具】
{tool_briefs}

【怎么选工具】
- 学习**知识问答/讲解/百科**（"为什么""怎么做""什么意思"、知识点不理解、解题思路）→ fan_knowledge_agent。
- **知识若有出处/老师/题目需要检索**才走检索：
  - **能抽出课程结构、维度**（年级/学段/科目/教材版本/学期/免费）→ edu_search。
    典型：「3 年级的课」「语文人教版同步课」「初一上数学」。
  - **其余一律** → edu_fuzzy_search，把整句话填进 retext。（抽象课程、特色课、兴趣类、复习/辅导/冲刺…）

【参数怎么写】
- edu_search 的 params 形如 {{"query": <条件树>}}：
  - 单值：{{"field":"grade","value":"高三"}}
  - 多值：{{"field":"first_subject","values":["语文","数学"],"operator":"or"}}
  - 组合：{{"and":[...]}} / {{"or":[...]}} / {{"not":{{...}}}}
  - **k12_category 是默认维度**：只要是 K12 同步课程类检索，补上
    {{"field":"k12_category","values":["同步课","同步拓展课"],"operator":"or"}}。
  - 条件只有一个时，query 直接写那个叶子，不要包 and。
- edu_fuzzy_search 的 params 形如 {{"retext": "用户原话"}}（整句，不要改写）。
- fan_knowledge_agent 的 params 形如 {{"messages":"用户原话整体传入（不要改写）"}}（教育知识问答）。

【可用字段】
{fields}

【输出】
只输出一个 JSON 对象，形如 {{"tool":"工具名","params":{{...}}}}，不要任何解释。
只抽用户明确表达的信息，不臆造；用户没提的字段不要出现。
"""


def norm_retext(q: str) -> str:
    """edu_fuzzy_search 的 retext 约定：strip + 直引号→弯引号（对齐 golden 的书写）。

    实测：identity 385/386、strip_punct 只有 293/386（教育域 golden **保留**标点），
    故这里只做引号规整，不做标点切除。
    """
    q = (q or "").strip()
    return re.sub(r'"([^"]*)"', lambda m: "“" + m.group(1) + "”", q)


def _tool_briefs(tools) -> str:
    lines = []
    for t in tools:
        desc = " ".join(t.description.replace("\\n", " ").split())
        lines.append(f"- {t.name}：{desc[:900]}")
    return "\n".join(lines)


def build_messages(domain, query: str) -> list[dict]:
    system = _SYSTEM_TMPL.format(
        today=date.today().isoformat(), tool_briefs=_tool_briefs(domain.tools), fields=_FIELD_CATALOG
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"用户的话：{query}\n只输出 JSON："},
    ]


def _parse(msg: dict, valid: set[str]) -> tuple[str, dict] | None:
    """从模型输出里取 (tool, params)；工具名不合法 / 不是 JSON → None。"""
    obj = _parse_json_block(str(msg.get("content") or ""))
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool") or obj.get("tool_name") or ""
    params = obj.get("params")
    if tool not in valid or not isinstance(params, (dict, type(None))):
        return None
    return tool, params or {}


def _required(domain, tool: str) -> set[str]:
    return _TOOL_REQUIRED.get(tool, set())


async def pipeline(req: PredictRequest, domain) -> Prediction:
    query = req.query or ""
    messages = build_messages(domain, query)
    valid = set(domain.tools_by_name)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            msg = await chat(
                messages,
                metadata=build_llm_metadata(purpose=PURPOSE_ONECALL, domain=domain.key, attempt=attempt),
            )
        except Exception as exc:  # noqa: BLE001 网关失败 → 兜底
            logger.error("education LLM 调用失败：%s", exc)
            break
        parsed = _parse(msg, valid)
        if parsed is None:
            logger.warning("education LLM 输出不合法（第 %d 次）：%r", attempt, str(msg.get("content"))[:200])
            continue
        tool, params = parsed
        if tool in _RETEXT_TOOLS and "retext" not in params:
            params["retext"] = norm_retext(query)
        if tool == "fan_knowledge_agent":
            params = rebuild_fan_params(params, scene_hint="education_qa", fallback_query=query)
        return Prediction(domain=domain.name, tool=tool, params=params, hit_source="llm")

    # 两层兜底：fuzzy 第一层；query 呈知识问答形态 → 本域 fan（education_qa 场景）。
    if _looks_like_qa(query):
        return Prediction(domain=domain.name, tool="fan_knowledge_agent",
                          params=domain_fan_params("education", query), hit_source="fallback_qa")
    return Prediction(domain=domain.name, tool=FALLBACK_TOOL, params={"retext": query},
                      hit_source="fallback")