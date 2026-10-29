"""有声（audio）域：LLM 为中心，无规则层、无 fewshot、无 badcase。

决策全部在 `pipeline.py`：一次 LLM 调用直接出 {tool, params}；LLM 失败/输出不合法
→ audio_search 兜底（action=search，query=原话）。没有 badcase / 精确缓存 / 规则。
"""

from __future__ import annotations

import logging
from datetime import date

from app.llm import _parse_json_block, chat
from app.metadata import PURPOSE_ONECALL, build_llm_metadata
from app.models import Prediction, PredictRequest

logger = logging.getLogger("hcTools.domains.audio")

FALLBACK_TOOL = "audio_search"
MAX_ATTEMPTS = 2

# 2 个工具的完整定义（来自原 domains/audio/schema.json，全量内联，不再依赖 schema.json）。
_TOOL_SPECS = [
    {
        'name': 'audio_search',
        'description': '功能：把检索文本原样透传给第三方有声内容提供方（喜马拉雅）做全文搜索，召回有声专辑（听书/相声/评书/儿童故事/知识付费等）媒资并返回结果；本工具不做字段级结构化过滤，召回全部由喜马拉雅站内全文搜索完成。\\naction 选择：「我要听/播放」等起播意图用 play；「找/搜/查」等查找意图用 search；指向不明时默认 search。是否列表展示/直接起播/熄屏起播由下游按命中数与终端状态决定，LLM 无需指定。\\n适用场景：用户口播查找、听、播放某个有声专辑/主播/分类，典型说法「我想听郭德纲的相声」「播放三国演义评书」「来点有声书」。\\n不适用场景：音乐/歌曲搜索用 music_song_search；历史记录起播用 audio_history。',
        'required': ['action', 'query'],
    },
    {
        'name': 'audio_history',
        'description': '功能：查询用户的有声收听历史记录，支持按有声分类和精确时间范围筛选，返回符合条件的历史记录。\\n请求必须通过 action 指定操作类型：用户明确表达「播放/继续听/接着听」我听过的某个内容等从历史记录起播的意图时用 play；表达「查看/查一下/我听过什么/最近听了什么」等查询历史记录的意图时用 search；说法无明确指向时默认 search。\\n适用场景：用户表达了查看历史记录的意图时使用，如“我听过的有声书”、“最近听了什么”、“我听了什么”、“查一下我的播放记录”。\\n时间解析：time 按 Asia/Shanghai（北京时间）解析，格式为“yyyy-MM-dd HH:mm:ss TO yyyy-MM-dd HH:mm:ss”，起止时间均包含。',
        'required': ['action'],
    },
    {
        'name': 'fan_knowledge_agent',
        'description': '有声内容知识问答（fan_knowledge_agent）。用户对有声音频/听书/评书/主讲人等的任何疑问：内容属性问答（如"这本书讲了什么""郭德纲的代表作有哪些"）、知识类问题（"XX是什么""为什么""怎么用"），都走知识问答链路（audio_qa 场景）。不做具体媒资检索或起播（那是 audio_search/audio_history）。\\n 参数：{"messages":"用户原话整体传入（不要改写）"}。',
        'required': ['messages'],
    },
]

_TOOL_REQUIRED = {s["name"]: list(s["required"]) for s in _TOOL_SPECS}

# 第二层兜底：本域知识问答（fan_knowledge_agent）。LLM/audio_search 都解不出时，
# 落 audio_qa 场景的知识问答，避免掉进无 sceneType 的通用 qa。
from app.fanparams import domain_fan_params
from app.fanparams import rebuild_fan_params  # noqa: E402
from app.fanparams import looks_like_qa as _looks_like_qa  # noqa: E402

_SYSTEM_TMPL = """你是电视语音助手的音频内容意图解析器。读用户的话，输出该调用的工具名和参数。

【工具】
{tool_briefs}

【怎么选工具】
按下面的顺序判断，先命中先返回：
1. 用户明确要 **查/看收听历史**（播放记录、收听历史、最近听过、刚听的、昨天听的）
   → audio_history。
2. 有声内容的 **知识问答/内容属性问答/介绍**（「这本书讲了什么」「夜听/郭德纲的代表作」「评书的起源」）→ fan_knowledge_agent。
3. 其余一律 **audio_search**（找/听/播某个有声内容）。

【audio_search 参数】
```json
{{"action": "search" | "play", "query": "用户原话原样透传"}}
```
- action：起播意图（我要听/播放/请放）→ "play"；查找意图（找/搜/查/有没有/是什么）→ "search"；
  指向不明时默认 "search"。
- query：**用户原话原样透传**，不要改写、不要剔指令词（如「我想听郭德纲的相声」整体传入）。

【audio_history 参数】
```json
{{"action": "search" | "play", "time": "yyyy-MM-dd HH:mm:ss TO yyyy-MM-dd HH:mm:ss"}}
```
- action：查看 → "search"；播放/继续听历史中的内容 → "play"。
- time：只有用户提了时间（今天/昨天/上周/哪段时间）才给；否则省略。

【fan_knowledge_agent 参数】
```json
{{"messages": "用户原话整体传入（不要改写）"}}
```
对有声内容/知识类开放问答使用。

【输出】
只输出一个 JSON 对象，形如 {{"tool":"工具名","params":{{...}}}}，不要任何解释。
只抽用户明确表达的信息，不臆造；用户没提的可选字段不要出现。
"""


def _tool_briefs(tools) -> str:
    lines = []
    for t in tools:
        desc = " ".join(t.description.replace("\\n", " ").split())
        lines.append(f"- {t.name}：{desc[:900]}")
    return "\n".join(lines)


def build_messages(domain, query: str) -> list[dict]:
    system = _SYSTEM_TMPL.format(today=date.today().isoformat(), tool_briefs=_tool_briefs(domain.tools))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"用户的话：{query}\n只输出 JSON："},
    ]


def _parse(msg: dict, valid: set[str]) -> tuple[str, dict] | None:
    """从模型输出里取 (tool, params)；工具不合法 / 不是 JSON → None。"""
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
            logger.error("audio LLM 调用失败：%s", exc)
            break
        parsed = _parse(msg, valid)
        if parsed is None:
            logger.warning("audio LLM 输出不合法（第 %d 次）：%r", attempt, str(msg.get("content"))[:200])
            continue
        tool, params = parsed
        # 协议层：fan 绑场景；query 是 audio_search 必填，模型漏填用原话补
        if tool == "fan_knowledge_agent":
            params = rebuild_fan_params(params, scene_hint="audio_qa", fallback_query=query)
        else:
            if tool == "audio_search" and not params.get("query"):
                params["query"] = query
        return Prediction(domain=domain.name, tool=tool, params=params, hit_source="llm")

    # 两层兜底：第一层 audio_search（含 query=原话透传）；第二层本域 fan（知识问答形态才落）。
    if _looks_like_qa(query):
        return Prediction(domain=domain.name, tool="fan_knowledge_agent",
                          params=domain_fan_params("audio", query), hit_source="fallback_qa")
    return Prediction(domain=domain.name, tool=FALLBACK_TOOL,
                      params={"action": "search", "query": query}, hit_source="fallback")