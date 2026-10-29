"""音乐（music）域：LLM 为中心，无规则层、无 fewshot、无 badcase。

决策全部在 `pipeline.py`：一次 LLM 调用直接出 {tool, params}；LLM 失败/输出不合法
→ music_song_search 兜底（retext=原话）。9 个工具全部扁平槽位、无嵌套 query DSL，
工具之间无包含关系，由 LLM 直接选。
"""

from __future__ import annotations

import logging
from datetime import date

from app.llm import _parse_json_block, chat
from app.metadata import PURPOSE_ONECALL, build_llm_metadata
from app.models import Prediction, PredictRequest

logger = logging.getLogger("hcTools.domains.music")

FALLBACK_TOOL = "music_song_search"
MAX_ATTEMPTS = 2

# 9 个工具的完整定义（来自原 domains/music/schema.json，全量内联，不再依赖 schema.json）。
_TOOL_SPECS = [
    {
        'name': 'music_song_search',
        'description': '酷狗音乐歌曲搜索工具(music_song_search)。当用户表达搜索歌曲、找歌、点歌、听某首歌的意图时触发。\\n检索模式：扁平结构化槽位检索，提取到对应信息则携带对应槽位，未提取到则不包含该key。除 retext 外所有业务槽位均为数组多值：同一槽位内多个取值为维度内 AND，不同槽位之间亦为 AND；OR 意图(如\\"周杰伦或者林俊杰的歌\\")或 NOT 意图(如\\"除了周杰伦以外的歌\\")酷狗不支持。\\n固定候选字段：version/gender/tag/toplist 取值必须归一化到枚举，无法归一时不传该维度。 \\n不适用场景——出现以下任一情况不要调用本工具：1) 用户意图不是听歌搜歌（如想看MV、看音乐视频）；2) OR 意图(如\\"周杰伦或者林俊杰的歌\\")或 NOT 意图(如\\"除了周杰伦以外的歌\\")，酷狗仅支持 AND；3) 用户强调地区/语言/年代等本工具未接入的维度。4）当用户的问句中无有效槽位 5) 用户问句中包含合作场景 \\n禁用场景：用户想\\"唱/K歌/找伴奏点唱\\"时走 music_ksong_search，禁止调用本工具。',
        'required': ['retext'],
    },
    {
        'name': 'music_song_semantic_search',
        'description': '酷狗音乐语义歌曲搜索工具(music_song_semantic_search)。当用户表达想听歌曲，但**提取不出任何有效业务槽位**，需要走慢链路改写时，调用该工具，传入结构仅有retext，典型例子:今天加班到很晚放点提神的歌、今天要干一天的活放点适合的歌、适合今天天气听的歌。当用户询问xxx和xxx合作的歌曲这种带有”合作“词语的语句调用该工具，典型例子:周杰伦和方文山合作的歌曲。 \\n不适用场景——出现以下任一情况不要调用本工具：1) 用户意图不是听歌搜歌（如想看MV、看音乐视频）；2) 用户的问句中包含明确的歌曲名、歌手名、专辑名等信息时，不走该工具 \\n禁用场景：用户想\\"唱/K歌/找伴奏点唱\\"时走 music_ksong_search，禁止调用本工具。',
        'required': ['retext'],
    },
    {
        'name': 'music_song_mv_search',
        'description': '酷狗音乐MV搜索工具(music_song_mv_search)。当用户表达搜索MV、找MV、看某首歌的MV的意图时触发。\\n检索模式：扁平结构化槽位检索，提取到对应信息则携带对应槽位，未提取到则不包含该key。除 retext 外所有业务槽位均为数组多值：同一槽位内多个取值为维度内 AND，不同槽位之间亦为 AND；OR 意图(如\\"周杰伦或者林俊杰的歌\\")或 NOT 意图(如\\"除了周杰伦以外的歌\\")酷狗不支持。\\n固定候选字段：version/gender/tag/toplist 取值必须归一化到枚举，无法归一时不传该维度。\\n不适用场景——出现以下任一情况不要调用本工具：1) 提取不出任何明确单一槽位(如\\"心情不好了来点歌\\"等情绪/场景/氛围/节日描述)；2) OR 意图(如\\"周杰伦或者林俊杰的歌\\")或 NOT 意图(如\\"除了周杰伦以外的歌\\")，酷狗仅支持 AND；3) tag 取值不在酷狗枚举表内；4) 用户强调地区/语言/年代等本工具未接入的维度。\\n禁用场景：用户想\\"唱/K歌/找伴奏点唱\\"时走 music_ksong_search，禁止调用本工具',
        'required': ['retext'],
    },
    {
        'name': 'music_ksong_search',
        'description': 'K歌搜索工具(music_ksong_search)。当用户表达在 K歌/全民K歌/酷狗K歌 中搜索歌曲、找伴奏、点唱的意图时触发。\\n检索模式：与 music_song_search 保持一致的扁平结构化槽位检索。除 retext 外所有业务槽位均为数组多值：同一槽位内多个取值为维度内 AND，不同槽位之间亦为 AND；OR 意图(如\\"周杰伦或者林俊杰的歌\\")或 NOT 意图(如\\"除了周杰伦以外的歌\\")不支持。\\n空值规则：song/singer/album/lyricist/composer/ip 允许传空数组 [](抽取不到明确值时占位)；其余业务槽位(version/gender/tag/lyrics/toplist)须非空。无论哪些槽位为空，请求内至少要有一个业务槽位非空。\\n槽位分类：A类(song/singer/album/lyricist/composer/version/gender/ip，可任意 AND 组合)、B类 tag(半开放标签，数组多值，维度内 AND)、排他类 lyrics(歌词片段)/toplist(榜单，仅单独使用，不可与其他槽位组合，schema 已约束)。\\n固定候选字段：version/gender/toplist 取值必须归一化到枚举，无法归一时不传该维度。\\n不适用场景——出现以下任一情况不要调用本工具：1) 提取不出任何明确单一槽位(如\\"心情不好了来点歌\\"等情绪/场景/氛围/节日描述)；2) OR 意图或 NOT 意图，仅支持 AND；3) tag 取值不在酷狗枚举表内；4) 用户强调地区/语言/年代等本工具未接入的维度。\\n禁用场景：用户仅想\\"听/放/播放\\"歌曲(走 music_song_search)而非\\"唱/K歌/找伴奏点唱\\"时，禁止调用本工具。',
        'required': ['retext'],
    },
    {
        'name': 'music_song_qqmusic_search',
        'description': 'QQ音乐歌曲搜索工具(music_song_qqmusic_search)。当用户表达在 QQ 音乐中搜索/播放歌曲的意图时触发。必须包含：QQ音乐时，才能调用。\\n入参：仅需传入 keywords 数组，由 QQ 音乐后端拼接检索；song/singer/ip/toplist 不作为独立字段，统一并入 keywords。\\nkeywords 提取规则：仅从用户语料提取以下 4 类有效槽位的值，按词拆为数组元素——song(歌曲名，如“恐龙抗狼”)、ip(影视/游戏/动漫作品名，如“泰坦尼克号”)、singer(歌手名，如“周杰伦”)、toplist(榜单名，如“飙升榜”)。“QQ音乐”是平台名，非有效槽位，不入 keywords。去除“播放”、“来点”等指令性前后缀；情绪/场景/氛围等非 4 类有效槽位的描述词不要放入 keywords。\\n默认值：4 类有效槽位均提不出时，keywords 填默认值“热门”，不要传空数组。',
        'required': ['retext', 'keywords'],
    },
    {
        'name': 'music_song_recommend',
        'description': '音乐随意听工具(music_song_recommend)。当用户表达\\"随意听/随便听/随心听\\"这类无明确目标、由系统个性化推荐播歌的意图时触发。查询直达词：随心听。\\n处理逻辑：不提取任何检索槽位，直接进入个性化推荐播歌链路(推荐结果由酷狗侧按用户画像返回)。\\n触发示例：\\"随心听\\"、\\"随便放点歌\\"、\\"随意听\\"。\\n边界与禁用场景：1. 用户带任何明确检索条件(歌手/歌曲/专辑/榜单/标签等，如\\"放周杰伦的歌\\")时，走 music_song_search，禁止调用本工具；2. 情绪/场景/氛围等模糊描述找歌(如\\"心情不好了来点歌\\")，禁止调用本工具；3. 想唱/K歌走 music_ksong_search。',
        'required': ['retext'],
    },
    {
        'name': 'music_song_history',
        'description': '音乐历史记录工具(music_song_history)。当用户表达查看/播放\\"音乐历史记录/播放历史/最近播放\\"这类基于历史播放数据的意图时触发。\\n触发方式：1. 中控直达配置；2. 模型意图用于泛化\\"音乐历史记录\\"相关说法，处理逻辑统一跳转到音乐历史记录。\\n处理逻辑(按终端能力分流)：如果终端支持起播，直接跳转播放页，从历史记录拿数据播放；否则跳转历史记录页面并亮屏。\\n触发示例：\\"播放历史记录\\"、\\"放我最近听的歌\\"、\\"打开音乐播放历史\\"。\\n边界与禁用场景：1. 用户要找的是\\"收藏\\"内容(如\\"播放我收藏的歌曲\\")时，走 music_song_favorite_search，禁止调用本工具；2. 带明确检索条件找新歌(如\\"放周杰伦的晴天\\")走 music_song_search；3. 想唱/K歌走 music_ksong_search。',
        'required': ['retext'],
    },
    {
        'name': 'tvchannel_music_search',
        'description': '频道搜索音乐工具(tvchannel_music_search)。当用户表达\\"在某个频道/电视台/频道号上搜索或播放音乐\\"这类频道+音乐的组合意图时触发，即在指定电视频道语境下找歌/放歌的请求。\\n处理逻辑：固定回复不支持。\\n触发示例：\\"CCTV15放首歌\\"、\\"在音乐频道放周杰伦的歌\\"、\\"用5频道放音乐\\"。\\n边界与禁用场景：1. 不带频道限定、单纯找歌/放歌(如\\"放周杰伦的歌\\")走 music_song_search，禁止调用本工具；2. 想唱/K歌走 music_ksong_search。',
        'required': ['retext'],
    },
    {
        'name': 'music_song_favorite_search',
        'description': '音乐收藏搜索工具(music_song_favorite_search)。当用户表达播放/搜索\\"自己收藏的音乐内容\\"的意图时触发，覆盖三类收藏对象：1. 收藏的歌曲(如\\"我想听我收藏的歌曲\\")；2. 收藏的歌单(如\\"播放我收藏的歌单\\"、\\"播放收藏歌单XXXXXX\\")；3. 收藏的专辑(如\\"播放我收藏的专辑\\")。\\n处理逻辑：如果终端支持起播，直接跳转播放页，从收藏记录拿数据播放；否则跳转收藏记录页面并亮屏。\\n触发示例：\\"我想听我收藏的歌曲\\"、\\"播放我收藏的歌单\\"、\\"播收藏的专辑\\"。\\n边界与禁用场景：1. 用户要找的是\\"历史记录/最近播放\\"(如\\"放我最近听的歌\\")时，走 music_song_history，禁止调用本工具；2. 播放未收藏的普通歌曲/歌手/专辑(如\\"放周杰伦的晴天\\")走 music_song_search；3. \\"随意听/随心听\\"这类无目标推荐播歌走 music_song_recommend。',
        'required': ['retext'],
    },
    {
        'name': 'fan_knowledge_agent',
        'description': '音乐知识问答与闲聊（fan_knowledge_agent）。关于歌曲/歌手/音乐的任何疑问：歌手属性问答（如"周杰伦哪年出道""这首歌什么风格""孙燕姿的处女座是什么"）、音乐行业开放问题、冷知识、闲聊，都走音乐知识问答链路（agent_music 场景，下游再分 music_chat_qa / music_info_qa）。不做歌曲媒资检索（那是 music_song_search）。\\n参数：{"messages":"用户原话整体传入（不要改写）"}。',
        'required': ['messages'],
    },
]

_TOOL_REQUIRED = {s["name"]: list(s["required"]) for s in _TOOL_SPECS}
_RETEXT_TOOLS = set(_TOOL_REQUIRED)

# 第二层兜底：本域知识问答（agent_music 场景）。
from app.fanparams import (  # noqa: E402
    domain_fan_params,
    looks_like_qa as _looks_like_qa,
    rebuild_fan_params,
)

_FIELD_CATALOG = """- song 歌名
- singer 歌手
- album 专辑名
- version 版本（原唱/翻唱/伴奏/纯音乐）
- gender 歌手性别
- ip 影视/动漫 IP 名
- tag 标签（摇滚/民谣/经典老歌/儿歌…）
- lyrics 歌词片段（用户说了某句歌词时填这里）
- toplist 榜单名（热歌榜/新歌榜…）
- lyricist 作词人
- composer 作曲人
- keywords QQ音乐关键词（仅 music_song_qqmusic_search 用）"""

_SYSTEM_TMPL = """你是音乐语音助手的意图解析器。读用户的话，输出该调用的工具名和参数。

【工具】
{tool_briefs}

【怎么选工具】
按下面的顺序判断，**先命中先返回**：
1. 说了 **K歌/唱歌/想唱** → music_ksong_search。
2. 说了 **MV/音乐视频/这首歌的视频** → music_song_mv_search。
3. 说了 **QQ音乐** → music_song_qqmusic_search（keywords 里放关键词）。
4. 说了 **电视频道/卫视/CCTV/某台** → tvchannel_music_search。
5. 要 **回看听歌记录**（听过/历史/播放记录）→ music_song_history。
6. 要 **看收藏**（收藏/我喜欢）→ music_song_favorite_search。
7. **说清了歌名/歌手**（或专辑/作词/作曲/榜单）→ music_song_search。
8. **说不清是什么歌、只能用描述找**（「那个什么什么歌」「节奏很欢快的」「适合婚礼放的歌」）
   → music_song_semantic_search。
9. 单纯要**推歌**没给具体歌名（随便听/来点歌/换一批）→ music_song_recommend。
10. 其余**音乐知识问答/歌手背景/闲聊**（「周杰伦哪年出道」「这首歌的风格」「XX的歌还有谁翻唱过」）→ fan_knowledge_agent。

【参数怎么写】
所有工具的 params 都必须有 `retext` = **用户原话**（原样填，不要自己改词、不要翻译）。
再按用户话里提到的维度补槽位：
- 单值槽位：`"song": ["歌名"]`、`"singer": ["歌手"]` —— **值一律用数组**。
- `lyrics` 用于用户说了某句歌词；`toplist` 用于榜单名；`keywords` 仅 QQ音乐用。
- 用户**没提到**的槽位不要出现（不要填 null / 空串 / 空数组）。
- 音乐知识问答/闲聊 → fan_knowledge_agent，params 填 {{"messages":"用户原话整体传入（不要改写）"}}。

可用槽位：
{fields}

【输出】
只输出一个 JSON 对象，形如 {{"tool":"工具名","params":{{...}}}}，不要任何解释。
"""


def _tool_briefs(tools) -> str:
    lines = []
    for t in tools:
        desc = " ".join(t.description.replace("\\n", " ").split())
        lines.append(f"- {t.name}：{desc[:900]}")
    return "\n".join(lines)


def build_messages(domain, query: str) -> list[dict]:
    system = _SYSTEM_TMPL.format(
        today=date.today().isoformat(),
        tool_briefs=_tool_briefs(domain.tools),
        fields=_FIELD_CATALOG,
    )
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
            logger.error("music LLM 调用失败：%s", exc)
            break
        parsed = _parse(msg, valid)
        if parsed is None:
            logger.warning("music LLM 输出不合法（第 %d 次）：%r", attempt, str(msg.get("content"))[:200])
            continue
        tool, params = parsed
        # 协议层：retext 必填，模型漏填用原话补
        if tool in _RETEXT_TOOLS and "retext" not in params:
            params["retext"] = query
        # 本域知识问答：LLM 主动选中 fan_knowledge_agent → 重建 fan 契约（agent_music 场景）
        if tool == "fan_knowledge_agent":
            params = rebuild_fan_params(params, scene_hint="agent_music", fallback_query=query)
        return Prediction(domain=domain.name, tool=tool, params=params, hit_source="llm")

    # 两层兜底：fuzzy 第一层；query 呈知识问答形态 → 本域 fan（agent_music 场景）。
    if _looks_like_qa(query):
        return Prediction(domain=domain.name, tool="fan_knowledge_agent",
                          params=domain_fan_params("music", query), hit_source="fallback_qa")
    return Prediction(domain=domain.name, tool=FALLBACK_TOOL, params={"retext": query},
                      hit_source="fallback")