"""vod 域决策主轴：LLM 为中心，单次调用直接出 {tool, params}。

流程（无规则层、无 fewshot）：
    system prompt（工具说明 + 参数写法）+ 用户原话 → 一次 LLM 调用 → 解析 JSON → 返回
    LLM 两次都失败（网关异常 / 输出不合法）→ vod_fuzzy_search 整句兜底

确定性代码只做「协议层」的事：解析 JSON、校验工具名、补齐 schema 里 required 的 retext。
不做任何语义判断（不改工具、不改槽位、不补 sort）——这些全部交给模型（可训练）。
"""

from __future__ import annotations

import logging
from datetime import date

from app.llm import _parse_json_block, chat
from app.metadata import PURPOSE_ONECALL, build_llm_metadata
from app.models import Prediction, PredictRequest

logger = logging.getLogger("hcTools.domains.vod")

# 第二层兜底：本域知识问答（video 场景）。
from app.fanparams import (  # noqa: E402
    domain_fan_params,
    looks_like_qa as _looks_like_qa,
    rebuild_fan_params,
)

FALLBACK_TOOL = "vod_fuzzy_search"
MAX_ATTEMPTS = 2

# 6 个工具的完整定义（从 domains/vod/schema.json 全量内联，不再依赖 schema.json）。
# 描述与 schema 原文逐字一致；required 只保留协议层 retext 补填所需字段名。
_TOOL_SPECS = [
    {
        'name': 'vod_search_all',
        'description': '影视媒资组合式搜索与起播工具(vod_search_all)。当用户表达搜索、查找或播放影视内容的意图时触发，包括按标题、演员、导演、分类、地区、语言、标签等维度筛选，以及按评分、热度、发布时间和播放量排序的场景。请求必须通过action指定操作类型：search表示搜索，play表示起播。支持24个精确匹配维度(title/actor/director/area/category/channel等)、2个状态维度(is_fee：0表示免费、1表示付费；is_over：0表示连载中、1表示已完结)、3个范围维度(age_range年龄范围、release_year发布时间范围，格式yyyyMMdd；rate评分范围0～10)，以及3个媒资维度(series部数或季数、video_index集数或期数、voice_start_pos起播位置，单位为秒)。\\n检索模式：DSL 组合检索（形态 A）。query 为递归布尔树，不同槽位间可 and/or/not 任意嵌套；同一槽位多值用 values + operator。\\n适用场景：用户进行查询或或者播放内容，如「我想看刘德华的英文电影」「搜一下最新的好莱坞电影」「最热门的韩剧」。\\n不适用场景：相关推荐、查找相似的媒资改用vod_relate_search；根据用户偏好进行媒资推荐改用vod_personalized_search；查询用户的历史观看记录改用vod_history；描述性等方式表达影视查找意图的场景，包括剧情描述、台词描述、复杂组合条件等改用vod_fuzzy_search。\\n提参约束：同一字段的多个候选值优先用叶子节点的 values + operator=or，不要拆成多个同字段的 or 子节点。排除语义用外层 not 包裹，不要在 value 里写「非」「除了」。\\n时间解析：涉及到的时间范围为范围维度中的发布时间范围release_time和voice_start_pos起播位置，其中release_time为精确到日的yyyyMMdd 格式字符串，voiceStartPos为代表起播位置的整数，单位为秒，如第 10 分钟传 600，3 分 09 秒传 189。',
        'required': ['query', 'action', 'retext'],
    },
    {
        'name': 'vod_fuzzy_search',
        'description': '影视媒资模糊搜索工具。用于处理用户以模糊、描述性等方式表达影视查找意图的场景，包括剧情描述、台词描述、复杂组合条件等，系统将自然语言解析为结构化筛选参数后进行检索。\\n适用场景：1. 剧情/画面描述类查询，如\\"骑着蓝色大鸟的人的电影\\"、\\"主角在沙漠里找水的片子\\"；2. 复杂组合描述，如\\"那个演功夫片的胖子导演的高分电影\\"；3. 台词/对白相关的模糊查找。 \\n不适用场景：相关媒资推荐、查找相似的媒资改用vod_relate_search；根据用户偏好进行媒资推荐改用vod_personalized_search；查询用户的历史观看记录改用vod_history；语料包含明显清晰的搜索槽位使用vod_search或vod_search_all。\n\n注意：增加媒资的总集次和更新集次信息',
        'required': ['retext'],
    },
    {
        'name': 'vod_relate_search',
        'description': '影视媒资相关检索工具(vod_relate_search)。当用户基于某个影视标题、演员、导演或分类请求相似、相关内容时触发。支持4个精确匹配维度(title、actor、director、category)。\\n检索模式：DSL 组合检索（形态 A）。query 为递归布尔树，不同槽位间可 and/or/not 任意嵌套；同一槽位多值用 values + operator。\\n适用场景：用户进行相似媒资查询，如「类似流浪地球的电影」「推荐一些类似刘德华演的赌神的影片」。\\n不适用场景：用户语料包含明显清晰的搜索槽位切非相似推荐场景使用vod_search或vod_search_all；根据用户偏好进行媒资推荐改用vod_personalized_search；查询用户的历史观看记录改用vod_history；描述性等方式表达影视查找意图的场景，包括剧情描述、台词描述、复杂组合条件等改用vod_fuzzy_search。\\n提参约束：同一字段的多个候选值优先用叶子节点的 values + operator=or，不要拆成多个同字段的 or 子节点。排除语义用外层 not 包裹，不要在 value 里写「非」「除了」。本工具仅用于相关推荐，不负责内容起播。',
        'required': ['query'],
    },
    {
        'name': 'vod_personalized_search',
        'description': '基于用户历史偏好进行个性化影视内容推荐，根据用户的观看记录、兴趣标签等画像信息推荐用户可能感兴趣的媒资内容。\\n适用场景：用户表达了个性化推荐意图时使用，如\\"推荐我喜欢的电影\\"、\\"给我推荐点好看的\\"、\\"推荐我喜欢的视频\\"、\\"有什么好看的推荐一下\\"。可带分类参数如\\"推荐我喜欢的综艺\\"，也可不带分类如\\"推荐我喜欢的视频\\"。\\n不适用场景：用户语料包含明显清晰的搜索槽位切非相似推荐场景使用vod_search或vod_search_all；相关媒资推荐、查找相似的媒资改用vod_relate_search；查询用户的历史观看记录改用vod_history；描述性等方式表达影视查找意图的场景，包括剧情描述、台词描述、复杂组合条件等改用vod_fuzzy_search。\\n结果使用：根据解析出的分类偏好从用户画像中匹配推荐内容；当未传category时基于用户整体兴趣进行推荐。',
        'required': [],
    },
    {
        'name': 'vod_history',
        'description': '影视历史记录工具(vod_history)。查询用户的历史观看记录，支持按影视分类和精确时间范围筛选，返回符合条件的历史记录。\\n适用场景：用户表达了查看历史记录的意图时使用，如“我昨天看过的电影”、“最近看了什么”、“前天看过的综艺”、“我的观看历史”、“查一下我的播放记录”。\\n不适用场景：用户语料包含明显清晰的搜索槽位切非相似推荐场景使用vod_search或vod_search_all；根据用户偏好进行媒资推荐改用vod_personalized_search；描述性等方式表达影视查找意图的场景，包括剧情描述、台词描述、复杂组合条件等改用vod_fuzzy_search；当用户基于某个影视标题、演员、导演或分类请求相似、相关内容时使用vod_relate_recommend。\\n时间解析：time 按 Asia/Shanghai（北京时间）解析，格式为“yyyy-MM-dd HH:mm:ss TO yyyy-MM-dd HH:mm:ss”，起止时间均包含。月：支持本月、上个月、上上个月，范围为对应月份第一天00:00:00至最后一天23:59:59。周：一周从周一开始，支持本周、这周、上周、上上周，以及本周一至本周日、上周一至上周日，范围为对应日期或周的起止时间。日：支持今天、昨天、前天、大前天、本月第一天、本月最后一天、上个月第一天、上个月最后一天。未指定时段时，日期范围为当天00:00:00至23:59:59。时段：凌晨为01:00:00至04:59:59；早、早上、清晨为05:00:00至07:59:59；上午为08:00:00至10:59:59；中、中午、正午为11:00:00至12:59:59；下午为13:00:00至16:59:59；傍晚、黄昏为17:00:00至18:59:59；晚上为19:00:00至22:59:59；深夜、子夜为当日23:00:00至次日00:59:59。只出现时段词但未指定日期时，按当天解析。\\n结果使用：category为系统解析的分类筛选条件，time为解析出的单个连续精确时间范围；未传字段表示不做该维度限制。',
        'required': [],
    },
    {
        'name': 'vod_person_search',
        'description': '影视人物查询(vod_person_search)。\\n适用场景：用户知道某个明星，通过明星名称进行搜索，但没用具体的问题，如“刘德华是谁”、“导演饺子”、“安妮海瑟薇的个人简介”、“杨钰莹的介绍”。\\n不适用场景：用户语料包含明显清晰的搜索槽位切非相似推荐场景使用vod_search或vod_search_all；根据用户偏好进行媒资推荐改用vod_personalized_search；描述性等方式表达影视查找意图的场景，包括剧情描述、台词描述、复杂组合条件等改用vod_fuzzy_search；当用户基于某个影视标题、演员、导演或分类请求相似、相关内容时使用vod_relate_recommend；查询用户的历史观看记录改用vod_history。\\n结果使用：若单个人物,返回单人物卡片，若多个人物,返回多人物卡片 。',
        'required': [],
    },
    {
        'name': 'fan_knowledge_agent',
        'description': '影视领域知识问答（fan_knowledge_agent）。影视内容的任何疑问都走这里：作品/剧情/人物/导演/演员/奖项的属性问答（如「天龙八部的导演是谁」「刘德华演过哪些电影」「这部电影为什么叫这个名字」）、开放性问题解答（为什么/怎么拍/剧情原理/背后故事/冷知识）。不做媒资起播与维度检索（那是 vod_search_all）。\n参数：{"messages":"用户原话整体传入（不要改写）"}。',
        'required': ['messages'],
    },
]

# required 之外的参数校验完全交给 LLM prompt；这里只做协议层的 retext 补填。
_TOOL_REQUIRED = {s["name"]: list(s["required"]) for s in _TOOL_SPECS}

_SYSTEM_TMPL = """你是电视语音助手的影视意图解析器。读用户的话，输出该调用的工具名和参数。
今天是 {today}。

【工具】
{tool_briefs}

【各工具参数】
- vod_search_all：{{"action":"search"|"play", "retext":"用户原话", "query":<条件树>, "sort":<排序，可选>}}
  - action：搜索/查找/浏览 → search；播放/打开/看第几集 → play。
  - sort 形如 {{"维度":{{"order":"desc"|"asc"}}}}，维度：new（最新）、hot（热度）、rate（评分）、
    play（播放量）；只在用户表达了排序诉求时给，可多个并存。
- vod_fuzzy_search：{{"retext":"用户原话"}}
- vod_relate_search：{{"query":<条件树>}}，字段只用 title / actor / director / category。
- vod_personalized_search：{{"category":"电影"|"电视剧"|"纪录片"|"综艺"|"戏曲"|"幼儿"|"动漫"}}，用户没说分类则 {{}}。
- vod_history：{{"category":["电视剧",...], "time":"yyyy-MM-dd HH:mm:ss TO yyyy-MM-dd HH:mm:ss"}}，
  两个字段都可省；time 按今天日期推算（如「昨天」= 昨天 00:00:00 TO 23:59:59）。
- vod_person_search：{{"person_name":["人名", ...]}}
- 通用知识问答 fan_knowledge_agent：{{"messages":"用户原话整体传入（不要改写）"}}

【关键分流——属性提问也是知识问答】
- 影视内容/人物/作品的任何疑问（某部作品的导演/演员是谁、「为什么」「怎么拍」「讲什么原理」）都归 **fan_knowledge_agent**（messages 传用户原话整体，走影视知识问答）。
- 「刘德华是谁」这类人物本身 → fan_knowledge_agent 或 vod_person_search（检索人物卡片）均可。
- 只有用户明确表达要『找/看』某内容（起播、维度检索）才走 vod_search_all 等媒资工具。

【query 条件树】
可用字段：
{fields}

节点形态（可任意嵌套）：
- 单值：{{"field":"category","value":"电影"}}
- 多值：{{"field":"tag","values":["悬疑","动作"],"operator":"or"}}
- 范围：{{"field":"rate","from":"8","to":"*"}}（无界端填 "*"；release_time 用 yyyyMMdd）
- 组合：{{"and":[...]}} / {{"or":[...]}} / {{"not":{{...}}}}
- 条件只有一个时直接写那个叶子，不要包 and。

【输出】
只输出一个 JSON 对象：{{"tool":"工具名","params":{{...}}}}，不要任何解释。
只抽用户明确表达的信息，不臆造；用户没提的可选字段不要出现。
"""

_FIELDS = """- title 作品名
- actor 演员
- director 导演、编剧
- entertainer 参与人员、艺人表演者（含歌手）
- role 角色名
- dubbing 配音演员
- hostess 主持人
- writer 作者
- prize 影视奖项
- sub_prize 子奖项
- category 影视分类（电影/电视剧/综艺/纪录片/动漫/戏曲…）
- tag 标签、题材
- area 地区
- language 语言
- channel 频道、电视台
- company 出品公司
- vender_name 供应商
- comedy_brand 喜剧厂牌
- creation_source 剧本来源
- target 受众
- gender 性别
- definition 清晰度（is_4k / is_3d / hdr）
- sound 音效
- technology 拍摄技术
- is_fee 免付费（"0"=免费，"1"=付费）
- is_over 状态（"0"=连载中，"1"=已完结）
- age_range 年龄范围
- release_time 发布时间范围
- rate 评分范围（0~10）
- series 第几部/第几季（数字字符串）
- video_index 第几集/第几期（数字字符串）
- voice_start_pos 起播位置（秒）"""


def _tool_briefs(tools) -> str:
    lines = []
    for t in tools:
        desc = " ".join(t.description.replace("\\n", " ").split())
        lines.append(f"- {t.name}：{desc[:600]}")
    return "\n".join(lines)


def build_messages(domain, query: str) -> list[dict]:
    system = _SYSTEM_TMPL.format(
        today=date.today().isoformat(), tool_briefs=_tool_briefs(domain.tools), fields=_FIELDS
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"用户的话：{query}\n只输出 JSON："},
    ]


_TOOL_ALIAS = {
    # 评测集人工标注 与 历史 SFT/DPO 数据大量使用 "vod_search" 表达影视检索
    # （区分于组合 DSL 的 vod_search_all）；运行时 schema 只注册了 vod_search_all，
    # 这里把模型习惯输出的别名归一为 schema 工具名，避免误判「输出不合法」→ 兜底。
    "vod_search": "vod_search_all",
}


def _parse(msg: dict, valid: set[str]) -> tuple[str, dict] | None:
    """从模型输出里取 (tool, params)；工具名不合法 / 不是 JSON → None。"""
    obj = _parse_json_block(str(msg.get("content") or ""))
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool") or obj.get("tool_name") or ""
    tool = _TOOL_ALIAS.get(tool, tool)
    params = obj.get("params")
    if tool not in valid or not isinstance(params, (dict, type(None))):
        return None
    return tool, params or {}


def _required(domain, tool: str) -> set[str]:
    t = domain.tools_by_name.get(tool)
    req = (t.parameters.get("required") if t else None) or []
    return set(req) if isinstance(req, list) else set()


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
            logger.error("vod LLM 调用失败：%s", exc)
            break
        parsed = _parse(msg, valid)
        if parsed is None:
            logger.warning("vod LLM 输出不合法（第 %d 次）：%r", attempt, str(msg.get("content"))[:200])
            continue
        tool, params = parsed
        # 协议层：绑定 fan 场景；schema 里 required 的 retext 若模型漏填，用原话补
        if tool == "fan_knowledge_agent":
            params = rebuild_fan_params(params, scene_hint="video", fallback_query=query)
        else:
            if "retext" in _required(domain, tool) and not params.get("retext"):
                params["retext"] = query
        return Prediction(domain=domain.name, tool=tool, params=params, hit_source="llm")

    # 两层兜底：vod_fuzzy_search 第一层；query 呈知识问答形态 → 本域 fan（video 场景）。
    if _looks_like_qa(query):
        return Prediction(domain=domain.name, tool="fan_knowledge_agent",
                          params=domain_fan_params("vod", query), hit_source="fallback_qa")
    return Prediction(domain=domain.name, tool=FALLBACK_TOOL, params={"retext": query},
                      hit_source="fallback")
