"""少儿（children）域：LLM 为中心，无规则层、无 fewshot、无 badcase。

决策全部在 `pipeline.py`：一次 LLM 调用直接出 {tool, params}；LLM 失败/输出不合法
→ educ_fuzzy_search 整句兜底。没有 badcase / 精确缓存 / 规则 / 后处理改写。
"""

from __future__ import annotations

import logging
from datetime import date

from app.llm import _parse_json_block, chat
from app.metadata import PURPOSE_ONECALL, build_llm_metadata
from app.models import Prediction, PredictRequest

logger = logging.getLogger("hcTools.domains.children")

FALLBACK_TOOL = "educ_fuzzy_search"
MAX_ATTEMPTS = 2

# 4 个工具的完整定义（来自原 domains/children/schema.json，全量内联，不再依赖 schema.json）。
_TOOL_SPECS = [
    {
        'name': 'educ_search_all',
        'description': '少儿媒资搜索工具(educ_search_all)。当用户表达了搜索、查找少儿内容的意图时触发，包括按标题(title)、媒资类型(content_type)、蛛网标签二级分类(children_second_genre)/三级分类(children_third_genre)、少儿培养目标(training_objectives)、角色(role)、少儿八大智能(multiple_intelligences)、国家(country)、出品公司(company)、语言(language)、性别(gender)、节日(festival)、奖项(prize)、音效(features)等维度筛选，以及按免付费状态(is_fee)过滤和发布时间范围(release_time, 格式yyyyMMdd)筛选的场景。支持13个精确匹配维度、1个状态维度(is_fee 0|1)和2个范围维度(age_range 年龄范围、release_time 发布时间区间)以及3个媒资维度(series部数或季数、video_index集数或期数、voice_start_pos起播位置，单位为秒)。\\n检索模式：DSL 组合检索（形态 A）。query 为递归布尔树，不同槽位间可 and/or/not 任意嵌套；同一槽位多值用 values + operator。\\n不适用场景：相关推荐、查找相似的媒资改用educ_relate_search；查询用户的历史观看记录改用educ_history；描述性等方式表达影视查找意图的场景，包括剧情描述、台词描述、复杂组合条件等改用educ_fuzzy_search。\\n提参约束：同一字段的多个候选值优先用叶子节点的 values + operator=or，不要拆成多个同字段的 or 子节点。排除语义用外层 not 包裹，不要在 value 里写「非」「除了」。\\n时间解析：涉及到的时间范围为范围维度中的发布时间范围release_time和voice_start_pos起播位置，其中release_time为精确到日的yyyyMMdd 格式字符串，voiceStartPos为代表起播位置的整数，单位为秒，如第 10 分钟传 600，3 分 09 秒传 189。',
        'required': ['query', 'action', 'retext'],
    },
    {
        'name': 'educ_fuzzy_search',
        'description': '少儿媒资模糊搜索工具。用于处理用户以模糊、描述性等方式表达少儿查找意图的场景，包括剧情描述、台词描述、复杂组合条件等，系统将自然语言解析为结构化筛选参数后进行检索。\\n适用场景：1. 剧情/画面描述类查询，如\\"骑着蓝色大鸟的人的动画片、\\"主角在沙漠里找水的动画\\"；2. 复杂组合描述，如\\"那个胖子导演的高分动画；3. 台词/对白相关的模糊查找。 \\n不适用场景：相关媒资推荐、查找相似的媒资改用educ_relate_search；查询用户的历史观看记录改用educ_history；语料包含明显清晰的搜索槽位使用educ_search或educ_search_all。',
        'required': ['retext'],
    },
    {
        'name': 'educ_relate_search',
        'description': '少儿相关媒资搜索。当基于用户描述的参考媒资进行相似少儿内容推荐。\\n检索模式：DSL 组合检索（形态 A）。query 为递归布尔树，不同槽位间可 and/or/not 任意嵌套；同一槽位多值用 values + operator。\\n适用场景：用户表达了\\"基于某内容找类似的\\"、\\"推荐相关的\\"、\\"找同类型的\\"、\\"还有没有像XX这样的\\"等相似性推荐意图时使用，如\\"我想看类似小猪佩奇的动画片\\"、\\"推荐和汪汪队差不多的少儿节目\\"、\\"有没有熊出没那种动画\\"。\\n不适用场景：查询用户的历史观看记录改用educ_history；描述性等方式表达影视查找意图的场景，包括剧情描述、台词描述、复杂组合条件等改用educ_fuzzy_search；语料包含明显清晰的搜索槽位使用educ_search或educ_search_all。。\\n提参约束：同一字段的多个候选值优先用叶子节点的 values + operator=or，不要拆成多个同字段的 or 子节点。排除语义用外层 not 包裹，不要在 value 里写「非」「除了」。',
        'required': ['query', 'action', 'retext'],
    },
    {
        'name': 'educ_history',
        'description': '少儿历史记录工具(educ_history)。查询用户的少儿历史观看记录，支持按时间范围（精确到秒）筛选，返回符合条件的历史记录。\\n适用场景：用户表达了查看少儿历史记录的意图时使用，如\\"我看过的动画\\"、\\"上周看过什么\\"、\\"最近一个月的观看记录\\"、\\"2024年3月给宝宝放过哪些内容\\"、\\"昨天晚上给宝宝看的动画片\\"。\\n不适用场景：用户语料包含明显清晰的搜索槽位切非相似推荐场景使用educ_search或educ_search_all；描述性等方式表达影视查找意图的场景，包括剧情描述、台词描述、复杂组合条件等改用educ_fuzzy_search；当用户基于请求查询与某媒资类型相似、相关内容时使用educ_relate_recommend。\\n时间解析：time 按 Asia/Shanghai（北京时间）解析，格式为“yyyy-MM-dd HH:mm:ss TO yyyy-MM-dd HH:mm:ss”，起止时间均包含。月：支持本月、上个月、上上个月，范围为对应月份第一天00:00:00至最后一天23:59:59。周：一周从周一开始，支持本周、这周、上周、上上周，以及本周一至本周日、上周一至上周日，范围为对应日期或周的起止时间。日：支持今天、昨天、前天、大前天、本月第一天、本月最后一天、上个月第一天、上个月最后一天。未指定时段时，日期范围为当天00:00:00至23:59:59。时段：凌晨为01:00:00至04:59:59；早、早上、清晨为05:00:00至07:59:59；上午为08:00:00至10:59:59；中、中午、正午为11:00:00至12:59:59；下午为13:00:00至16:59:59；傍晚、黄昏为17:00:00至18:59:59；晚上为19:00:00至22:59:59；深夜、子夜为当日23:00:00至次日00:59:59。只出现时段词但未指定日期时，按当天解析。\\n结果使用：time为解析出的单个连续精确时间范围，根据该条件查询并返回用户的少儿历史观看记录；未传字段表示不做该维度限制。',
        'required': [],
    },
    {
        'name': 'fan_knowledge_agent',
        'description': '少儿知识问答与深度对答（fan_knowledge_agent）。孩子/家长关于少儿内容/角色的任何疑问：知识讲解、百科问答、角色/作品属性问答（如"小猪佩奇的弟弟是谁""这部动画片的主角是谁"）、开放问答与聊闲天，都走少儿知识问答链路（child 场景）。不做结构化媒资检索（那是 educ_search_all/educ_fuzzy_search）。\\n参数：{"messages":"用户原话整体传入（不要改写）"}。',
        'required': ['messages'],
    },
]

# 需要 retext 补全的工具（educ_relate_search / educ_history 不带，见 golden 实证）
_RETEXT_TOOLS = {
    "educ_search_all": {"query", "action", "retext"},
    "educ_fuzzy_search": {"retext"},
}
_TOOL_REQUIRED = {s["name"]: list(s["required"]) for s in _TOOL_SPECS}

# 第二层兜底 + LLM 主动命中：本域知识问答（child 场景）。fan 参数由 fanparams 重建契约。
from app.fanparams import (  # noqa: E402
    domain_fan_params,
    looks_like_qa as _looks_like_qa,
    rebuild_fan_params,
)

_FIELD_CATALOG = """- title 作品名（动画/动漫/绘本/儿歌/节目名等）
- content_type 内容形态（动画/动漫/绘本/儿歌/故事/科普/真人…）
- children_second_genre 二级题材（如 经典/益智/搞笑/启蒙）
- children_third_genre 三级细分题材（如 情绪管理/社交智能/科普海洋）
- training_objectives 培养目标（如 情绪管理/语言表达/逻辑思维）
- multiple_intelligences 多元智能维度
- role 角色名（动画里的人物，不是演员）
- is_fee 免付费（"0"=免费，"1"=付费/VIP）
- age_range 适龄范围（from/to 为年龄数字）
- language 语言（如 国语版/英文版/日语版）
- gender 性别倾向
- festival 节日主题（如 春节/圣诞）
- company 出品公司
- country 国家/地区
- release_time 发行年份范围（from/to 用 yyyyMMdd）
- series 第几部或第几季（数字字符串，从 1 开始）
- video_index 第几集或第几期（数字字符串，从 1 开始）
- voice_start_pos 起播位置（秒）"""

_SYSTEM_TMPL = """你是电视语音助手的少儿内容意图解析器。读用户的话，输出该调用的工具名和参数。

【工具】
{tool_briefs}

【怎么选工具】
按下面的顺序判断，**先命中先返回**：
1. 明确要 **回看观看记录**（我看过、我的历史、播放记录）→ educ_history。
2. 基于某个片名/角色要 **相似的**（类似、相似、像…一样、同类型、还有没有）→ educ_relate_search。
3. 只能整句语义理解的 **描述性表达**（封词、剧情描述、风格描述、「哪部/哪个」、
   成长主题、厉害在哪里…）→ educ_fuzzy_search。
4. 剩下的都是「找内容」：
   - 能抽出 **结构化筛选维度**（片名/内容形态/题材/角色/年龄/语言/免费/第几集…）
     → educ_search_all。它已涵盖全部结构化检索能力。
   - 抽不出结构化维度 → educ_fuzzy_search。
5. 开放性问题/知识问答（“XX是什么”“为什么”“怎么算”“百科”…）→ fan_knowledge_agent。

【搜索参数怎么写】educ_search_all 的 params 形如
{{"action":"search"或"play", "retext":"用户原话", "query":<条件树>, "sort":<排序>}}
- action：搜索/查找/浏览 → search；播放/打开/看第几页 → play。
- retext：用户原话（**不必自己清理标点**）。
- sort：只在用户明确表达排序时才加。
- educ_fuzzy_search / educ_relate_search / educ_history：params 填 {{"retext": "用户原话"}}（整句）。
- fan_knowledge_agent：params 填 {{"messages":"用户原话整体传入（不要改写）"}}。

【query 条件树】
可用字段（**只写 field 名，取值按哪张**）：
{fields}

节点形态（任意嵌套）：
- 单值：{{"field":"content_type","value":"动画"}}
- 多值：{{"field":"title","values":["宝宝巴士","小猪佩奇"],"operator":"or"}}
- 范围：{{"field":"age_range","from":"0","to":"3"}}（无界端写 "*"，一律字符串；release_time 用 yyyyMMdd）
- 组合：{{"and":[...]}} / {{"or":[...]}} / {{"not":{{...}}}}
- 条件只有一个时，query 直接写那个叶子，不要包 and。
- 字段无法归一化到常用值时，不要自造新值。

【输出】
只输出一个 JSON 对象，形如 {{"tool":"工具名","params":{{...}}}}，不要任何解释。
只抽用户明确表达的信息，不臆造；用户没提的可选字段不要出现（不要填 null / 空串 / 空数组）。
"""


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


# 评测集人工标注 与 历史 SFT/DPO 数据大量使用 "educ_search"（349 行）与
# "educ_relate_recommend"（14 行）表达检索/相关推荐；运行时 schema 只注册了
# educ_search_all / educ_relate_search，这里把模型习惯输出的别名归一为 schema 工具名，
# 避免误判「输出不合法」→ 兜底。与 vod 的 _TOOL_ALIAS 同款修复。
_TOOL_ALIAS = {
    "educ_search": "educ_search_all",
    "educ_relate_recommend": "educ_relate_search",
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
            logger.error("children LLM 调用失败：%s", exc)
            break
        parsed = _parse(msg, valid)
        if parsed is None:
            logger.warning("children LLM 输出不合法（第 %d 次）：%r", attempt, str(msg.get("content"))[:200])
            continue
        tool, params = parsed
        # 协议层：required 的 retext 若模型漏填，用原话补
        if tool in _RETEXT_TOOLS and "retext" not in params:
            params["retext"] = query
        # 本域知识问答：LLM 主动选中 fan_knowledge_agent → 重建 fan 契约（child 场景）
        if tool == "fan_knowledge_agent":
            params = rebuild_fan_params(params, scene_hint="child", fallback_query=query)
        return Prediction(domain=domain.name, tool=tool, params=params, hit_source="llm")

    # 两层兜底：fuzzy 第一层；query 呈知识问答形态 → 本域 fan 第二层（child 场景）。
    if _looks_like_qa(query):
        return Prediction(domain=domain.name, tool="fan_knowledge_agent",
                          params=domain_fan_params("children", query), hit_source="fallback_qa")
    return Prediction(domain=domain.name, tool=FALLBACK_TOOL, params={"retext": query},
                      hit_source="fallback")