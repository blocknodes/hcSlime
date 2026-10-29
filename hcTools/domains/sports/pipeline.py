"""体育（sports）域：LLM 为中心，无规则层、无 fewshot、无 badcase。

决策全部在 `pipeline.py`：一次 LLM 调用直接出 {tool, params}；LLM 失败/输出不合法
→ sports_match_search 兜底。参数里的相对日期（今天/明天/周X/上一轮）由确定性
`_resolve_time` 补全 —— 模型推算不了基准日，这是协议层而非语义决策。
"""

from __future__ import annotations

import logging
import re
from datetime import date

from app.llm import _parse_json_block, chat
from app.metadata import PURPOSE_ONECALL, build_llm_metadata
from app.models import Prediction, PredictRequest

logger = logging.getLogger("hcTools.domains.sports")

FALLBACK_TOOL = "sports_match_search"
MAX_ATTEMPTS = 2

# 9 个工具的完整定义（来自原 domains/sports/schema.json，全量内联，不再依赖 schema.json）。
_TOOL_SPECS = [
    {
        'name': 'sports_match_search',
        'description': "体育赛事赛程组合式搜索工具(sports_match_search)。基于槽位检索赛程，命中后支持单场比赛跳转播放。当用户表达了查询、查找体育赛事赛程/对阵/比赛安排的意图时触发，包括按球队/俱乐部(sport_team，含主队sport_team_home/客队sport_team_guest)、运动员/球星(sport_star)、赛事名称(sport_game)、运动项目(sport_name)、比赛阶段(sport_competition_phase)、相对轮次(round_offset)、赛季(sport_season)、赛事日期(sport_time)等维度筛选，以及按比赛状态(live_state)过滤、赛事时间区间(sport_time_range)筛选的场景。典型用法：'皇马的比赛'、'皇马主场的比赛'、'NBA的比赛'、'中超的比赛'、'今天有什么足球比赛'。支持10个精确匹配维度、1个状态维度(live_state)和1个范围维度(sport_time_range)。布尔组合能力：不同槽位之间支持 AND/OR/NOT 任意嵌套(query 节点的 and/or/not)；同一槽位多值用 operator 选 and/or(默认 or)；NOT 排除用外层 not 节点表达；live_state 多值固定 OR 语义。工具边界：本工具只找赛程(对阵场次)；找体育录像/回放/集锦等点播视频用 sports_vod_search；预约比赛用 sports_match_reservation。",
        'required': ['query'],
    },
    {
        'name': 'sports_vod_search',
        'description': '体育视频组合式搜索工具(sports_vod_search)。用于查询【影视-体育分类媒资】，当用户表达了搜索、查找、推荐体育视频/录像/回放/集锦/纪录片等点播内容的意图时触发，包括按球队/俱乐部(sport_team，含主队sport_team_home/客队sport_team_guest)、运动员/球星(sport_star)、赛事名称(sport_game)、运动项目(sport_name)、比赛阶段(sport_competition_phase)、相对轮次(round_offset)、节目标题(title)、赛季(sport_season)、赛事日期(sport_time)等维度筛选，以及按赛事时间区间(sport_time_range)筛选的场景。支持11个精确匹配维度、1个范围维度(sport_time_range)。视频媒资本身均为已结束/可回看点播内容，状态固定无需也不应通过状态字段过滤。布尔组合能力：不同槽位之间支持 AND/OR/NOT 任意嵌套(query 节点的 and/or/not)；同一槽位多值用 operator 选 and/or(默认 or)；NOT 排除用外层 not 节点表达。工具边界：本工具找的是可点播的视频媒资(录像、回放、集锦、专题片)，且独有 title(节目/纪录片/专题片名)维度——这是与赛事搜索(sports_match_search，只查赛程对阵)最大的差异点；sports_match_search 找的是赛程(单场比赛跳转播放)；预约比赛用 sports_match_reservation。',
        'required': ['query'],
    },
    {
        'name': 'sports_rank_search',
        'description': "体育榜单查询工具(sports_rank_search)。用于查询体育排行榜/积分榜/球员数据榜等排名数据，命中后直接出榜单卡片，不做自然语言重述。适用前提：用户表述能锁定'赛事 + 榜单类型'且限定当前赛季。不适用场景(不调用本工具，由编排层走默认回复)：跨赛季/历史排名、预测或比较类问题、不限定赛事的泛化排名。典型说法：'中超积分榜'、'英超射手榜'、'NBA得分榜'、'NBA排行榜'。工具边界：本工具只处理榜单/排名类需求；要查赛程/对阵用 sports_match_search；要查球队本体资料用 sports_team_search。",
        'required': ['sport_game', 'sport_rank_type'],
    },
    {
        'name': 'sports_team_search',
        'description': "球队信息查询工具(sports_team_search)。根据球队名(sport_team)查询球队资料，返回球队卡片及关联的赛程/资讯数据。当用户表达查看某支球队本体信息、资料、主页的意图时触发，典型说法如'皇马'、'皇家马德里'、'查一下湖人'、'我想看国足'。仅含1个开放型精确匹配维度(sport_team)，检索 team_name 字段，需100%匹配才召回，因此应填写球队规范名或用户原词，不要自行改写。工具边界：本工具只查球队本体信息(球队卡片+相关赛程/资讯)；要按球队、赛事、时间等多维度检索对阵赛程用 sports_match_search；要找录像/回放/集锦等点播视频用 sports_vod_search。",
        'required': ['sport_team'],
    },
    {
        'name': 'sports_match_reservation',
        'description': "功能：预约体育赛事中的单场比赛或批量预约多场比赛(订阅开赛提醒/预约观看)。\n适用场景：用户表达预约/订阅/提醒某场比赛时，如'预约皇马的比赛''订阅明天的中超''提醒我詹姆斯的比赛开始''皇马和巴萨的比赛都帮我预约'，按球队/球星/赛事/项目/阶段/相对轮次/赛季/日期等维度(详见各 const)定位比赛，以及按时间区间(sport_time_range)筛选。\n多场预约：支持命中多场比赛时全部预约，不要求唯一收敛。\n动作边界：只执行预约。\n状态默认：预约天然仅针对未开始(live_state=1)的比赛，该状态由系统固定锁定，本工具不提供也不接受状态字段，无需/不应在 query 中传状态条件。\n关联工具：仅查赛程用 sports_match_search；找录像/回放用 sports_vod_search。\n逻辑：维度间支持 AND/OR/NOT 任意嵌套；同一维度多值用 operator 选 and/or；NOT 排除用外层 not 节点表达。",
        'required': ['query'],
    },
    {
        'name': 'sports_match_forecast',
        'description': "功能：预测某场具体比赛的结果(胜负/赛果/胜率概率)。\n适用场景：用户表达预测某一场具体比赛胜负时，如'预测今晚皇马对巴萨谁能赢''湖人这场能赢吗''帮我预测昨天结束的湖人比赛'(已结束比赛可做赛果复盘)，按球队/球星/赛事/项目/阶段/相对轮次/赛季/日期等维度(详见各 const)定位比赛，支持比赛状态(live_state，覆盖未开始/进行中/已结束/回看)和时间区间(sport_time_range)。\n动作边界：只做单场比赛的结果预测，不查赛程/不预约/不放录像。\n单场 vs 赛事预测(强制分流)：本工具仅承接'单场比赛预测'(对两支球队/两名选手之间一场具体对决预测胜负，结果以胜负卡片呈现)。判定要点：用户问的是'某一场谁赢'→本工具。\n关联工具：仅查赛程用 sports_match_search；预约比赛用 sports_match_reservation；找录像/回放用 sports_vod_search；\n逻辑：维度间支持 AND/OR/NOT 任意嵌套；同一维度多值用 operator 选 and/or；NOT 排除用外层 not 节点表达；live_state 多值固定 OR 语义。",
        'required': ['query'],
    },
    {
        'name': 'sports_match_lineup_search',
        'description': "体育比赛阵容查询工具(sports_match_lineup_search)。针对上下文中已锁定的某一场具体比赛，查询双方的完整阵容(如'这场比赛的阵容'、'首发名单是谁')。仅部分足球比赛支持。结果使用：直接展示阵容卡片，含首发+替补(同线上一致，不单独区分首发/替补分组)。本工具是指整份阵容名单的查询；若用户只问某一名球员在本场是否首发/是否上场(如'这场比赛C罗首发了吗')，属于单点信息问答，应使用 sports_match_event_qa_search，不属于本工具。前置条件：本工具不做赛程检索，不接受球队/赛事/时间等槽位，必须依赖上下文(当前播放或上一轮对话)已确定的具体比赛。数据覆盖限制：本工具仅覆盖部分足球比赛，非足球项目及无第三方数据的足球场次可能查不到，此时按空结果处理，不要臆造数据。",
        'required': [],
    },
    {
        'name': 'sports_match_event_card_search',
        'description': "体育比赛赛况查询工具(sports_match_event_card_search)。针对上下文中已锁定的某一场具体比赛，查询本场比赛的整体赛况(如'这场比赛的赛况'、'比赛情况怎么样')。仅部分足球比赛支持。结果使用：直接展示事件汇总卡片，按时间线汇总全场关键事件。与 sports_match_event_qa_search 不同——本工具是概览式的整场赛况展示，event_qa 是针对某类具体事件的提问并由模型总结作答。前置条件：本工具不做赛程检索，不接受球队/赛事/时间等槽位，必须依赖上下文(当前播放或上一轮对话)已确定的具体比赛。数据覆盖限制：本工具仅覆盖部分足球比赛，非足球项目及无第三方数据的足球场次可能查不到，此时按空结果处理，不要臆造数据。",
        'required': [],
    },
    {
        'name': 'sports_match_statistics_card_search',
        'description': "体育比赛技术统计查询工具(sports_match_statistics_card_search)。针对上下文中已锁定的某一场具体比赛，查询本场比赛的技术统计数据(如'这场比赛的数据'、'控球率多少'、'射门数据')。仅部分足球比赛支持。结果使用：直接展示技术统计卡片，含控球率、射门数等量化指标。与 sports_match_event_card_search 不同——本工具是量化统计指标，event_card 是离散的事件时间线。前置条件：本工具不做赛程检索，不接受球队/赛事/时间等槽位，必须依赖上下文(当前播放或上一轮对话)已确定的具体比赛。数据覆盖限制：本工具仅覆盖部分足球比赛，非足球项目及无第三方数据的足球场次可能查不到，此时按空结果处理，不要臆造数据。",
        'required': [],
    },
    {
        'name': 'fan_knowledge_agent',
        'description': '体育知识问答（fan_knowledge_agent）。关于体育的任何疑问都走这里：规则解释、球队/球星背景属性问答、历史战绩、预测闲聊（如"梅西的身高是多少""点球大战规则是什么""世界杯几年举办一次""某队赢了没有"），走体育知识问答链路（sport_qa + 具体 intent）。不做赛事/赛程/榜单检索（那是 sports_match_search 等）。\\n参数：{"messages":"用户原话整体传入（不要改写）"}。',
        'required': ['messages'],
    },
]

_TOOL_REQUIRED = {s["name"]: list(s["required"]) for s in _TOOL_SPECS}

# 第二层兜底：本域知识问答（sport_qa 场景）。
from app.fanparams import (  # noqa: E402
    domain_fan_params,
    looks_like_qa as _looks_like_qa,
    rebuild_fan_params,
)

# 条件树可用字段全集（schema 侧命名）。title 亦为合法字段（作品名精确匹配）。
_ALL_FIELDS = {
    "sport_team", "sport_team_home", "sport_team_guest", "sport_star",
    "sport_game", "sport_name", "sport_competition_phase", "sport_season",
    "sport_time", "sport_time_range", "round_offset", "live_state", "title",
}

# 时间类字段：取值由确定性解析补全，不依赖 LLM 推算
_TIME_FIELDS = {"sport_time", "sport_time_range", "round_offset"}

# 卡工具（matchId 由上下文决定；没有显式 matchId 时 params 输出 {}）
_NO_PARAM_TOOLS = {
    "sports_match_lineup_search",
    "sports_match_event_card_search",
    "sports_match_statistics_card_search",
}

_FIELD_CATALOG = """- sport_team 球队/俱乐部名
- sport_team_home 主队（用户说「X主场」时用这个）
- sport_team_guest 客队（用户说「X客场/对阵X的客队」时用）
- sport_star 运动员/球星名
- sport_game 赛事名称（如 中超/西甲/英超/NBA/欧洲杯/奥运会/F1）
- sport_name 运动项目（如 足球/篮球/乒乓球/跳水/格斗/山地自行车）
- sport_competition_phase 比赛阶段（如 常规赛/季后赛/预选赛/半决赛/小组赛）
- sport_season 赛季（如 2025-2026）
- sport_time 单日日期（yyyyMMdd）
- sport_time_range 时间区间（from/to，形如 "20260824 00:00:00"）
- round_offset 相对轮次（"1"=下一轮，"-1"=上一轮，相对于当前轮次）
- live_state 比赛状态（"1"=未开始，"3"=进行中，"4"=已结束）"""

_SYSTEM_TMPL = """你是电视语音助手的体育意图解析器。读用户的话，输出该调用的工具名和参数。

【工具】
{tool_briefs}

【怎么选工具】
按下面的顺序判断，**先命中先返回**：
1. 「**预约/订阅/提醒**某场比赛」→ sports_match_reservation。
2. 「**某支球队本身**」的资料/主页/介绍（如「皇马」「查一下湖人」「我想看国足」）
   → sports_team_search。
3. 「**积分榜/排名/排行榜/榜单**」→ sports_rank_search（同时要 sport_game + sport_rank_type）。
4. 上下文里**已锁定一场比赛**，问「**阵容/首发/大名单**」→ sports_match_lineup_search。
5. 上下文里**已锁定一场比赛**，问「**赛况/战况/比分/发生了什么**」→ sports_match_event_card_search。
6. 上下文已锁定一场比赛，问「**技术统计/控球率/射门数/数据**」→ sports_match_statistics_card_search。
7. 「**预测/谁能赢/能赢吗/胜率/胜负**」某一(些)具体比赛 → sports_match_forecast
   （条件树定位要预测的比赛；**只有出现明确预测词才算 forecast**，只报对阵是查赛程）。
8. 要**看视频**（视频/录像/集锦/回放/精彩瞬间/纪录片/颁奖/开幕式）→ sports_vod_search。
9. 其余「**查赛程/对阵/比赛安排**」（找比赛、看有什么比赛）→ sports_match_search。
10. 其余**体育知识问答/规则/球星背景/比分闲谈**（「点球规则是什么」「世界杯几年一届」「梅西身高」）→ fan_knowledge_agent。

注意「比赛回放/比赛回看/比赛录像」属于 **sports_match_search**（按 live_state 过滤已结束），
不是 sports_vod_search。

【参数怎么写】
A) sports_team_search：`{{"sport_team":"球队名"}}`
B) sports_rank_search：`{{"sport_game":"赛事","sport_rank_type":"积分榜"}}`
   （sport_rank_type 候选：球队积分/综合排名/射手榜/助攻榜…）
C) lineup / event / statistics 三个卡工具：**params 输出 `{{}}`**（作用于已锁定比赛）
D) match_search / vod_search / reservation / forecast：`{{"query": <条件树>}}`
E) 体育知识问答/规则解释（"点球规则是什么""世界杯几年一届"）→ fan_knowledge_agent，
   params 填 `{{"messages":"用户原话整体传入（不要改写）"}}`

【query 条件树】
可用字段：
{fields}

节点形态（任意嵌套）：
- 单值：{{"field":"sport_game","value":"中超"}}
- 多值：{{"field":"sport_team","values":["尤文图斯","AC米兰"],"operator":"or"}}
- 组合：{{"and":[...]}} / {{"or":[...]}} / {{"not":{{...}}}}
- 条件只有一个时，query 直接写那个叶子，不要包 and。

取值口径：
- **时间维度照常写出来**（`sport_time` / `sport_time_range` / `round_offset`），
  具体日期由后处理补全，你只需填 `{{"field":"sport_time","value":""}}` 占位。
  用户说「上一轮/下一轮/下场」用 `round_offset`（"-1"/"1"）。
- 「X主场」用 sport_team_home；「X客场」用 sport_team_guest。
- 说了球队或球星就别重复给运动项目，除非用户也说了。
- 字段无法归一到常用值时不要造新值。

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


def _exact(field: str, value: str) -> dict:
    return {"field": field, "value": value}


def _strip_time(node):
    """摘掉条件树里全部时间叶（模型给的取值不可靠）。"""
    if not isinstance(node, dict):
        return node
    if node.get("field") in _TIME_FIELDS:
        return None
    if "and" in node and isinstance(node["and"], list):
        inner = [_strip_time(x) for x in node["and"]]
        inner = [x for x in inner if x is not None]
        if not inner:
            return None
        return inner[0] if len(inner) == 1 else {"and": inner}
    out = {k: _strip_time(v) for k, v in node.items()}
    out = {k: v for k, v in out.items() if v is not None}
    return out or None


def _as_list(node) -> list:
    if node is None or node == {}:
        return []
    if isinstance(node, list):
        out = []
        for x in node:
            out.extend(_as_list(x))
        return out
    if isinstance(node, dict) and "and" in node and isinstance(node["and"], list):
        return _as_list(node["and"])
    return [node]


def _merge(node, extra: list[dict]) -> dict:
    """把确定性时间叶并入条件树，保持 golden 形状。"""
    rest = _as_list(node)
    all_leaves = rest + list(extra)
    if not all_leaves:
        return {}
    if len(all_leaves) == 1:
        return all_leaves[0]
    return {"and": all_leaves}


# 时间基准：评测双锚零漂移（run.sh 里 HC_SPORTS_RUNTIME_TODAY=1 → 真实当天；否则用固定锚）。
import os
_runtime_today = os.environ.get("HC_SPORTS_RUNTIME_TODAY", "0") == "1"
_BASE_DATE = None  # 若置为 date(2026,8,24) 之类则是评测锚，None 用今天


def _rel_base(q: str):
    """返回日期基准：线上当天；评测可设锚。"""
    if _BASE_DATE is not None:
        return _BASE_DATE
    from datetime import date as _d
    return _d.today()


def _parse_date_cn(q: str, base) -> str | None:
    """解析「明天/后天/昨天/周X/X月X日」→ yyyyMMdd（简化：真实的完整解析器在 [old] rules，此处覆盖最常见）。"""
    from datetime import timedelta
    for w, off in (("明天", 1), ("后天", 2), ("昨天", -1), ("前天", -2)):
        if w in q:
            return (base + timedelta(days=off)).strftime("%Y%m%d")
    m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", q)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            return date(base.year, month, day).strftime("%Y%m%d")
        except ValueError:
            pass
    return None


def _resolve_time(query: str, base) -> list[dict]:
    """确定性解析时间维度 → 条件树叶子。返回空列表表示无法解析。"""
    leaves: list[dict] = []
    if re.search(r"上一轮|上轮", query):
        leaves.append(_exact("round_offset", "-1"))
    if re.search(r"下一轮|下轮|下场|下场比赛", query):
        leaves.append(_exact("round_offset", "1"))
    d = _resolve_date_cn(query, base)
    if d:
        leaves.append(_exact("sport_time", d))
    return leaves


def _resolve_date_cn(q: str, base) -> str | None:
    """解析常见中文日期；无识别返回 None。"""
    return _resolve_date_cn_impl(q, base)


def _resolve_date_cn_impl(q: str, base) -> str | None:
    from datetime import timedelta
    for w, off in (("明天", 1), ("后天", 2), ("昨天", -1), ("前天", -2)):
        if w in q:
            return (base + timedelta(days=off)).strftime("%Y%m%d")
    m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", q)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            return date(base.year, month, day).strftime("%Y%m%d")
        except ValueError:
            return None
    if re.search(r"今天|今日", q):
        return base.strftime("%Y%m%d")
    m = re.search(r"(?:周|星期)([日一二三四五六天])", q)
    if m:
        target = "一二三四五六日".find("日" if m.group(1) == "天" else m.group(1))
        cur = base.weekday()  # 0=周一
        diff = (target - cur) % 7
        return (base + timedelta(days=diff)).strftime("%Y%m%d")
    return None


def fix_params(query: str, tool: str, params: dict) -> tuple[str, dict]:
    """确定性补全时间类字段取值（今天/明天/周X等模型算不准的）。"""
    if not isinstance(params, dict):
        return tool, {}
    if tool in _NO_PARAM_TOOLS:
        return tool, {}
    q = params.get("query")
    if not isinstance(q, dict):
        return tool, params
    base = _rel_base(query)
    resolved = _resolve_time(query, base)
    node = _strip_time(q)
    if resolved:
        node = _merge(node, resolved)
    return tool, {**params, "query": node}


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
            logger.error("sports LLM 调用失败：%s", exc)
            break
        parsed = _parse(msg, valid)
        if parsed is None:
            logger.warning("sports LLM 输出不合法（第 %d 次）：%r", attempt, str(msg.get("content"))[:200])
            continue
        tool, params = parsed
        if tool == "fan_knowledge_agent":
            # 本域知识问答：重建 fan 契约（sport_qa，intent 由 fanparams 判 quiz 形态）
            params = rebuild_fan_params(params, scene_hint="sport_qa", fallback_query=query)
        else:
            try:
                tool, params = fix_params(query, tool, params)
            except Exception as exc:  # noqa: BLE001 补全失败不阻断
                logger.warning("sports fix_params 失败（%s）：%s", tool, exc)
        return Prediction(domain=domain.name, tool=tool, params=params, hit_source="llm")

    # 两层兜底：match_search 第一层；query 呈知识问答形态 → 本域 fan（sport_qa 场景）。
    if _looks_like_qa(query):
        return Prediction(domain=domain.name, tool="fan_knowledge_agent",
                          params=domain_fan_params("sports", query), hit_source="fallback_qa")
    return Prediction(domain=domain.name, tool=FALLBACK_TOOL,
                      params={"query": {"field": "sport_time", "value": ""}},
                      hit_source="fallback")