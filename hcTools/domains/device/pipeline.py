"""设备（device）域：LLM 为中心，无规则层、无 few-shot、无 bad-case。

决策全部在 `pipeline.py`：一次 LLM 调用直接出 {tool, params}；LLM 失败/输出不合法
→ common_control 兜底。

参数是扁平 5 槽 {operation, object, value, device, location}（solve_* 只给 intent，
timer 可有 date_time）。object 是封闭 canonical 词表（_OBJ_TOOL，208 条）：
LLM 负责选工具/槽位 + object 用原话；词表做「最长命中 → canonical 拼写」的**协议层
归一**（命中才改，未命中保留 LLM 判定）。这部分是参数契约而非决策外挂。
"""

from __future__ import annotations

import logging
import re
from datetime import date

from app.llm import _parse_json_block, chat
from app.metadata import PURPOSE_ONECALL, build_llm_metadata
from app.models import Prediction, PredictRequest

from ._vocab import _OBJ_TOOL

logger = logging.getLogger("hcTools.domains.device")

FALLBACK_TOOL = "common_control"
MAX_ATTEMPTS = 2

# 17 个工具的完整定义（来自原 domains/device/schema.json，全量内联，不再依赖 schema.json）。
_TOOL_SPECS = [
    {
        'name': 'numeric_adjust',
        'description': '功能：负责设备幅值相关参数的控制，设置项包括音量、清晰度、对比度、色度、亮度、分辨率、倍速、刷新率，设置操作包括打开、关闭、提升、升高、增加、降低、减少、设置、查询操作。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'power_control',
        'description': '功能：负责设备开机、关机、重启控制。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'timer_control',
        'description': '功能：负责设备定时控制与查询，包括定时开机、定时关机、定时熄屏，以及查询剩余关机时长。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'source_switch',
        'description': '功能：负责电视设备的输入信号源控制，控制项包括信号源的切换、打开和关闭，信号源包括HDMI、typec、usb、VGA、分量、视频、Switch、xbox、ps4、ps5、机顶盒、模拟电视、数字电视、地面数字电视、直播电视。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'playback_control',
        'description': '功能：负责设备的播放控制，控制项包括播放、停止、退出、重播、快进、快退、跳转、列表播放、顺序播放、随机播放、循环播放、上一个、下一个等。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'video_picture_save_share',
        'description': '功能：负责设备中视频和图片的存储和分享控制，包括视频或图片的保存、删除，以及分享到微信和抖音等。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'mode_control',
        'description': '功能：负责设备的模式控制，负责实现各种模式的切换控制，包括图像模式、声音模式、音效模式、混响模式、整机模式、护眼模式、商场模式等。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'screen_layout',
        'description': '功能：负责电视屏幕画面布局的控制，包括分屏、全屏、小屏、画面缩放。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'screen_lift_rotation',
        'description': '功能：负责显示屏的物理旋转和升降控制，包括横屏、竖屏、自由角度旋转、支架升高或降低。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'demo_control',
        'description': '功能：负责设备效果演示的打开或关闭控制，包括关键词演示、展示、demo、体验。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'display_control',
        'description': '功能：负责设备画质、背光、畸变校正相关功的打开或者关闭控制。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'audio_control',
        'description': '功能：负责设备音质、音效相关功能的打开或者关闭控制。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'smart_camera',
        'description': '功能：负责摄像头、雷达、AI视觉相关功能的打开和关闭。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'network_control',
        'description': '功能：负责设备网络功能的打开和关闭。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'screensaver_control',
        'description': '功能：负责设备屏幕保护的控制，包括屏幕的开启、关闭；屏保、壁纸的开启、关闭和设置；熄屏、亮屏、待机、睡眠的开启和关闭。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'common_control',
        'description': '功能：负责设备兜底功能的打开和关闭。',
        'required': ['operation', 'object'],
    },
    {
        'name': 'solve_picture_sound_problem_control',
        'description': '功能：负责处理设备图像和声音问题的控制，解决只描述设备控制类问题的请求，包括图像控制类问题、声音控制类问题。',
        'required': ['intent'],
    },
    {
        'name': 'fan_knowledge_agent',
        'description': '设备知识问答（fan_knowledge_agent）。用户对电视设备使用/功能/设置的任何疑问解释（如"HDMI是什么""怎么连接蓝牙""为什么没有声音""这个接口干嘛用的"），都走设备知识问答链路（device_control_qa 场景）。不做设备控制动作（那是 common_control 等）。\\n参数：{"messages":"用户原话整体传入（不要改写）"}。',
        'required': ['messages'],
    },
]

_TOOL_REQUIRED = {s["name"]: list(s["required"]) for s in _TOOL_SPECS}

# 第二层兜底：本域知识问答（device_control_qa 场景）。
from app.fanparams import (  # noqa: E402
    domain_fan_params,
    looks_like_qa as _looks_like_qa,
    rebuild_fan_params,
)


def _norm(s: str) -> str:
    """词表 key 的归一：小写去空格。"""
    return re.sub(r"[\s._\-]", "", (s or "").lower())


def _longest_match(low_q: str):
    """原话归一后做最长词表面命中，返回 (tool, canonical)。"""
    for k, (tool, canon) in _OBJ_TOOL.items():
        if k and k in low_q:
            return tool, canon
    return None


def _fallback_object(query: str) -> str:
    """未命中词表时的兜底 object：剥离动词后其余串小写去空格。"""
    t = query.strip()
    for v in ("打开", "启动", "进入", "开启", "切换", "调用", "启用", "展开", "进行"):
        if t.startswith(v):
            t = t[len(v):]
            break
    t = re.sub(r"的$", "", t).strip().lower().replace(" ", "")
    if t == "音乐功能":
        t = "音乐播放器"
    return t


_SYSTEM_TMPL = """你是电视设备控制意图解析器。读用户的话，输出该调用的工具名和参数。

【工具】
{tool_briefs}

【怎么选工具】
按下面的顺序判断，先命中先返回：
1. 用户在描述 **画质/声音的问题**（偏色、太暗、模糊、噪点、拖尾…）而不是下达控制指令
   → solve_picture_sound_problem_control。
2. 出现具体 **信号源**（HDMI/USB/VGA/机顶盒/电视源/外接设备）→ source_switch。
3. **播放控制**（快进、快退、跳转、暂停、上下集、列表、倍速、重播、退出）→ playback_control。
4. **屏幕布局**（分屏、全屏、小屏、画面缩放）→ screen_layout。
5. **数值调节**（音量、亮度、对比度、色度、清晰度、分辨率、刷新率、麦克风音量、静音）
   → numeric_adjust。
6. **定时关机**（关机 + 时间/定时/多久后）→ timer_control。
7. **电源**（开机、关机、重启、唤醒，无时间）→ power_control。
8. **模式**（图像/声音/音效/护眼/商场/会议/混响模式）→ mode_control。
9. **屏保族**（屏保、壁纸、壁画、屏保画册、屏保VIP、熄屏亮屏待机睡眠）→ screensaver_control。
10. **屏幕升降旋转**（横竖屏、旋转、支架升降）→ screen_lift_rotation。
11. 其余**具体功能对象**（网络、摄像头、音响、背光、演示 demo、低音…）→ 对应工具
    （network_control / smart_camera / audio_control / display_control / demo_control）。
12. 都对不上又像「打开某功能」→ common_control。
13. **设备使用/功能/设置的疑问解释**（「HDMI是什么」「怎么连蓝牙」「为什么没声音」）→ fan_knowledge_agent。

【参数怎么写】
params 是扁平 5 槽（{{operation, object, value, device, location}}）：
- operation：操作类型，从该工具允许的取值里选（打开/关闭/设置/提高/降低/查询/播放/暂停…）。
- object：控制对象，用**用户原话里的功能名**（后端会做规范化映射，不要自己改写）。
- value：数值或单位（"30"、"首"、"集"），没有就 ""。
- device：仅在用户明确说「电视」时填 "电视"，否则 ""。
- location：恒为 ""。
- timer_control 额外可带 date_time（如 "30分钟"、"22:00"、"定时"）。
- solve_picture_sound_problem_control **只输出** {{"intent":"<问题意图>"}}，不要 5 槽。
- 「开通/购买/续费/办理 屏保VIP」这类**购买/续费**类请求 params 输出空对象 {{}}。
- 设备使用/功能/设置的疑问解释（"HDMI是什么""怎么连蓝牙""为什么没声音"）→ fan_knowledge_agent，
  params 填 {{"messages":"用户原话整体传入（不要改写）"}}。

【输出】
只输出一个 JSON 对象，形如 {{"tool":"工具名","params":{{...}}}}，不要任何解释。
"""


def _tool_briefs(tools) -> str:
    lines = []
    for t in tools:
        desc = " ".join(t.description.replace("\\n", " ").split())
        lines.append(f"- {t.name}：{desc[:600]}")
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


def fix_params(query: str, tool: str, params: dict) -> tuple[str, dict]:
    """协议层归一：object 若是 208 词表命中，改为 canonical 拼写（LLM 判定不影响）。"""
    if not isinstance(params, dict):
        return tool, {}
    if tool == "solve_picture_sound_problem_control":
        params.setdefault("intent", "")
        return tool, params
    low = _norm(query)
    hit = _longest_match(low)
    obj = params.get("object")
    if hit:
        hit_tool, canon = hit
        # 归一 object；tool 以 LLM 判定为准（词表仅提供 canonical 拼写）
        if obj is None or not obj:
            params["object"] = canon
        else:
            params["object"] = canon
    if not params.get("object"):
        params["object"] = _fallback_object(query)
    params.setdefault("value", "")
    params.setdefault("device", "")
    params.setdefault("location", "")
    params.setdefault("operation", params.get("operation") or "打开")
    return tool, params


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
            logger.error("device LLM 调用失败：%s", exc)
            break
        parsed = _parse(msg, valid)
        if parsed is None:
            logger.warning("device LLM 输出不合法（第 %d 次）：%r", attempt, str(msg.get("content"))[:200])
            continue
        tool, params = parsed
        if tool == "fan_knowledge_agent":
            params = rebuild_fan_params(params, scene_hint="device_control_qa", fallback_query=query)
        else:
            try:
                tool, params = fix_params(query, tool, params)
            except Exception as exc:  # noqa: BLE001
                logger.warning("device fix_params 失败（%s）：%s", tool, exc)
        return Prediction(domain=domain.name, tool=tool, params=params, hit_source="llm")

    # 两层兜底：common_control 第一层；query 呈知识问答形态 → 本域 fan（device_control_qa 场景）。
    if _looks_like_qa(query):
        return Prediction(domain=domain.name, tool="fan_knowledge_agent",
                          params=domain_fan_params("device", query), hit_source="fallback_qa")
    return Prediction(domain=domain.name, tool=FALLBACK_TOOL,
                      params={"operation": "打开", "object": _fallback_object(query),
                              "value": "", "device": "", "location": ""},
                      hit_source="fallback")