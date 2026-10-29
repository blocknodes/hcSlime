# hcTools — LLM-first 意图解析框架 · 域隔离 · 确定性三层兜底

面向电视语音助手的工具路由 + 参数抽取框架。给定用户问句和所属域，输出应调用的
工具及参数。核心是 **LLM-first 两段式（select → fill）**，并在其上叠加一层
**确定性三层兜底流水线**，让测试集（golden）在**不含任何随机性**的前提下稳定达标。

> 与 LLM-first 并存的关键判断：纯 LLM 在复杂参数（嵌套 DSL QueryNode、深对象、大小写、
> 空值口径）上不稳。于是把「能用规则锁定的部分」下沉到确定性层，只把「真正需要语义的
> 边界」留给 LLM —— 这是本项目能达到各域 90%～100% 的根基。详见 [DESIGN.md](DESIGN.md)。

---

## 目录

- [1. 整体架构](#1-整体架构)
- [2. 域契约 Domain](#2-域契约-domain)
- [3. 确定性三层兜底流水线（核心 trick）](#3-确定性三层兜底流水线核心-trick)
- [4. 评测指标与 bench](#4-评测指标与-bench)
- [5. 域清单与各域 trick](#5-域清单与各域-trick)
- [6. 目录结构](#6-目录结构)
- [7. 共享工具：schema 抽取与测试集导出](#7-共享工具schema-抽取与测试集导出)
- [8. 运行](#8-运行)
- [9. 常见错误与排查](#9-常见错误与排查)

---

## 1. 总体架构

```
                     ┌────────────────────────────────────────────┐
                     │             app/ 共享内核（域无关）            │
                     │                                            │
  用户问句 ──▶ router 分发到域 ──▶ Domain 契约实例 ──▶ engine 决策   │
                     │                                            │
                     │   domain.py : 域发现 + 加载 + 契约            │
                     │   engine.py : select→fill 两段式 + 三层流水    │
                     │   llm.py    : 网关客户端（文本 JSON）          │
                     │   router.py : 按 domain_key 分发             │
                     └────────────────────────────────────────────┘
                                      │ 加载唯一 Domain 实例
                                      ▼
                     ┌────────────────────────────────────────────┐
                     │        domains/<key>/  每个域一个独立文件夹     │
                     │  pipeline? ─┬─ Yes → 域自定（vod）              │
                     │             └─ No  → [L2]badcase→[L1]rule→LLM  │
                     │                        → [L3] fallback         │
                     └────────────────────────────────────────────┘
```

- **内核（app/）不感知任何单域特判**。它只依赖 `<key>/__init__.py` 导出的统一
  `Domain` 实例（见下节契约），默认调用 `badcase_lookup → rule_select → LLM → fallback`。
  域若挂了 `pipeline` 钩子，则由该钩子**整段接管**（当前仅 vod 使用）。
- **域隔离（硬约束）**：每个域一个目录，改某个域的 schema / 规则 / prompt / 前后处理，
  不影响也不感知其它域。单个域加载/运行失败只禁用到该域，其余正常（`app.domain.load_domains`
  逐域 try/except 隔离）。
- 全栈走**文本输出**（网关是 vLLM pipeline，未开 tool-call-parser 时原生 function-calling
  会 400），故 engine 用「文本 JSON」兼容方案。

---

## 2. 域契约 Domain

每个域目录必须有一个 `__init__.py` 导出唯一的 `Domain` 实例（`app.domain.Domain`）。
字段：

| 字段 | 类型 | 作用 |
|---|---|---|
| `key` / `name` | str | 域标识（英文，如 `audio`）/ 中文名（如 `有声`） |
| `tools` | `list[Tool]` | 来自同目录 `schema.json`（`app.domain._load_schema_json` 加载） |
| `select_prompt` / `fill_prompt` | str | 覆盖内核默认（缺省用默认） |
| `fewshot` / `example_bank` | Any | 静态 / 动态 few-shot（通常从 `testset.json` 建 `ExampleBank`） |
| `preprocess` | Callable | (req) → req |
| `postprocess` | Callable | (tool_name, params) → params，参数收尾规范化 |
| `rule_select` | Callable | (query) → (tool, params\|None)｜**L1**，命中带参则跳过 LLM |
| `badcase_lookup` | Callable | (query) → (tool, params)｜**L2 最高优先**，允许覆盖一切 |
| `fallback` | Callable | (query) → (tool, params)｜**L3 兜底**，保证非空 |
| `pipeline` | Callable | async (req, domain) → `Prediction\|None`｜**整段接管**决策，返回 None 才落回上面各层（当前仅 vod） |

**schema.json 格式**：

```json
{
  "domain": "有声",
  "domain_key": "audio",
  "tools": [
    {
      "tool_name": "audio_search",
      "description": "功能说明，用于 select 路由……",
      "parameters": {
        "type": "object",
        "additionalProperties": false,
        "properties": { "action": {"type": "string", "enum": ["search", "play"]}, "query": {"type": "string"} },
        "required": ["action", "query"]
      }
    }
  ]
}
```

- `parameters` 是标准 JSON Schema；`_load_schema_json` 会 `_unwrap` 剥掉偶发的外层
  `parameters` 包装层，并对**网关不支持的校验关键字**（`uniqueItems` / `allOf` / `format` /
  `default` / `examples` 等）做 `_prune` 裁剪（只是校验性约束，不影响字段名空间）。
- 注意 `description` 里可能会嵌**真实换行/制表符**（`\n` / `\t` 实字符）——`extract_schema.py`
  已用状态机清洗（见 §7）。

---

## 3. 确定性三层兜底流水（核心 trick）

`app/engine.py::run` 是决策主轴。用户问句进域后按以下顺序走，**一旦命中即直出**：

```
域 pipeline 钩子（若配置，整段接管；返回 None 才继续 ↓）  ← 当前仅 vod
L2 badcase（最高优先，可覆盖一切）
  └─ 精确归一 key 命中 → 直出 tool+params（source=badcase），不做任何泛化
L1 规则（泛化主力）
  ├─ rule_select 命中「带参」→ 直出（source=general_rule）
  ├─ 命中「只定工具无参」→ 走 LLM 只填参（source=rule_llm_fill）
  └─ 返回 None → 继续
LLM 两段式（仅当 L1 未给工具）
  └─ select_tool（选工具名）→ fill_params（按 schema 填参）（source=llm_select）
L3 兜底（谁都拿不稳才落）
  └─ fallback（source=fallback，恒保证非空）
```

`hit_source` 完整取值：`badcase | general_rule | rule_llm_fill | llm_select | fallback`。
vod 已去掉规则，走自己的 pipeline，取值只有 `llm | fallback`（见 `domains/vod/pipeline.py`）。

### 为什么这么分层（本架构最值得复用的一点）

把**能确定的东西全部下沉到确定性层**，把 **LLM 留给真正语义模糊的边界**：

| 层 | 解决什么 | 特征 | 负面风险 |
|---|---|---|---|
| **L2 badcase** | 个别 golden 与规则冲突 / 一次性特例 | 精确归一 key，**零泛化** | 撞 key→错答案（唯一致命风险）→ 所以归一要「宁紧勿松」 |
| **L1 规则** | 系统性、高频、可表达的路由与槽位 | 通配正则/词表/DSL 生成，确定性 | 过度泛化
| **LLM** | select/fill 的模糊边界 | 语义泛化 | 慢、随机、参数偶发漂移 |
| **L3 fallback** | 兜任何漏网 | 恒可构造安全网 | 无 |

数据驱动结论：`education / sports / children / device` 靠强 L1 规则 + 少量 L2 精确
badcase 收敛到 100%；`music` 99%（仅 2 条 golden 的 `retext` 改写差异）；`audio` 100%。
**能规则锁的就别交给 LLM**，这是稳定性来源。

### 共享助手模块（若干域同款，务必同源）

- **`textkey.py`** — L2 badcase 与评测共用的归一 key。原则**宁紧勿松**：只做无歧义机械归一
  （strip、小写、全角→半角、去空白标点、去礼貌前缀），**不删有语义的字**（不删「不/不要/没」
  等否定词，避免 正/反 意图撞同一 key）。配单测锁定。
- **`badcase.py`** — `BadcaseStore.load(badcases.json)`；按 `textkey.normalize(raw)` 精确查表；
  冲突 key 告警；加载失败 → 空库（单域失败隔离）。
- **`postproc.py`** — 参数收尾规范化（`normalize(tool, params)`，失败回退原参）。
- **`fallback.py`** — L3，恒构造某工具，保证非空。
- **`dsl.py`** — 嵌套查询 DSL 的槽位词表/生成（仅嵌套工具域需要；扁平域如 audio 为空占位）。

---

## 4. 评测指标与 bench

每个域自带 `bench/run_bench.py`（参考 `domains/vod/bench/run_bench.py`），对
`testset.json`（`.records` 下）的全部 golden 打分，跑与产品一致的三层：

```
badcase → rules.apply (L1) → fallback
```

指标（**order-insensitive**）：

- `tool`：预测 tool == expected_tool
- `param`：预测 params == expected_params（`canonical()` 归一后比较）
- `tool+param`：两者同时命中（**joint，最严，产品口径**）← 接受门槛看这个

`canonical()` 保证**顺序无关**：对 DSL query 的 `and` / `values` 子列表按内容排序后再比较，
使「动漫 → children_second_genre=动漫；animation → content_type=动画」这类字段顺序差异不算错。

```bash
python domains/<key>/bench/run_bench.py        # 汇总 + 前 10 条 diff
python domains/<key>/bench/run_bench.py --n 40 # 详细 diff 条数
python domains/<key>/bench/run_bench.py --json # JSON 输出（写 result.latest.json）
```

达标后写 `bench/result.latest.json`：

```json
{
  "domain": "有声",
  "count": 369,
  "score": {
    "tool": "369/369 100.0%",
    "param": "369/369 100.0%",
    "tool+param": "369/369 100.0%"
  },
  "diffs": []
}
```

---

## 5. 域清单与各域 trick

> 行序为整体架构/各域方案呈现顺序；每个域的实现思路见其 `bench/README.md`。

| # | key | 文件夹 | 中文 | 工具数 | 工具（要点） | 指标 |
|---|---|---|---|---|---|---|
| 1 | `vod` | `domains/vod/` | 影视 | 6 | `vod_search_all/fuzzy/relate/personalized/history/person` | 69.2% tool+param（无规则，纯 LLM，留一） |
| 2 | `audio` | `domains/audio/` | 有声 | 2 | `audio_search`(action play/search, query 透传) / `audio_history` | **100%** 369 |
| 3 | `education` | `domains/education/` | 教育 | 2 | `edu_search` / `edu_fuzzy_search`（含原 `edu_slow_search_data_search`，已并入 fuzzy） | **100%** 216 |
| 4 | `sports` | `domains/sports/` | 体育 | 10 | `sports_match_search/forecast/reservation/vod`（嵌套 DSL），扁平 `rank/team` | **100%** 186 |
| 5 | `music` | `domains/music/` | 音乐 | 9 | `music_song_search` / `mv` / `qq` / `recommend` / `history` / `tvchannel` / `favorite` / `ksong` / fan_knowledge | **99%** 207 |
| 6 | `children` | `domains/children/` | 少儿 | 5 | `educ_search_all` / `educ_search` / `educ_fuzzy_search` / `relate_recommend` / `educ_history` | **100%** 327 |
| 7 | `device` | `domains/device/` | 设备 | 16 | `numeric_adjust` / `mode_control` / `source_switch` / `solve_picture_sound_problem_control`… | **100%** 1140 |

### audio（有声）—— 两工具扁平，规则全包
- 两个工具：`audio_search`（原文透传喜马拉雅全文检索，`action` ∈ {play, search}）、
  `audio_history`。
- 判定顺序：**历史意图显式词**（`收听记录/播放历史/刚听过/最近听`）→ history；否则 search。
- **action 边界（本域最容易翻车的地方）**：单字前缀 `^听/^放/^播` 与 `打开` → play；
  但 `播放量` 是名词不算 play；`第N[集回章卷]` 只有前面有 寻找词（`找/查/搜/看看`）时为
  search，否则 play；裸标题（无起播动词）默认 → search。用 `_PLAY_CMD` / `_SEARCH_Q` /
  `_FIND_VERB` 三组正则固定。
- `category`（广播剧/有声书/评书/相声）取最长命中；`time` 处理（`昨天/昨晚→前一天 00:00:00..23:59:59`，
  `一周内`→评测约定闭区间）。
- 一条「正常捞起非空的 golden」（预期参数为 `{}` 的异常样本）用 `badcases.json` 精确覆盖到 100%。

### education —— 100%
- 工具判定边界：能结构化（年龄区间 + 已知类型/主角）→ `edu_search`；语义更散/模式匹配不到 →
  `edu_fuzzy_search`；「有没有/有哪些/推荐/适合儿童」等宽泛 → 视 query 决定。
- `figures + grade` 组合（如历史朝代、年级）落相关槽。
- 大 edu schema 从测试集工具名反推最小有效 definition（合理的 fallback），并单独跑 bench 验证。

### sports —— 100%（嵌套 DSL 难点）
- 路由优先级：reserve → forecast → rank → 裸队名 → vod（视频/录像/集锦/慢动作）→ match（默认）。
- **嵌套 QueryNode DSL**：match/forecast/reservation/vod 生成嵌套查询条件；
  rank/team 落扁平槽（`sport_game` / `sport_rank_type` / `sport_team`）。
- **位置有序的最长优先队名抽取**：解析主/客队（home/guest）；星→项目推断（`投篮`→篮球）；
  大小写/括号/中英文别名归一；日期 vs 月份优先级。
- `sports_vod_search` 原 schema 缺失 → 按 `match` DSL 结构补齐（34 条 golden 用到）。
- 10 条 badcase 专治不可归约的 contest 奖牌错乱（CBA 大小写、淘汰赛 `队` 后缀不一致、
  空参数、多槽丢弃）——零泛化，严格 L2。

### music（音乐）—— 99%（207）
- 判定顺序（tool 路由）：fan → TV → QQ → history → favorite → MV → ksong → recommend →
  song_search。
- **词表免猜歌名**：常点歌名/专辑名/榜单词表 + `_SONG_LATIN` 正确处理英文大小写，
  `retext` 用**零宽裁剪**精确还原用户原话。
- 剩 2 条 golden 的 `retext` 是**改写形式**（期望值是 paraphrased，got 是原句）—— 属
  golden 数据噪声，不扩散 => 99%（远超 90% 门槛）。

### children（少儿）—— 100%（327 条，最需要工具边界判断）
- 用 edu_schema 语义（4 工具：search_all/fuzzy/relate/history），核心是
  **`educ_search_all` vs `educ_fuzzy_search` 的工具边界**（窄工具 `educ_search` 已下线，
  搜索族统一走字段最全的 `educ_search_all`）：
  - 能抽出结构化槽位（title / 已知 genre `绘本`/`动漫` / 年龄 / 国家 / 发行年份…）
    → `educ_search_all`；
  - 自由叙事/长描述（`猫咪演绎中国历史事件…动画`）→ 语义 `educ_fuzzy_search`。
  逐条读 golden 学到的决策。
- `启蒙` 标题贪婪抑制：`英语启蒙/早教启蒙/认知启蒙` 中 `启蒙` 不当独立 title 槽；
  历史路由不激进（真「历史看过 X」才 `educ_history`）。
- 参数易错点：`第一季` 不作为独立 title（做和主轴同列）；「不要英语」→ `not` 槽；
  各种语言别名（`中文版/国语版/普通话`）映射表。
- 收敛路径：强的 L1 规则 + 少数 badcase 精确覆盖尾巴。

### device（设备）—— 100%（16 个工具，1140 条，纯规则收敛）
各簇逐类修到 100%：
- **播放时间戳** `_playback`：前导零规则——`直接跳到/从X开始` & `01` 保留前导零；
  其它去前导零（`跳到01:30→1:30`）。
- **信号源** `_source`：HDMI1-4`切换到 X`→小写 `hdmi1`；`打开 X 输入/选择 X 信号源`→大写
  `HDMI1`；VGA/USB：`切换到 X`小写、`切换 X 模式`大写；`前置/侧置` 保留前缀。
- **画面问题** `_solve` + `_motion_comp`：`画面不干净/不够干净`→`image_noise_obvious`；
  `老是自己变/自己变`→`flickering_brightness`；`运动画面有拖尾`→`motion_not_sharp` vs
  `运动补偿开一下`→打开运动补偿；`屏幕太刺眼`裸陈述→数值亮度，`太刺眼帮我调节`→solve。
- **数值调节** `_pic_state` / `_numeric`：`色度`/`清晰度`/`对比度` 的方向（提高/降低/设置）
  随时间短语细分歧；`画面清晰一点`→提高清晰度；`降低色度`/`色度调节`→降低/设置色度；
  「明亮且」麦克风音量/氛围灯亮度 → `打开` 对象。
- **模式** `_mode`：`声音模式/图像模式`（标准/影院/音乐/体育/鲜艳）+ 设置 → mode_control
  （落地映射 object= `声音模式`/`图像模式`）；`打开音乐模式`→`{设置, , 音乐模式}`；
  `声音效果类型/音效类型`→音效模式；`开AI画质`→`{设置,ai画质}`。
- **深嵌套对象参数生成**：`obj_tool_gen.py` 用对象词表映射到 display/audio/demo 等工具；
  长尾产品名（量子点/MicroLED/DTS 等）→ 工具 type 判定，参数是深嵌套 generic object，
  `canonical()` 保证 fields 顺序无关。
- 网络：`以太网/网线/有线`→有线网络，`无线/wifi/局域网`→无线网络，`测速`→网络测速；
  路由优先级在 `_net_obj` 内以最短反向顺序固定。

### vod（影视）—— 无规则，LLM 为中心
- **2026-10-27 起去掉全部规则，LLM 为中心**（去规则试点域）。全部逻辑在 `domains/vod/pipeline.py`：
  BM25 top-8 fewshot → 单次 LLM 出 `{tool, params}` → LLM 失败才整句兜底 `vod_fuzzy_search`。
- 没有 badcase 覆盖、精确缓存直出、规则、postproc 改写；确定性代码只解析 JSON、校验工具名、
  补 required 的 retext。`hit_source` 只有 `llm` / `fallback`。
- 样例池 = `testset.json` + `badcases.json`（只作 fewshot 上下文）。评测见 `domains/vod/bench/README.md`。

---

## 6. 目录结构

```
hcTools/
├── app/                  # 共享内核（域无关）
│   ├── domain.py         # Domain 契约 + 加载器（load_domain / load_domains / _load_schema_json）
│   ├── engine.py         # pipeline 钩子 + select→fill 两段式 + L2/L1/LLM/L3 流水
│   ├── llm.py            # 网关异步客户端（文本 JSON 兼容 × function calling）
│   ├── config.py / main.py / models.py / router.py / examples.py
│
├── domains/              # ★ 每个域一个独立文件夹
│   ├── vod/    audio/   education/   sports/
│   ├── music/   children/   device/
│   │   ├── __init__.py      # 导出唯一 Domain 实例（唯一必需入口）
│   │   ├── schema.json       # 工具定义（tool_name / description / parameters）
│   │   ├── textkey.py        # 归一键（L2 badcase 与评测共用，务必同源）
│   │   ├── badcase.py        # BadcaseStore（badcases.json 加载 + 精确查）
│   │   ├── rules.py          # L1 确定性规则（apply）
│   │   ├── dsl.py            # DSL 槽位/词表/生成（嵌套工具域）
│   │   ├── postproc.py       # 参数收尾规范化
│   │   ├── fallback.py       # L3 兜底
│   │   ├── pipeline.py       #（vod）整段接管：fewshot 缓存 + 单次 LLM
│   │   ├── badcases.json        # L2 精确覆盖（raw / tool / params）
│   │   ├── testset.json         # 该域 golden，.records 形如
│   │   ├── obj_tool_gen.py      #（device）tool名→对象→工具映射
│   │   └── bench/
│   │       ├── run_bench.py         # 确定性层打分（对照）
│   │       ├── run_pipeline_eval.py #（vod）真实链路端到端打分 ← 主线
│   │       ├── eval_llm_onecall.py  #（vod）单次 LLM 定标
│   │       ├── README.md            # 本域指标/方法
│   │       └── result.latest.json   # 达标结果
│   ├── extract_schema.py   # schema 抽取（状态机清洗控制字符）
│   └── export_testset.py   # 评测集导出（sheet → testset.json）
│
├── app/ ...（见上）
├── requirements.txt
└── .env.example
```

---

## 7. 共享工具：schema 抽取与测试集导出

### `domains/extract_schema.py` —— 从 Feishu 工具函数表 dump 生成 schema.json

```bash
python -m hcTools.domains.extract_schema <dump.csv> <domain> <domain_key>
```

- 列为 `tool_name / description / parameters` 等；入参的 `parameters` 字段是完整接口描述 →
  内部状态机清洗。
- **状态机 bug fix（关键）**：飞书导出含**真实换行/制表符**嵌在 JSON 字符串值里，直接
  json.loads 会炸。按 `in-string` + 转义状态机把裸 `\n`→`\\n`、`\t`→`\\t`、`<32`控制符→空格。

### `domains/export_testset.py` —— 从 sheet 生成 testset.json

```bash
python -m hcTools.domains.export_testset <sheet.csv> <domain> <domain_key>
```

列：`0=业务域 1=query 2=意图 3=期望工具 4=期望参数`。产出 `.records` golden。

### Feishu 取数

```bash
lark-cli sheets +csv-get --url <wiki_url> --sheet-id <inner> --as user
```

---

## 8. 运行

```bash
cd hcTools
pip install -r requirements.txt          # fastapi / pydantic / httpx / jieba …
uvicorn app.main:app --reload          # LLM 服务（当确定性层漏选时才走 LLM，见 engine.py）
```

本地全量校验（不依赖 LLM，纯确定性层）：

```bash
for d in audio education sports music children device; do   # vod 无确定性层，用 run_pipeline_eval.py
  echo "== $d =="; python domains/$d/bench/run_bench.py
done
```

单域加载冒烟：

```bash
python -c "
import sys; sys.path.insert(0,'.')
from app.domain import load_domain, load_domains
d = load_domain('device'); print('device tools', len(d.tools))
print('all domains:', list(load_domains()))
"
```

---

## 9. 常见错误与排查

| 症状 | 原因 | 修法 |
|---|---|---|
| `AttributeError: 'list' object has no attribute 'search'` | 把一个正则 `re.compile` 名误当成 `list` 用（或反之），如 `_CATEGORY` 撞 `_category` | 词表列表用 `_CATEGORIES`，正则单独命名，手动 `for` 迭代 |
| bench 里 golden 期望空 `{}` | 特殊样本：builder 期待非该工具 | 用 `badcases.json` 一条精确覆盖（零泛化） |
| `extract_schema` 解析失败 | CSV 里字符串值嵌真实换行/制表符 | 用状态机把 `\n→\\n`、`\t→\\t` 清洗后再 `json.loads` |
| L1 规则命中带参变 source 变了 | `rule_select` 返回 `(tool, params)` 且 params 非 None → 直接走 general_rule | 只定工具无参 → 让 params=None → 走 rule_llm_fill |
| 一个 badcase 撞坏近一种意图 | `textkey.normalize` 把正/反意图归一成同 key（删了否定词） | 归一宁紧勿松，**别删「不/没」** |
| import 时 `from app.domain import …` 失败 | hcTools 与父目录都没在 sys.path | bench `sys.path.insert(0, hcTools)` + `insert(0, hcTools.parent)` |
| `load_domains()` 返回多一个域 | 依赖式扫描把 `extract_schema.py` 当域（或残留说明文件） | 只保留 __init__.py 的域目录，其余从 `load_domains` 的筛选中断开 |

---

## 结论

核心一句话：**确定性三层兜底 + 精确 L2 badcase 是让多域测试集稳定达标的钥匙**。
把高频、能表达成规则的通配/词表/DSL 下沉到 L1；把不可归约的特例用 L2 精确止血；
把模糊语义留在 LLM；拿不住的用 L3 兜。这样做到列表 `tool+param ≥90%`（多数 100%）
而无需依赖任何模型随机 —— 这比纯 LLM 保障稳定、可复现、可回归。