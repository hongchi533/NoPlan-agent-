"""Agent 工具集

工具分两类：
  基础工具 — 确定性计算，无 LLM 调用，纯 DB 操作或数学计算
  任务工具 — 委托给 sub-agent，内含独立 LLM 推理

InteractiveAgent 同时持有两类工具；
BackgroundAgent 只使用基础工具。
"""
from datetime import date, datetime, timedelta
import logging
from typing import List, Optional, Dict, Any

from app.agent.parser import ParserAgent
from app.memory import embeddings
from app.memory.retrieval import MemoryRetrieval, get_last_year_today, get_time_capsule_candidates
from app.memory.decay import recompute_all_strengths
from app.store import db
from app.models.schemas import Event, MoodRecord, Preference, MOOD_EMOJI

logger = logging.getLogger(__name__)

# 星期几注入工具返回（同 interactive 的理由：日期能算，星期不能让模型猜——
# 实测不注入时它瞎编）
WEEKDAY_NAMES = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
WEEKDAY_SHORT = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]  # 周期任务描述用："每周三"比"每星期三"顺口

# ═══════════════════════════════════════════════════════
#  Sub-agent 实例（任务工具委托的目标）
# ═══════════════════════════════════════════════════════

parser_agent = ParserAgent()
retrieval_agent = MemoryRetrieval()

# ═══════════════════════════════════════════════════════
#  1. Function Calling 定义（OpenAI 标准格式）
# ═══════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    # ─── 任务工具（含 sub-agent LLM 推理）───
    {
        "type": "function",
        "function": {
            "name": "parse_and_record",
            "description": "解析一句话并存储为生活记录：已做的事(done)、计划的事(plan)、心情(mood)，一句话含多项可同时记录。输入可以是用户原话，也可以是你替用户排好、带具体时间的完整安排（如“明天 17:00-18:00 美容护肤”）。适用：用户陈述生活内容要记录时，以及你完成时间规划后要把排好的安排落库时。用户要修改/挪动已记录的事件时不要用本工具（会新建出重复条目），改用 update_event。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要记录的文本：用户原话，或你排好时间的完整安排",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "语义搜索记忆：按内容/主题/人物/模糊时间检索生活记录，返回记录素材和使用说明（由你组织成回答）。适用于：没有具体指某一天的回忆，如'我上个月练过吉他吗''最近状态怎么样'。query 传用户原话、不要提炼成关键词（'上次/什么时候'这类措辞承载提问意图）。用户提到任何时间表述（上上周/上上个月/去年国庆/半年前/上周到这周）时，先按今天换算成绝对日期范围传 date_from/date_to，换算不了或纯内容回忆就不传。注意：用户问具体某一天干了什么时改用 get_overview。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户原话照传（含'上次/什么时候'等措辞）；时间换算只走 date 参数",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "时间范围起始日 YYYY-MM-DD（含端点），由用户的时间表述按今天换算；无时间表述不传",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "时间范围结束日 YYYY-MM-DD（含端点）；只记得大概起点时与 date_from 传同一天",
                    }
                },
                "required": ["query"],
            },
        },
    },
    # ─── 基础工具（确定性计算，无 LLM）───
    {
        "type": "function",
        "function": {
            "name": "get_overview",
            "description": "获取某天的日程总览，包括事件和心情。适用于：用户想看具体某天或今天的安排。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_date": {
                        "type": "string",
                        "description": "目标日期，YYYY-MM-DD 格式，null 表示今天",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_last_year",
            "description": "获取去年今天的记忆记录。适用于：用户问去年今天做了什么，或打开 app 时主动推送。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_capsule",
            "description": "获取时光胶囊：自动选取衰减到唤起区间的旧记忆，适合重新提醒用户。适用于：有值得唤起的旧记忆时。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_plan",
            "description": "按名称查找事件并返回带 event_id 的列表，默认只找未完成计划(plan)。适用于：拿 plan 的 event_id 调 complete_plan；要修改事件时（含已完成）传 include_done=true 拿 id 再调 update_event。查找范围是单日、默认今天——先结合对话上下文想清楚目标日期再传 target_date。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "事件名称关键词，如'跑步''周报'",
                    },
                    "target_date": {
                        "type": "string",
                        "description": "查找日期，YYYY-MM-DD 格式，null 表示今天",
                    },
                    "include_done": {
                        "type": "boolean",
                        "description": "是否包含已完成(done)事件，默认 false",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_plan",
            "description": "把一个计划(plan)标记为已完成(done)。必须先通过 find_plan 获取 event_id，再调用本工具。不要凭空编造 event_id。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "要完成的事件 ID（由 find_plan 返回）",
                    }
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_event",
            "description": "增量修改已记录事件的字段：只改传入的字段，未传的保持原样。适用于用户想调整、挪动已记下的事——如'改到三点''挪到下周三''备注改成跟李总'。event_id 必须来自 find_plan（结合上下文传 target_date 定位日期），不要凭空猜。与近邻工具分工：记录新内容/新日程 → parse_and_record（本工具不新建）；把计划标记完成 → complete_plan。",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "要修改的事件 ID（由 find_plan 返回）",
                    },
                    "title": {"type": "string", "description": "新标题（不传=不改）"},
                    "date": {"type": "string", "description": "新日期 YYYY-MM-DD；相对时间（下周三/后天）先换算成绝对日期再传"},
                    "start_time": {"type": "string", "description": "新开始时刻 HH:MM（24小时制）"},
                    "end_time": {"type": "string", "description": "新结束时刻 HH:MM（24小时制）"},
                    "note": {"type": "string", "description": "新备注（整体替换原备注）"},
                    "tags": {
                        "type": "array", "items": {"type": "string"},
                        "description": "新标签列表（整体替换），从 工作/生活/运动/社交/饮食/学习/娱乐/家务/惊喜 中选",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consolidate",
            "description": "整理记忆：重算所有记忆的衰减强度。适用于：夜间整理或用户主动触发。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_task",
            "description": "注册定时任务。两种模式：mode=direct 到点提醒用户一件事（text=提醒内容，如'带酒'）；mode=agent 到点替用户执行一件事（prompt=自然语言委托，如'根据今天的天气安排出门计划'）。周期支持每天/每周/每月/每年。相对时间（明早/三天后）必须先换算成绝对日期时间再传参。注意：单纯记录日程（'明天九点开会'）不要用本工具——日程自带开始前 15 分钟的提醒，重复注册会提醒两次。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["direct", "agent"],
                        "description": "direct=到点提醒 / agent=到点执行",
                    },
                    "schedule_type": {
                        "type": "string",
                        "enum": ["once", "daily", "weekly", "monthly", "yearly"],
                        "description": "once=一次性 / daily=每天 / weekly=每周 / monthly=每月 / yearly=每年",
                    },
                    "at": {
                        "type": "string",
                        "description": "once 必填：绝对时间，ISO 格式 YYYY-MM-DDTHH:MM:SS",
                    },
                    "hour": {
                        "type": "integer",
                        "description": "daily/weekly/monthly/yearly 必填：小时 0-23",
                    },
                    "minute": {
                        "type": "integer",
                        "description": "选填：分钟 0-59，默认 0",
                    },
                    "weekday": {
                        "type": "integer",
                        "description": "weekly 必填：0=周一 … 6=周日",
                    },
                    "day": {
                        "type": "integer",
                        "description": "monthly/yearly 必填：几号 1-31（当月没有 31 号时自动按月末算）",
                    },
                    "month": {
                        "type": "integer",
                        "description": "yearly 必填：月份 1-12",
                    },
                    "text": {
                        "type": "string",
                        "description": "direct 模式必填：到点要提醒的内容",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "agent 模式必填：到点执行的自然语言委托",
                    },
                },
                "required": ["mode", "schedule_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "列出用户注册的所有定时任务（到点提醒和到点执行），含任务 ID 与下次触发时间。用户想查看或取消定时任务时先用这个。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_task",
            "description": "取消一个定时任务（用 list_tasks 拿到的 task_id）。一次性任务触发后会自动消失，无需取消。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "任务 ID（task_ 开头，由 register_task/list_tasks 提供）",
                    }
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "holiday_info",
            "description": "查中国法定节假日。三种用法：不传参数=下一个假期是哪个、还有几天、放几天；"
                           "传 name=查某个节日的放假安排（如'国庆''五一'，返回起止日期和天数）；"
                           "传 date=查某一天是放假/调休上班/普通日子（YYYY-MM-DD）。"
                           "数据只覆盖到国务院已公布安排的年份。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "节日名：元旦/春节/清明/劳动节(五一)/端午/中秋/国庆(十一)",
                    },
                    "date": {
                        "type": "string",
                        "description": "要查的日期 YYYY-MM-DD",
                    },
                },
                "required": [],
            },
        },
    },
]

# ═══════════════════════════════════════════════════════
#  2. 工具分类
# ═══════════════════════════════════════════════════════

# 任务工具：委托给 sub-agent，内含独立 LLM 推理，需要 async
TASK_TOOLS = {"parse_and_record"}

# 基础工具：确定性计算，无内部 LLM，纯 DB/检索操作
# （search_memory 2026-09-04 起无内部 LLM：粗筛+向量选材，素材直给交互层叙述；
#   register/list/delete_task 是文件操作，无内部 LLM，同样归基础工具）
BASIC_TOOLS = {"get_overview", "recall_last_year", "recall_capsule", "find_plan", "complete_plan", "consolidate",
               "register_task", "list_tasks", "delete_task", "holiday_info", "search_memory"}

# 会改写存储的工具（跨 TASK/BASIC）：诚实不变量的判断依据——
# 这类工具全失败时，LLM 的"已记录"式回复必为幻觉，agent loop 会覆写为失败话术
WRITE_TOOLS = {"parse_and_record", "complete_plan", "update_event", "consolidate", "register_task", "delete_task"}

# 需要异步执行的工具 = 全部任务工具（内部有 async LLM 调用）
#   + search_memory（向量查询走 embedding API；无 LLM 但仍是 async）
# 注意是副本：动态注册（MCP 等）会往 ASYNC_TOOLS 加异步工具，不能与 TASK_TOOLS 共享同一 set
ASYNC_TOOLS = set(TASK_TOOLS) | {"search_memory"}

# ═══════════════════════════════════════════════════════
#  3. 工具执行函数
# ═══════════════════════════════════════════════════════

# ─── 任务工具实现 ────────────────────────────────────

async def parse_and_record(text: str) -> str:
    """[任务工具] 解析并记录 → 委托 ParserAgent(sub-agent)"""
    logger.info(f"[task-tool] parse_and_record 委托 ParserAgent 解析: {text[:50]}")
    items = await parser_agent.parse_and_create(text)
    if not items:
        # 空结果也是信号：不返回空字符串（主 LLM 会失去判断依据），
        # 明说"没识别出内容"，它会自然地引导用户补充信息
        return "没有从这句话里识别出可记录的内容（事件/心情/偏好都没有）"
    summaries = []
    embedded_items = []   # 落库成功的新记录 → 向量补算（开关关时 ingest 内部直接返回）

    for item in items:
        if isinstance(item, Preference):
            # 用户显式偏好：同 pattern 已存在则更新，否则新增
            existing = db.load_preferences()
            found = next((p for p in existing if p.pattern == item.pattern), None)
            if found:
                db.update_preference(found.id, {
                    "rules": item.rules, "source": "manual",
                    "source_text": item.source_text, "updated_at": item.updated_at,
                })
                summaries.append(f"已更新偏好：{item.pattern} → {item.rules}")
            else:
                db.add_preference(item)
                summaries.append(f"已记录偏好：{item.pattern} → {item.rules}")
        elif isinstance(item, MoodRecord):
            saved = db.add_mood(item)
            embedded_items.append(saved)
            emoji = MOOD_EMOJI.get(saved.mood, "😐")
            summaries.append(f"已记录心情：{emoji} {saved.content or ''}（{saved.date}）")
        else:
            # 去重：同日期+同标题+同类型+同开始时间
            #   结束时间相同 → 完全重复，跳过
            #   结束时间不同 → 用户在修正时长，更新已有事件
            dup = _find_duplicate(item)
            if dup:
                if item.end_time and item.end_time != dup.end_time:
                    db.update_event(dup.id, {"end_time": item.end_time})
                    summaries.append(f"已更新时长：{dup.title} {dup.start_time}~{item.end_time}（原 {dup.end_time or '未指定'}）")
                    logger.info(f"[dedup] 更新事件时长: {dup.title} {dup.start_time} {dup.end_time} → {item.end_time}")
                else:
                    time_str = f" {dup.start_time}" if dup.start_time else ""
                    summaries.append(f"跳过重复事件：{dup.title}（{dup.date}{time_str} 已记录过）")
                    logger.info(f"[dedup] 跳过重复事件: {dup.title} {dup.date} {dup.start_time}")
                continue
            saved = db.add_event(item)
            embedded_items.append(saved)
            type_label = "计划" if saved.type == "plan" else "已完成"
            time_str = f" {saved.start_time}" if saved.start_time else ""
            tags_str = "、".join(saved.tags) if saved.tags else ""
            summaries.append(f"已记录事件：{type_label} | {saved.date}{time_str} {saved.title} [{tags_str}]")
            # 时间重叠：正常记录不受影响，只把"撞车"事实带回给主 LLM——
            # 提不提、怎么提的基调由 SYSTEM_PROMPT 定（轻度提醒，不说教）
            conflicts = _find_time_conflicts(saved)
            if conflicts:
                c = conflicts[0]
                c_time = f"{c.start_time}~{c.end_time}" if c.end_time else c.start_time
                more = f"（另有 {len(conflicts) - 1} 条同时段）" if len(conflicts) > 1 else ""
                summaries.append(f"⚠️ 时间重叠：{saved.start_time} 与 {c_time} 的「{c.title}」撞车{more}")
            # 不匹配检测：用户行为与 auto 偏好不符 → 计数
            _check_preference_mismatch(saved)

    # 向量召回补算：记录已安全落库，这里失败只 log 不重试（n-gram 旁路兜住，
    # 下次回填对齐）。去重跳过/改时长的旧事件向量暂不重算，同由回填收敛
    if embedded_items:
        await embeddings.ingest(embedded_items)

    return "\n".join(summaries)


def _find_duplicate(event: Event) -> Optional[Event]:
    """查找重复事件：同日期 + 同标题 + 同类型 + 同开始时间"""
    existing = db.get_events_by_date(event.date)
    for e in existing:
        if (e.title == event.title
                and e.type == event.type
                and e.start_time == event.start_time):
            return e
    return None


def _find_time_conflicts(event: Event) -> List[Event]:
    """查找同日时间重叠的其他事件（轻度提醒用，宁缺勿滥：只报真实区间相交）

    无 start_time 的不参与（连几点都不知道，谈不上撞）；
    end_time 缺省按瞬时处理，不虚构时长——误报的提醒比没有提醒更烦人。
    """
    if not event.start_time:
        return []
    conflicts = []
    for e in db.get_events_by_date(event.date):
        if e.id == event.id or not e.start_time:
            continue
        # "HH:MM" 定长格式，字符串比较即时间比较
        a_end = event.end_time or event.start_time
        b_end = e.end_time or e.start_time
        # 严格相交（端点相接不算撞：13:00~14:00 接 14:00 是先后衔接）
        # + 同刻必报（两点开会/两点吃药——与前端日视图"同刻并列"同一语义）
        if (event.start_time < b_end and e.start_time < a_end) or event.start_time == e.start_time:
            conflicts.append(e)
    return conflicts


def _check_preference_mismatch(event: Event):
    """检测新记录的事件是否与已有偏好不符，不符则累计 mismatch，达 3 次删除偏好"""
    prefs = db.load_preferences()
    for p in prefs:
        if p.pattern not in (event.title or ""):
            continue
        mismatch = False
        pref_time = p.rules.get("default_time")
        if pref_time and event.start_time and event.start_time != pref_time:
            mismatch = True
        # 标签不参与错配判定：标签是解析器自己的语义分类，与偏好无关——
        # 若参与，解析器自选的标签会被当成"偏好失效"计数，3 次后误删偏好
        if not mismatch:
            continue
        count = p.evidence.get("recent_mismatches", 0) + 1
        if count >= 3:
            db.delete_preference(p.id)
            logger.info(f"[pref-lifecycle] 偏好 '{p.pattern}' 连续{count}次不匹配，已删除")
        else:
            db.update_preference(p.id, {
                "evidence": {"recent_mismatches": count},
                "updated_at": event.created_at,
            })
            logger.info(f"[pref-lifecycle] 偏好 '{p.pattern}' 不匹配 {count}/3 次")


# ─── 基础工具实现 ────────────────────────────────────

async def search_memory(query: str, date_from: str = "", date_to: str = "") -> str:
    """[基础工具] 语义搜索记忆 → 检索管线取素材（无内部 LLM），素材直给交互 agent 叙述

    date_from/date_to：交互层把相对时间表述（上上周/去年国庆/半年前）按今天
    换算成的绝对范围；不传时 retrieval 用规则表兜底
    """
    logger.info(f"[search] search_memory 检索素材: {query[:50]} | 时间参数: {date_from or '-'}~{date_to or '-'}")
    material = await retrieval_agent.search(query, date_from, date_to)
    # 素材全文进日志：叙述层出问题时，审计"叙述者当时看到了什么"靠这条
    # （2026-09-04 看电视事故能定位，就是因为候选列表在日志里可见）
    logger.info(f"[search] 素材 {len(material)}字:\n{material}")
    return material


def get_overview(target_date: Optional[str] = None) -> str:
    """[基础工具] 获取总览，纯 DB 查询 + 格式化"""
    if target_date is None:
        target_date = date.today().isoformat()

    events = db.get_events_by_date(target_date)
    moods = db.get_moods_by_date(target_date)

    # 星期随日期注入：LLM 复述"周几的安排"、推算"下周三"时不用再猜
    weekday = WEEKDAY_NAMES[date.fromisoformat(target_date).weekday()]
    lines = [f"📅 {target_date} {weekday} 日程："]

    if not events and not moods:
        lines.append("  暂无记录")
    else:
        for e in events:
            check = "✓" if e.type == "done" else "○"
            if e.start_time and e.end_time:
                time_str = f"{e.start_time}-{e.end_time}"   # 带时段：规划者才知道占用到几点
            else:
                time_str = e.start_time or "-"
            lines.append(f"  {check} {time_str} {e.title} {' '.join(e.tags)}")
        for m in moods:
            emoji = MOOD_EMOJI.get(m.mood, "😐")
            lines.append(f"  {emoji} {m.content or ''}")

    return "\n".join(lines)


def recall_last_year() -> str:
    """[基础工具] 去年今天，纯 DB 查询"""
    items = get_last_year_today()
    if not items:
        return "去年今天没有记录"
    lines = ["📅 去年今天："]
    for i in items:
        if isinstance(i, Event):
            time_str = f" {i.start_time}" if i.start_time else ""
            lines.append(f"  {i.date}{time_str} {i.title}")
        else:
            emoji = MOOD_EMOJI.get(i.mood, "😐")
            lines.append(f"  {i.date} {emoji} {i.content or ''}")
    return "\n".join(lines)


def recall_capsule() -> str:
    """[基础工具] 时光胶囊，纯 DB 查询 + 衰减计算"""
    items = get_time_capsule_candidates()
    if not items:
        return "没有需要唤起的记忆（唤起窗内暂无候选，或最近的都已展示过）"
    lines = ["⏳ 时光胶囊："]
    for i in items:
        days = (date.today() - date.fromisoformat(i.date)).days
        lines.append(f"  {days}天前: {i.title}")
    return "\n".join(lines)


def find_plan(keyword: str, target_date: Optional[str] = None, include_done: bool = False) -> str:
    """[基础工具] 按名称模糊查找事件，返回带 event_id 的列表（默认只找未完成 plan）"""
    if target_date is None:
        target_date = date.today().isoformat()

    events = db.get_events_by_date(target_date)
    plans = [e for e in events if (include_done or e.type == "plan") and keyword in e.title]

    if not plans:
        what = "事件" if include_done else "未完成计划"
        return f"未找到包含'{keyword}'的{what}，请明确指出任务时间或换关键词"

    lines = [f"找到 {len(plans)} 个匹配的事件："]
    for p in plans:
        time_str = f" {p.start_time}" if p.start_time else ""
        tags_str = " ".join(p.tags) if p.tags else ""
        lines.append(f"  id={p.id} | {p.date}{time_str} {p.title} [{tags_str}]")

    return "\n".join(lines)


def complete_plan(event_id: str) -> str:
    """[基础工具] 完成计划，纯 DB 更新"""
    event = db.update_event(event_id, {"type": "done"})
    if event is None:
        return "事件不存在，无法完成"
    return f"已完成：{event.title}"


def update_event(event_id: str, title: str = None, date: str = None,
                 start_time: str = None, end_time: str = None,
                 note: str = None, tags: list = None) -> str:
    """[基础工具] 增量修改事件：只改传入的字段，未传的保持原样

    兜底与新建同款：格式校验（坏日期/时刻会毒化前端渲染）、未来无已完成
    （done 挪去未来=数据失真，改前拒绝）、挪动后撞车检测带 ⚠️ 回给主 LLM；
    挪动也是行为证据，喂偏好淘汰计数。参数名 date 遮蔽了模块级 datetime.date，
    函数内取"今天"用 datetime 类（两者同源，等价）。
    """
    updates = {k: v for k, v in {
        "title": title, "date": date, "start_time": start_time,
        "end_time": end_time, "note": note, "tags": tags,
    }.items() if v is not None}
    if not updates:
        return "没有要修改的字段：至少传一个（标题/日期/时刻/备注/标签）"

    # 格式校验：LLM 违反描述传了相对时间时，这里挡在落库前（坏字符串会让前端
    # date.fromisoformat 崩掉整页渲染）
    try:
        if "date" in updates:
            datetime.strptime(updates["date"], "%Y-%m-%d")
        for key in ("start_time", "end_time"):
            if key in updates:
                datetime.strptime(updates[key], "%H:%M")
    except ValueError as e:
        return f"格式不对（{e}）：date 要 YYYY-MM-DD、时刻要 HH:MM，相对时间先换算成绝对值再传"

    old = next((e for e in db.load_events() if e.id == event_id), None)
    if old is None:
        return "事件不存在，无法修改（event_id 需来自 find_plan 返回）"

    # 未来无已完成（parser 同款不变量）：done 挪到未来不是修改是失真
    new_date = date or old.date
    if old.type == "done" and new_date > datetime.today().strftime("%Y-%m-%d"):
        return f"没改：已完成的事不能挪到未来（{new_date}）。要改日期的话，先确认这件事其实还没做"

    event = db.update_event(event_id, updates)
    changed = "，".join(f"{k} → {v}" for k, v in updates.items())
    summary = f"已修改「{event.title}」：{changed}"
    conflicts = _find_time_conflicts(event)
    if conflicts:
        c = conflicts[0]
        c_time = f"{c.start_time}~{c.end_time}" if c.end_time else c.start_time
        more = f"（另有 {len(conflicts) - 1} 条同时段）" if len(conflicts) > 1 else ""
        summary += f"\n⚠️ 时间重叠：与 {c_time} 的「{c.title}」撞车{more}"
    _check_preference_mismatch(event)
    return summary


def consolidate() -> str:
    """[基础工具] 整理记忆衰减，纯计算 + DB 写入"""
    events = db.load_events()
    moods = db.load_moods()
    events, moods = recompute_all_strengths(events, moods)
    db.save_events(events)
    db.save_moods(moods)
    return f"整理完成：{len(events)} 条事件，{len(moods)} 条心情"


# ─── 定时任务工具实现（统一调度器的对话注册通道）─────
# 调度器实例由 main.py 接线时注入（set_task_scheduler）：tools 不 import scheduler，
# 避免 import 期互相拉起；未注入时返回错误串，agent 会如实告知用户

_task_scheduler = None


def set_task_scheduler(scheduler) -> None:
    """main.py 启动时注入统一调度器实例（依赖注入，防 import 环）"""
    global _task_scheduler
    _task_scheduler = scheduler


def register_task(mode: str, schedule_type: str = "once", at: str = "",
                  hour: int = None, minute: int = 0, weekday: int = None,
                  day: int = None, month: int = None,
                  text: str = "", prompt: str = "") -> str:
    """[基础工具] 注册定时任务：direct=到点提醒（零 LLM） / agent=到点起 loop 执行委托"""
    if _task_scheduler is None:
        return "错误：调度器未就绪，暂时无法注册定时任务"
    if schedule_type == "once":
        schedule = {"type": "once", "at": at}
    elif schedule_type == "daily":
        schedule = {"type": "daily", "hour": int(hour), "minute": int(minute or 0)}
    elif schedule_type == "weekly":
        schedule = {"type": "weekly", "weekday": int(weekday), "hour": int(hour), "minute": int(minute or 0)}
    elif schedule_type == "monthly":
        schedule = {"type": "monthly", "day": int(day), "hour": int(hour), "minute": int(minute or 0)}
    elif schedule_type == "yearly":
        schedule = {"type": "yearly", "month": int(month), "day": int(day),
                    "hour": int(hour), "minute": int(minute or 0)}
    else:
        return f"错误：不支持的 schedule_type {schedule_type}"
    payload = {"text": text.strip()} if mode == "direct" else {"prompt": prompt.strip()}
    try:
        task = _task_scheduler.register_user_task(mode, schedule, payload)
    except (ValueError, TypeError) as e:
        return f"错误：{e}"
    # 面向 LLM 的回执：时间说人话，ID 留给它日后取消用
    when = task["schedule"].get("at") or _describe_schedule(task["schedule"])
    kind = "到点提醒" if mode == "direct" else "到点执行"
    if task.get("_existing"):
        return f"该定时任务已存在（{kind}）：{when} · {text.strip() or prompt.strip()}（ID: {task['id']}），未重复创建"
    return f"已注册定时任务（{kind}）：{when} · {text.strip() or prompt.strip()}（ID: {task['id']}）"


def _describe_schedule(sched: dict) -> str:
    hm = f"{sched['hour']:02d}:{sched.get('minute', 0):02d}"
    if sched.get("type") == "daily":
        return f"每天 {hm}"
    if sched.get("type") == "weekly":
        return f"每{WEEKDAY_SHORT[sched['weekday']]} {hm}"
    if sched.get("type") == "monthly":
        return f"每月{sched['day']}日 {hm}"
    if sched.get("type") == "yearly":
        return f"每年{sched['month']}月{sched['day']}日 {hm}"
    return str(sched)


def user_tasks_raw() -> list:
    """结构化任务列表（background 取晨报素材用；对话面请用 list_tasks 的成串文本）"""
    if _task_scheduler is None:
        return []
    return _task_scheduler.list_user_tasks()


def list_tasks() -> str:
    """[基础工具] 列出用户注册的定时任务（提醒页与对话看到的是同一份 tasks.json）"""
    if _task_scheduler is None:
        return "错误：调度器未就绪"
    tasks = _task_scheduler.list_user_tasks()
    if not tasks:
        return "目前没有注册任何定时任务"
    lines = []
    for t in tasks:
        icon = "🔔" if t["kind"] == "remind" else "🤖"
        body = t["payload"].get("text") or t["payload"].get("prompt") or ""
        lines.append(f"{icon} [{t['id']}] {body} · 下次触发 {t.get('next_fire') or '未知'}")
    return "\n".join(lines)


def delete_task(task_id: str) -> str:
    """[基础工具] 取消定时任务。一次性任务触发即删——取消和完成是同一个动作"""
    if _task_scheduler is None:
        return "错误：调度器未就绪"
    deleted = _task_scheduler.delete_user_task((task_id or "").strip())
    if deleted is None:
        return f"错误：找不到任务 {task_id}（可能已触发自动移除，或已删除过）"
    body = deleted["payload"].get("text") or deleted["payload"].get("prompt") or ""
    return f"已取消定时任务：{body}（{task_id}）"


def holiday_info(name: str = "", date_str: str = "") -> str:
    """[基础工具] 法定节假日查询（chinesecalendar 离线数据，无网络无 key）

    三种问法互斥：name（查某节日安排）> date（查某天状态）> 都不传（下一个假期）
    """
    from datetime import datetime as _dt
    from app.holidays import describe_next, describe_date, describe_festival
    if (name or "").strip():
        return describe_festival(name)
    if not (date_str or "").strip():
        return describe_next()
    try:
        d = _dt.strptime(date_str.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return f"日期格式不对：{date_str}（要 YYYY-MM-DD）"
    return describe_date(d)


# ═══════════════════════════════════════════════════════
#  4. 工具名 → 函数映射
# ═══════════════════════════════════════════════════════

TOOL_FUNCTION_MAP = {
    # 任务工具
    "parse_and_record": parse_and_record,
    "search_memory": search_memory,
    # 基础工具
    "get_overview": get_overview,
    "recall_last_year": recall_last_year,
    "recall_capsule": recall_capsule,
    "find_plan": find_plan,
    "complete_plan": complete_plan,
    "update_event": update_event,
    "consolidate": consolidate,
    # 定时任务三件套（统一调度器的注册通道）
    "register_task": register_task,
    "list_tasks": list_tasks,
    "delete_task": delete_task,
    # 法定节假日（app/holidays.py 离线数据）
    "holiday_info": holiday_info,
}


# ═══════════════════════════════════════════════════════
#  4. 动态工具注册（MCP 等外部工具，启动时挂载）
# ═══════════════════════════════════════════════════════

def register_dynamic_tool(name: str, description: str, parameters: Dict[str, Any], func) -> None:
    """注册外部动态工具：原地并入内置注册表，InteractiveAgent 无感知。

    - parameters 传 MCP 的 inputSchema（JSON Schema，与 OpenAI function calling 格式同构）
    - func 必须是 async callable（MCP 转发涉及网络 IO）
    """
    TOOL_DEFINITIONS.append({
        "type": "function",
        "function": {
            "name": name,
            "description": description or "",
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    })
    TOOL_FUNCTION_MAP[name] = func
    ASYNC_TOOLS.add(name)
    logger.info(f"[tools] 动态工具已注册: {name}（工具总数 {len(TOOL_DEFINITIONS)}）")
