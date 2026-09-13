"""BackgroundAgent：后台 Agent

自主执行任务，不响应对话。触发方式：统一调度器（app/scheduler.py）到点 fire 或手动触发。

三种执行形态（调度器的 handler 注册表对应到这里）：
  run_nightly   纯代码——记忆维护（衰减重算 + 偏好发现），无 LLM
  run_overview  编排式——双轨产出：横幅版（今日日程/天气/去年今天，7 点快照，进信箱）
                + 栏目版（同看今日日程——空则整块缺省不提；另有本周/近几天事件
                提醒/节日素材，落盘首页卡片）
                （"总能先查天气再总结"是编排保证的，不是 prompt 求来的）
  run_weekly    编排式——同上，7 天事件的周总结
  run_task      agent 式——无状态 loop 执行用户的自然语言委托（"明早七点根据天气
                定出门计划"），fire 时才解释意图，工具副作用（建 plan）由 loop 内现有工具完成

与 InteractiveAgent 的关系：两种 persona 共享同一工具注册表（tools.py 的能力层），
但 loop 本体刻意不抽公共基类——交互版要会话记忆/状态机/降级文案，后台版要无状态/短促，
真正共享的只有 ~10 行"执行 tool_call"机械代码，其余是应当分化的策略。
"""
import json
import logging
import uuid
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional

from app.agent.base import BaseAgent
from app.config import DASHSCOPE_ENABLE_THINKING, BRIEFING_CITY
from app.memory.decay import recompute_all_strengths
from app.memory.retrieval import get_last_year_today, get_time_capsule_candidates
from app.store import db
from app.models.schemas import Preference, Event, MOOD_EMOJI

logger = logging.getLogger(__name__)

# 偏好自动发现的阈值
MIN_PATTERN_COUNT = 6    # 至少出现 6 次
MIN_CONFIDENCE = 0.7     # 众数占比至少 70%

WEEKDAY_NAMES = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

# 后台 agent loop 的系统提示：通知形态的输出约束是它和交互版最大的 persona 差异
TASK_SYSTEM_PROMPT = """你是记忆助手的后台执行体，在无人值守的时刻被定时任务唤醒，替用户执行一条事先委托的自然语言指令。

今天是 {today} {weekday}，现在 {now}（相对时间如"1小时后"以此刻为基准换算）。

原则：
- 先调用需要的工具收集事实（查天气用 maps_weather、查安排用 get_overview），再下结论
- 需要落地的安排（如"定一个出门计划"）直接用 parse_and_record 记录，记完在汇报里说明
- 你的最终回复会作为通知送达用户：开门见山说结论，三句话内说完
- 用户此刻不在场：不要寒暄、不要提问、不要确认，做了什么、结论是什么，说完即止
- 只做指令要求的事
"""

OVERVIEW_SYSTEM_PROMPT = """你是记忆助手的晨间播报员。根据提供的今日日程、天气、去年今天，写一段晨间总览。
要求：先一句话天气，再今日安排提要（按时间顺序），最后可以给一条轻描淡写的建议。
全文不超过 120 字，不制造焦虑，像朋友顺口一提。素材里没有的部分直接跳过，不要编造。"""

# 栏目版（首页卡片，挂一整天）：与横幅同看今日日程，空则默认忽略；主题是今天，过去未来只轻点
OVERVIEW_CARD_PROMPT = """你是记忆助手，是和用户很亲的可爱朋友，每天为首页「今日总览」卡片写一段话（它会挂一整天）。
提供的素材里有天气、今日日程、去年今天、本周记录、过去/未来几天的事、近几天的提醒、临近的节假日。
主题永远是今天：先一句话天气，再今日提要；不逐条复述日程（那是日历的活）。
素材里没有今日日程就直接跳过，绝不提"没有安排"这类话。
去年今天、本周记录、过去/未来几天的事、近几天的提醒、临近的节假日是备选素材，
所有备选素材加起来可以挑最值得说的轻轻提一下，没有值得说的就整块不提。
日程与记录素材每行都标着"计划"或"已完成"：计划是将来的事，只能朝前带一句期待，
绝不能写成已发生的样子；写出的每件事都要能在素材里找到对应的那一行，素材里没有的经历一概不提。
感受是用户自己的事：不臆测他的心情、状态（"余韵犹在""心情不错"都是编造）。
你和用户是平等的朋友，不是教练、评委或家长——这条高于一切：
他的生活过得怎么样，轮不到你打分。"节奏很棒""你做得不错"
这类武断的话无论夸还是贬，都是站在上位评价他，一条都不许出现；
你只陈述事实、陪伴，朝前看可以给温柔的安抚或建议（"早点休息"可以，"做得真好"不行），
也可以顺着事实往前送祝愿（"周末散了步，希望你今天更轻松"）。
语气要像可爱的朋友聊天，不制造焦虑、不对任何事做任何点评或评价，不刻意迎合、不渲染恐慌，
全文不超过 120 字，一段话，不提问，素材里没有的部分直接跳过，不要编造。"""

# 回忆卡片（时光胶囊/去年今天）：与总览同批的第三路组稿产物。
# 选材归代码（衰减唤起窗 + 冷却轮换），叙述归 LLM——素材空洞时它有权整段不说
PROACTIVE_CARD_PROMPT = """你是记忆助手，和用户很亲的朋友，为首页两张回忆卡片各写一小段文案。
「去年今天」的素材是一年前这一天的记录；「时光胶囊」的素材是隔了很久、刚好到该想起的时候的旧事（每条标着多少天前）。
两段各自独立：每段一两句话，把记录里具体的细节自然带出来，像朋友顺口想起一件旧事，说完就停。
写不出有温度的话就交白卷：素材空洞的那段返回空串，宁缺勿滥。
你们是平辈朋友：不点评不夸奖（"很棒""厉害"都是打分），记录里没写的情绪不替他臆测，素材里没有的细节一概不添。
只输出 JSON：{"last_year": "去年今天的文案或空串", "capsule": "时光胶囊的文案或空串"}"""

WEEKLY_SYSTEM_PROMPT = """你是记忆助手的周报播报员。根据提供的近 7 天日程与心情统计，写一段周总结。
要求：概括这周的主线（做了什么类型的事）、点一两件有代表性的具体事、结尾一句轻柔的回顾。
全文不超过 150 字，语气温暖不评判，没有素材的部分直接跳过，不要编造。"""

# 后台 loop 轮数上限：与交互版相同（合法的顺序多工具链需要余量）
TASK_MAX_ITERATIONS = 8
# 工具结果喂回 LLM 的截断长度（结果太长只会稀释指令）
TOOL_RESULT_MAX_CHARS = 2000


class BackgroundAgent(BaseAgent):
    """后台 Agent：调度器到点后的三种执行形态（纯代码 / 编排式 / agent 式）"""

    name = "background-agent"

    # ─── 形态一：纯代码（无 LLM）──────────────────────

    async def run_consolidate(self) -> Dict[str, Any]:
        """执行记忆整理：衰减重算 + 归档"""
        logger.info("[background-agent] 开始记忆整理 consolidate")
        start = datetime.now()

        events = db.load_events()
        moods = db.load_moods()
        events, moods = recompute_all_strengths(events, moods)
        db.save_events(events)
        db.save_moods(moods)

        elapsed = (datetime.now() - start).total_seconds()
        result = {
            "task": "consolidate",
            "events_count": len(events),
            "moods_count": len(moods),
            "elapsed_seconds": elapsed,
        }
        logger.info(f"[background-agent] consolidate 完成: {result}")
        return result

    def discover_preferences(self) -> List[Preference]:
        """从历史事件中自动发现偏好模式

        算法：按 title 分组 → 统计众数 → 次数≥6 且置信度≥0.7 → 生成 auto 偏好
        manual 偏好不受影响
        """
        events = db.load_events()
        groups: Dict[str, List[Event]] = {}
        for e in events:
            if e.title:
                groups.setdefault(e.title, []).append(e)

        now = datetime.now().isoformat()
        existing = db.load_preferences()
        discovered = []

        for title, group in groups.items():
            if len(group) < MIN_PATTERN_COUNT:
                continue

            # 已有 manual 偏好则跳过（manual 优先，不被 auto 覆盖）
            if any(p.pattern == title and p.source == "manual" for p in existing):
                continue

            # 统计众数
            times = [e.start_time for e in group if e.start_time]
            if not times:
                continue

            # 偏好只沉淀默认时刻；标签是解析器的语义判断，不作为偏好（不再产出 default_tag）
            rules = {}
            time_mode, time_count = Counter(times).most_common(1)[0]
            if time_count / len(group) >= MIN_CONFIDENCE:
                rules["default_time"] = time_mode
            if not rules:
                continue

            pref = Preference(
                id=str(uuid.uuid4()),
                pattern=title,
                rules=rules,
                source="auto",
                evidence={"recent_mismatches": 0},
                created_at=now,
                updated_at=now,
            )
            discovered.append(pref)

            # 已有同 pattern 的 auto 偏好 → 更新 rules；否则新增
            found = next((p for p in existing if p.pattern == title), None)
            if found:
                db.update_preference(found.id, {"rules": rules, "updated_at": now})
                logger.info(f"[background-agent] 更新 auto 偏好: {title} → {rules}")
            else:
                db.add_preference(pref)
                logger.info(f"[background-agent] 发现 auto 偏好: {title} → {rules}")

        logger.info(f"[background-agent] 偏好发现完成，共 {len(discovered)} 条")
        return discovered

    async def run_nightly(self) -> Dict[str, Any]:
        """每日夜间任务入口（调度器 builtin:nightly）"""
        logger.info("[background-agent] 开始每日夜间任务")
        consolidate_result = await self.run_consolidate()
        discovered = self.discover_preferences()

        return {
            "nightly": True,
            "consolidate": consolidate_result,
            "preferences_discovered": len(discovered),
        }

    # ─── 形态二：编排式（代码收集 + 一次组稿 LLM）──────

    async def run_overview(self) -> tuple:
        """每日 07:00 晨间总览（调度器 builtin:overview）。双轨产出：
        - 横幅版（返回 [0] → 信箱）：今日日程 + 天气 + 去年今天，空日程也要点出留白
        - 栏目版（返回 [1] → 首页卡片）：与横幅同构（天气后提今日提要），
          差别是今天没日程就整块缺省、连"没有安排"都不说
        "先查天气再总结"的顺序由代码编排保证——不依赖 prompt 自觉。"""
        banner = await self._overview_banner()
        card = ""
        try:
            card = await self._overview_card()
        except Exception as e:
            # 栏目失败不能连累横幅重发（durable 重试会把已送达的横幅再弹一次）：
            # 首页今天没有卡片而已，明天 7 点自然再来
            logger.error(f"[background-agent] 栏目版组稿失败，今日首页无卡片: {type(e).__name__}: {e}")
        return banner, card

    async def _overview_banner(self) -> str:
        """横幅版组稿：今日日程 + 天气 + 去年今天（7 点那一刻的快照，原样保留）"""
        from app.agent.tools import TOOL_FUNCTION_MAP, recall_last_year  # 延迟导入避开环

        today = date.today()
        events = db.get_events_by_date(today.isoformat())

        # 素材一：今日日程
        schedule_lines = [
            f"- {e.start_time or '全天'} {e.title}（{'计划' if e.type == 'plan' else '已完成'}"
            + (f"·{'/'.join(e.tags)}" if e.tags else "") + ")"
            for e in events
        ]
        schedule_text = "\n".join(schedule_lines) or "（今天还没有日程记录）"

        # 素材二：天气（MCP 未挂载/查询失败只降级这一项，不废掉整个总览）
        weather_text = "（今天天气服务不可用）"
        weather_func = TOOL_FUNCTION_MAP.get("maps_weather")
        if weather_func:
            try:
                weather_text = str(await weather_func(city=BRIEFING_CITY))
            except Exception as e:
                logger.warning(f"[background-agent] 晨览天气查询失败，降级跳过: {type(e).__name__}: {e}")
                weather_text = "（今天天气查询失败了）"

        # 素材三：去年今天（没有则不进素材）
        last_year_text = recall_last_year()
        last_year_block = "" if "没有" in last_year_text else f"\n【去年今天】\n{last_year_text}\n"

        prompt = f"""【生成时刻】{today.isoformat()} {WEEKDAY_NAMES[today.weekday()]} {datetime.now().strftime('%H:%M')}
（补跑/手动触发时此刻可能不是早晨——按素材里的时间事实描述，不要硬套晨间视角）

【今日日程】（{today.isoformat()} {WEEKDAY_NAMES[today.weekday()]}）
{schedule_text}

【天气】（{BRIEFING_CITY}）
{weather_text}
{last_year_block}
请写晨间总览。"""
        logger.info(f"[background-agent] 晨览素材就绪（日程 {len(events)} 条 / 天气 {'有' if '不可用' not in weather_text and '失败' not in weather_text else '无'} / 去年今天 {'有' if last_year_block else '无'}），开始组稿")
        return await self.call_llm(prompt, system_prompt=OVERVIEW_SYSTEM_PROMPT)

    async def _overview_card(self) -> str:
        """栏目版组稿：天气/今日日程/去年今天/本周/近几天事件提醒/节日。
        与横幅逻辑同构——天气后提今日提要；差别：今天没有日程就整块缺省，
        连"没有安排"都不说（横幅反而要点出留白）。"""
        from app.agent.tools import TOOL_FUNCTION_MAP, recall_last_year, user_tasks_raw  # 延迟导入避开环
        from app.holidays import upcoming_festivals

        today = date.today()
        blocks = [f"【生成时刻】{today.isoformat()} {WEEKDAY_NAMES[today.weekday()]} "
                  f"{datetime.now().strftime('%H:%M')}"]

        # 素材：天气（与横幅版同源；失败只降级这一项）
        weather_text = "（天气服务不可用）"
        weather_func = TOOL_FUNCTION_MAP.get("maps_weather")
        if weather_func:
            try:
                weather_text = str(await weather_func(city=BRIEFING_CITY))
            except Exception as e:
                logger.warning(f"[background-agent] 栏目天气查询失败，降级跳过: {type(e).__name__}: {e}")
                weather_text = "（天气查询失败了）"
        blocks.append(f"【天气】（{BRIEFING_CITY}）\n{weather_text}")

        # 素材：今日日程（与横幅同看；空则整块不进素材——默认忽略，不提"没有安排"）
        today_events = db.get_events_by_date(today.isoformat())
        today_lines = [
            f"- {e.start_time or '全天'} {e.title}（{'计划' if e.type == 'plan' else '已完成'}）"
            for e in today_events
        ]
        if today_lines:
            blocks.append(f"【今日日程】（{today.isoformat()} {WEEKDAY_NAMES[today.weekday()]}）\n"
                          + "\n".join(today_lines))

        # 素材：去年今天（没有则不进素材）
        last_year_text = recall_last_year()
        if "没有" not in last_year_text:
            blocks.append(f"【去年今天】\n{last_year_text}")

        # 素材：本周记录（周一→昨天；今天的绝不进，给 LLM 看形状，不给它盘点当日的机会）
        monday = today - timedelta(days=today.weekday())
        week_events = [e for e in db.get_events_by_date_range(monday.isoformat(), today.isoformat())
                       if e.date < today.isoformat()]
        week_lines = [
            f"- {e.date} {WEEKDAY_NAMES[date.fromisoformat(e.date).weekday()]} {e.title}"
            f"（{'计划' if e.type == 'plan' else '已完成'}）"
            for e in week_events
        ]
        if week_lines:
            blocks.append("【本周记录（周一至今）】\n" + "\n".join(week_lines))

        # 素材：前后三天的事件，拆成过去/未来两栏——方向写进栏目标题，
        # 比 N 天前/天后的行内括号更抗读串（实测混排时出现过把未来计划织成过去回忆）
        near = [e for e in db.get_events_by_date_range((today - timedelta(days=3)).isoformat(),
                                                       (today + timedelta(days=3)).isoformat())
                if e.date != today.isoformat()]
        past_lines, future_lines = [], []
        for e in near:
            diff = (date.fromisoformat(e.date) - today).days
            rel = f"{-diff} 天前" if diff < 0 else f"{diff} 天后"
            wd = WEEKDAY_NAMES[date.fromisoformat(e.date).weekday()]
            line = f"- {e.date} {wd}（{rel}）{e.title}（{'计划' if e.type == 'plan' else '已完成'}）"
            (past_lines if diff < 0 else future_lines).append(line)
        if past_lines:
            blocks.append("【过去三天】（都是已经过去的日子）\n" + "\n".join(past_lines))
        if future_lines:
            blocks.append("【未来三天】（都是还没到来的日子，只能期待）\n" + "\n".join(future_lines))

        # 素材：未来三天内的定时提醒（今天的任务属当日盘点，不进卡片；过去已触发即消失）
        task_lines = []
        try:
            for t in user_tasks_raw():
                nf = t.get("next_fire") or ""
                body = t.get("payload", {}).get("text") or t.get("payload", {}).get("prompt") or ""
                if not nf[:10] or not body:
                    continue
                diff = (date.fromisoformat(nf[:10]) - today).days
                if 1 <= diff <= 3:   # 未来 1~3 天才有期待感
                    hm = nf[11:16]
                    task_lines.append(f"- {nf[:10]}（{diff} 天后{' ' + hm if hm else ''}）提醒：{body}")
        except Exception as e:
            logger.warning(f"[background-agent] 栏目定时提醒素材获取失败，跳过: {type(e).__name__}: {e}")
        if task_lines:
            blocks.append("【近几天的提醒】\n" + "\n".join(task_lines))

        # 素材：14 天内的法定节假日（数据只到已公布年份，越界静默缺料）
        fests = upcoming_festivals(14)
        if fests:
            fest_lines = []
            for first, cn, span, gap in fests:
                when = "假期进行中" if gap <= 0 else f"还有 {gap} 天"
                fest_lines.append(f"- {cn}：{first.isoformat()} 开始放 {span} 天（{when}）")
            blocks.append("【临近的节假日】\n" + "\n".join(fest_lines))

        prompt = ("\n\n".join(blocks)
                  + "\n\n以上素材按主笔原则取舍后写今日总览卡片："
                    "先说今天，今天是主角；其余素材加起来最多轻轻点一两处，"
                    "没有值得说的整块不提。像朋友说话，不像公文。"
                    "你们是平辈朋友：不点评不夸奖（\"很棒\"\"做得不错\"都是打分），"
                    "安抚、建议、祝愿只朝前看（\"希望你更轻松\"\"早点休息哦\"）。")
        logger.info(f"[background-agent] 栏目素材就绪（{len(blocks)} 块），开始组稿")
        return await self.call_llm(prompt, system_prompt=OVERVIEW_CARD_PROMPT)

    async def run_proactive_cards(self) -> Optional[Dict[str, Optional[str]]]:
        """回忆卡片组稿（07:00 与总览同批）：时光胶囊 + 去年今天的叙述文案

        代码选材 → LLM 组稿。返回 None = 两边素材全空（没组稿，调用方不落盘，
        当天晚些时候回填的素材仍可走原文路）；返回 dict = 今天已叙述的判定，
        整卡交白卷（弃说）也是判定，照常落盘——弃说不退原文碎片（那是用户
        明说过"缺乏温度，失去意义"的东西），组稿失败才退。
        选中胶囊候选即写 surface_log（当日稳定，刷新不闪卡）——弃说的条目
        照常烧冷却：会被弃说的条目下次进来还是被弃说，无害；"选中"与"叙述"
        解耦才换得到这个稳定。
        """
        today = date.today()
        blocks = []

        ly_items = get_last_year_today()
        if ly_items:
            lines = []
            for i in ly_items:
                if isinstance(i, Event):
                    note = f"：{i.note}" if i.note else ""
                    lines.append(f"- {i.title}{note}")
                else:
                    lines.append(f"- 心情 {MOOD_EMOJI.get(i.mood, '😐')} {i.content or ''}")
            ly_date = (today - timedelta(days=365)).isoformat()
            blocks.append(f"【去年今天】（{ly_date} 的记录）\n" + "\n".join(lines))

        cap_items = get_time_capsule_candidates()
        if cap_items:
            lines = []
            for e in cap_items:
                days = (today - date.fromisoformat(e.date)).days
                note = f"：{e.note}" if e.note else ""
                lines.append(f"- {days} 天前：{e.title}{note}")
            blocks.append("【时光胶囊】（隔了很久、刚好到该想起的时候的旧事）\n" + "\n".join(lines))

        if not blocks:
            logger.info("[background-agent] 回忆卡片无素材（去年今天与唤起窗皆空），跳过组稿")
            return None

        prompt = (f"【生成时刻】{today.isoformat()} {WEEKDAY_NAMES[today.weekday()]} "
                  f"{datetime.now().strftime('%H:%M')}\n\n"
                  + "\n\n".join(blocks) + "\n\n请写两张卡片的文案。")
        logger.info(f"[background-agent] 回忆卡片素材就绪（去年今天 {len(ly_items)} 条 / "
                    f"胶囊 {len(cap_items)} 条），开始组稿")
        result = await self.call_llm_json(prompt, system_prompt=PROACTIVE_CARD_PROMPT)
        return {
            "capsule": str(result.get("capsule") or "").strip() or None,
            "last_year": str(result.get("last_year") or "").strip() or None,
        }

    async def run_weekly(self) -> str:
        """每周日 20:00 周总结（调度器 builtin:weekly）"""
        today = date.today()
        start = today - timedelta(days=6)
        events = db.get_events_by_date_range(start.isoformat(), today.isoformat())
        moods = db.get_moods_by_date_range(start.isoformat(), today.isoformat())

        # 逐日摘要：日期 → 事件数/完成数/标签
        day_lines = []
        for i in range(7):
            d = (start + timedelta(days=i)).isoformat()
            day_events = [e for e in events if e.date == d]
            done = sum(1 for e in day_events if e.type == "done")
            tags = Counter(t for e in day_events for t in e.tags)
            tag_text = "、".join(f"{tag}×{n}" for tag, n in tags.most_common(3))
            day_lines.append(
                f"- {d} {WEEKDAY_NAMES[(start.weekday() + i) % 7]}：{len(day_events)} 条记录"
                f"（完成 {done}）" + (f"，主要标签：{tag_text}" if tag_text else "")
            )
        schedule_text = "\n".join(day_lines) or "（近 7 天没有记录）"

        # 心情线索
        mood_items = [f"{getattr(m, 'mood', '')} {getattr(m, 'content', '')}".strip() for m in moods]
        mood_text = "\n".join(f"- {m}" for m in mood_items if m) or "（本周没有心情记录）"

        prompt = f"""【近 7 天日程统计】（{start.isoformat()} ~ {today.isoformat()}）
{schedule_text}

【本周心情记录】
{mood_text}

请写周总结。"""
        logger.info(f"[background-agent] 周结素材就绪（事件 {len(events)} 条 / 心情 {len(moods)} 条），开始组稿")
        return await self.call_llm(prompt, system_prompt=WEEKLY_SYSTEM_PROMPT)

    # ─── 形态三：agent 式（无状态 loop 执行自然语言委托）──

    async def run_task(self, prompt: str) -> str:
        """执行用户注册的 agent 定时任务：一次无状态 agent loop，无会话记忆无状态机。

        意图即载荷（借自 s12）：注册时只存 prompt，fire 时才解释——"明早"相对
        fire 时刻才有确定语义。返回文本进信箱。"""
        from app.agent.tools import (  # 延迟导入：能力层共享，但 import 期不互相拉起
            TOOL_DEFINITIONS, TOOL_FUNCTION_MAP, ASYNC_TOOLS, WRITE_TOOLS,
        )

        today = date.today()
        messages: List[dict] = [
            {"role": "system", "content": TASK_SYSTEM_PROMPT.format(
                today=today.isoformat(), weekday=WEEKDAY_NAMES[today.weekday()],
                now=datetime.now().strftime("%H:%M"))},
            {"role": "user", "content": f"[定时任务] {prompt}"},
        ]
        # 诚实不变量（同交互版）：写工具只失败未成功时，"已记录"式汇报必为幻觉
        write_ok, write_failed = False, False
        empty_retries = 0

        for iteration in range(1, TASK_MAX_ITERATIONS + 1):
            t0 = datetime.now()
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.3,
                # 工具选择错一次就丢数据，与交互版同一可靠性开关
                extra_body={"enable_thinking": DASHSCOPE_ENABLE_THINKING},
            )
            msg = response.choices[0].message
            usage = response.usage
            tok = (f", prompt={usage.prompt_tokens}tok, completion={usage.completion_tokens}tok"
                   if usage else "")
            elapsed = (datetime.now() - t0).total_seconds() * 1000
            if msg.tool_calls:
                logger.info(f"[background-agent] run_task iter {iteration} → tool_calls: "
                            f"{[tc.function.name for tc in msg.tool_calls]} ({elapsed:.0f}ms{tok})")
            else:
                logger.info(f"[background-agent] run_task iter {iteration} → 文本 ({elapsed:.0f}ms{tok})")

            messages.append({"role": "assistant", "content": msg.content or "", **(
                {"tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]} if msg.tool_calls else {}
            )})

            # 终止：无 tool_call 即最终汇报
            if not msg.tool_calls:
                final = (msg.content or "").strip()
                # 思考型模型偶发把答案留在思维链里：空文本给一次重试机会
                if not final and empty_retries < 1:
                    empty_retries += 1
                    messages.pop()
                    logger.warning("[background-agent] run_task 返回空文本（思维链吞了答案），重试")
                    continue
                if write_failed and not write_ok:
                    logger.warning("[background-agent] run_task 写工具全部失败，覆写疑似成功幻觉的汇报")
                    final = f"⚠️ 这条任务执行时存储出了问题，安排没有保存成功——原委托：{prompt}。可以稍后再试一次。"
                return final or "（任务执行完了，但没能生成汇报文字）"

            # 执行工具（与交互版共享同一注册表与异步约定）
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    result = "错误：参数不是合法 JSON"
                else:
                    func = TOOL_FUNCTION_MAP.get(tc.function.name)
                    if func is None:
                        result = f"错误：未知工具 {tc.function.name}"
                    else:
                        try:
                            result = (await func(**args) if tc.function.name in ASYNC_TOOLS
                                      else func(**args))
                        except Exception as e:
                            # 单工具失败不废掉整个任务：错误文本喂回 LLM 自行调整
                            # （交互版 fail_streak 熔断的压缩版——后台无人值守，救不了就如实汇报）
                            logger.warning(f"[background-agent] run_task 工具 {tc.function.name} 失败: "
                                           f"{type(e).__name__}: {e}")
                            result = f"错误：{type(e).__name__}: {e}"
                result = str(result)
                logger.info(f"[background-agent] run_task 工具 {tc.function.name} 完成: "
                            f"{' '.join(result.split())[:200]}")
                if tc.function.name in WRITE_TOOLS:
                    if result.startswith("错误"):
                        write_failed = True
                    else:
                        write_ok = True
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": result[:TOOL_RESULT_MAX_CHARS]})

        # 轮数用尽：如实说明，不假装完成
        logger.warning(f"[background-agent] run_task 达到轮数上限 {TASK_MAX_ITERATIONS}，如实汇报")
        return f"⚠️ 这条任务步骤太多没跑完（可能不完整），原委托：{prompt}。可以拆简单一点重新注册试试。"
