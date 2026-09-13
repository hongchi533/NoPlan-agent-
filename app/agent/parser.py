"""ParserAgent：自然语言 → 结构化数据

用户输入一句话，解析为 Event 和/或 MoodRecord。
一句话可能同时包含事件和情绪，返回多条记录。
"""
from datetime import date, datetime, timedelta
import json
import logging
import uuid
from typing import List, Union

from app.agent.base import BaseAgent
from app.config import DASHSCOPE_FAST_MODEL
from app.models.schemas import ParseResult, ParseResults, Event, MoodRecord, Preference
from app.store import db

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """解析失败（LLM 输出无法转为结构化记录）。

    上抛给 agent loop 统一转成失败消息，主 LLM 如实告知用户——
    绝不在这一层静默降级成"乱记录"。
    """

PARSER_SYSTEM_PROMPT = """你是一个私人日程助手，负责把用户的一句话解析为结构化数据。今天是 {today} {weekday}。

## 核心能力：一句话可能同时包含事件和情绪

用户的一句话可能既有事件又有情绪，你需要把它们都识别出来，返回多条记录。
- 例："下午写了代码，非常累但完成了核心功能" → 同时返回 done事件 + mood心情
- 例："今天跑了5公里，感觉超爽" → 同时返回 done事件 + mood心情
- 例："明天要交周报，好焦虑" → 同时返回 plan事件 + mood心情
- 例："下午去了趟宜家" → 只返回 done事件
- 例："今天有点累" → 只返回 mood心情

## 判断 type 的规则（最重要，必须严格遵守）

1. type = "mood"：用户在表达整体的情绪感受、内心状态
   - 真正的情绪词：累、开心、焦虑、舒服、踏实、郁闷、幸福、烦躁、满足、难过、感动、爽、兴奋
   - ⚠️ 以下不算 mood，只是对事物的评价/描述，不要过度分析：
     - 对食物/饮品的口味评价：甜、好喝、好吃、还不错、一般般
     - 对事情的客观评价：顺利、搞定、完成了、还行
     - 对事件的客观记录：下周要开组会
     - 这些归入事件的 note 字段即可，不要单独生成 mood 记录

2. type = "plan"：用户在说打算做、要做、准备做的事（还没发生）
   - 包含将来时态：要、打算、准备、计划、想去、明天、后天、周末

3. type = "done"：用户在说已经发生的事（过去时态或正在进行）

4. type = "preference"：用户在表达习惯、偏好、默认设置
   - 包含偏好词：习惯、默认、一般、通常、喜欢、偏好、总是、每次都
   - 例："我习惯早上六点跑步" → type="preference"
   - 例："以后跑步都算运动" → type="preference"
   - 例："写代码一般上午" → type="preference"
   - 偏好分两类，落不同字段：
     带默认时刻的习惯（"我一般十二点吃午饭"）→ time=默认时刻
     生活事实（"我喜欢一边吃饭一边喝咖啡"）→ note=一句话规范化复述，≤20字，
       独立可读、去掉"我喜欢/我一般"的口语壳（如"吃饭时喜欢配咖啡"）
   - title 用偏好的核心活动词（吃饭配咖啡→"喝咖啡"），不要编类目名

## 用户偏好（必须遵守）
{preferences}

## 其他规则

- 时间模糊推断："早晨"=08:00，"上午"=09:00-12:00，"中午"=12:00，"下午"=14:00-18:00，"晚上"=19:00，"周末"=最近一个周六，"下周"=下周一（一周从周一起算，禁止取周中/周末），"过两天"=今天+2，明天 = {tomorrow}
- 全天事件判定：只有日期描述（今天/明天/昨天/周末/下周三…），完全没有具体时刻、也没有时段词（早晨/上午/中午/下午/傍晚/晚上/凌晨）→ 全天事件，time 只写日期（"YYYY-MM-DD"），严格禁止编造时刻。生日、节日、纪念日、请假、假期、出差一整天、考试日等天然属于全天
- 多天事件按天展开：用户列举多个日期（"28日、29日、30日请假"）或说明连续多天（"22号到24号出差""请三天假"）→ 每天一条独立事件，time 各写各的日期。严格禁止输出"2026-09-28 - 2026-09-30"这类区间串——time 只支持单日期或当日时段，区间串会被截断成第一天，后面的天全部丢失
- 歧义时刻消歧（只说"九点""三点"没指上午/晚上）：现在是 {today} {now}，请据此计算时间, 用户此刻说的话不会舍近求远，24小时制
  - plan（还没发生）→ 取 {now} 之后最近的一个该时刻
  - done（已发生）→ 取 {now} 之前最近的一个该时刻
- 星期几推理规则：今天的绝对日期是 {today}，今天是 {weekday}
 - 当用户提及“下周三”，请先根据当前是“{weekday}”计算跨度n：然后在当前日期基础上往后加 n 天得到标准日期。
 - 严格禁止自行猜测年份和周几的对应关系，必须以此处给出的基准进行数学加减。
- title 最多5个字，细节放 note
- tags 只从以下选：工作、生活、运动、社交、饮食、学习、娱乐、家务、惊喜
  - 一个事件可以有多个tag
  - 影音、唱歌、节日、生日等归为娱乐
  - 打扫、做饭、洗衣、修理等归为家务
  - 收到礼物、意外之喜、惊喜时刻归为惊喜
- is_moment：是否为值得一年后回看的高光时刻——生日、纪念日、里程碑（第一次做到、终于
  达成）、强烈情绪时刻。判情绪强度，不判事件大小：小事配强情绪词（"吃了炸鸡，特别
  开心！"）也是 moment；没有感叹号照样标（"人生第一个半马"式里程碑措辞即算）。
  不算的是纯语气性感叹（"终于周五了！"——无具体事件）和轻度情绪（"挺舒服""还行"）
- mood 条目的 note 要自然通顺，补充事件背景让情绪有来处，不要只截取情绪词；
  过去时间词汇（昨天/昨晚等）必须剔除，“补记过去”的心情需要转成当天口吻——日期已由 time 字段承载；
  - ❌ "非常累但完成了核心功能"（没头没尾）
  - ✅ "下午写代码很累，但搞定了核心功能，累而有成就感"
  - ✅ "跑步5公里后感觉超爽"
- note/title: 仅保留事件的核心内容或主题。必须剔除已被提取到 time 字段中对应的时间词汇（如“明天”、“下午”等）。请将剩余内容提炼为精简的动宾结构（例如：“会议”、“与张总面谈”）。
  - ❌ 错误输出：note: 明天下午有个会议（time已经足以表示明天）
  - ✅ 正确输出：note: 会议

## 输出 JSON 格式（所有条目同一格式，按类型取用字段，能省的一律省略）

{{"items":[
  {{"type":"done","title":"简洁标题","time":"YYYY-MM-DD HH:MM-HH:MM","note":"细节","tags":["标签"]}},
  {{"type":"mood","time":"YYYY-MM-DD HH:MM","note":"情绪描述","mood":3}}
]}}

字段规则：
- time 日期时间合写一个字段："YYYY-MM-DD HH:MM-HH:MM"（有起止）、"YYYY-MM-DD HH:MM"（只有开始）、"YYYY-MM-DD"（无时间）、"HH:MM"（省日期=今天）
- mood 条目同样用 time 记具体时刻：只记过去和当下——未来还没发生，谈不上心情，对未来的预感/担忧/期待都是"现在感受到的情绪"，一律记为当下时刻（现在是 {today} {now}）；补记的过去按语境推断（如"昨晚" → "{yesterday} 22:00"）
  - ❌ "明天一定很糟糕" → time 用了明天的日期（心情被记到了未来）
  - ✅ time 用 {today} {now}，note 从当下感受出发（"为明天的事担忧，情绪低落"）
- mood 为 1-5 整数分制：1=很糟 2=偏低 3=一般 4=不错 5=很好；再强烈的情绪也只到 1 或 5，不要输出 0/6/7
- mood 条目不要 title/tags，分数放 mood 字段；done/plan 条目不要 mood 字段（心情用单独的 mood 条目表达）
- 值为空的字段直接省略，不要写 null
- is_moment 只在确实是高光时刻时才写 true，否则省略（判定看措辞，不依赖感叹号）

## 示例（只列最易错的情形，单条的 done/plan 直接照字段输出）

用户输入："下午写了代码，非常累但完成了核心功能"
输出：{{"items":[{{"type":"done","title":"写代码","time":"{today} 14:00-18:00","note":"完成了核心功能","tags":["工作"]}},{{"type":"mood","time":"{today} 15:00","note":"下午写代码很累，但搞定了核心功能，累而有成就感","mood":3}}]}}

用户输入："昨晚有点emo"
输出：{{"items":[{{"type":"mood","time":"{yesterday} 20:00","note":"有点emo","mood":2}}]}}

用户输入："明天一定很糟糕"
输出：{{"items":[{{"type":"mood","time":"{today} {now}","note":"为明天的事担忧，情绪低落","mood":2}}]}}

用户输入："我习惯早上六点跑步"
输出：{{"items":[{{"type":"preference","title":"跑步","time":"06:00","note":"习惯早上六点","tags":["运动"]}}]}}

用户输入："昨天一整天都在出差，晚上到家都快九点了，累死我了"
输出：{{"items":[{{"type": "done","title": "出差","time": "{yesterday}","note": "整天出差行程","tags": ["工作"]}},{{"type": "mood","time": "{yesterday} 21:00","note": "出差深夜才到家，身体极度疲惫","mood": 2}}]}}

用户输入：“等一下，我突然想起来下明天早会前要把周报发了，还有今晚七点别忘了去健身房”
输出：{{"items":[{{"type":"plan","title":"发送周报","time":"{tomorrow} 08:50","note":"在部门早会开始前完成发送","tags":["工作"]}},{{"type":"plan","title":"去健身房","time":"{today} 19:00","note":"力量/有氧训练","tags":["运动"]}}]}}

用户输入："帮我记一下，明天下午三点要跟李总开会"
输出：{{"items":[{{"type":"plan","title":"开会","time":"{tomorrow} 15:00","note":"跟李总开会","tags":["工作"]}}]}}

用户输入："今天是我生日"
输出：{{"items":[{{"type":"done","title":"生日","time":"{today}","note":"过生日","tags":["娱乐"],"is_moment":true}}]}}

用户输入："今天拿到了心仪公司的offer，激动到不行！"
输出：{{"items":[{{"type":"done","title":"拿offer","time":"{today}","note":"心仪公司offer到手","tags":["工作"],"is_moment":true}},{{"type":"mood","time":"{today} {now}","note":"拿到心仪offer，激动到不行","mood":5}}]}}

用户输入："跑完人生第一个半马，两小时完赛"
输出：{{"items":[{{"type":"done","title":"半马首秀","time":"{today}","note":"人生第一个半马，两小时完赛","tags":["运动"],"is_moment":true}}]}}

用户输入："昨天吃了炸鸡，特别开心！"
输出：{{"items":[{{"type":"done","title":"吃炸鸡","time":"{yesterday}","tags":["饮食"],"is_moment":true}},{{"type":"mood","time":"{yesterday}","note":"吃了炸鸡，特别开心","mood":5}}]}}
（mood 的 note 里"昨天"必须剔除：日期已由 time 字段承载，正文相对词隔天回看就是错的）

用户输入："明天要去郊游"
输出：{{"items":[{{"type":"plan","title":"郊游","time":"{tomorrow}","tags":["娱乐"]}}]}}

用户输入："今天喝了咖啡"
输出：{{"items":[{{"type":"done","title":"喝咖啡","time":"{today}","tags":["饮食"]}}]}}

"""

def _expand_time(raw: dict):
    """把紧凑 time 字段拆回 date/start_time/end_time（原地修改）

    LLM 线格式 → 内部字段：
      "2026-08-24 14:00-16:00" → date + start + end
      "2026-08-24 14:00"       → date + start
      "2026-08-24"             → date
      "14:00"                  → start（日期由调用方默认今天）
    """
    t = raw.pop("time", None)
    if not t or not isinstance(t, str):
        return
    t = t.strip()

    if len(t) == 5 and ":" in t:  # 纯 HH:MM
        raw["start_time"] = t
        return

    parts = t.split(" ")
    raw["date"] = parts[0]
    if len(parts) > 1:
        span = parts[1]
        if "-" in span:
            start, end = span.split("-", 1)
            raw["start_time"], raw["end_time"] = start, end
        else:
            raw["start_time"] = span


class ParserAgent(BaseAgent):
    """解析一句话为结构化事件/心情（可能同时返回多条）"""

    name = "parser"
    # 结构化解析是简单任务，用快模型提速
    model_override = DASHSCOPE_FAST_MODEL

    async def parse(self, text: str) -> List[ParseResult]:
        """解析用户输入，返回一条或多条解析结果（prompt 动态注入用户偏好）"""
        today = date.today().isoformat()
        # 获取星期几（0=周一, 6=周日）
        current_date = datetime.today().date()
        weekday_map = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        weekday = weekday_map[current_date.weekday()]
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        now = datetime.now().strftime("%H:%M")

        # 读取用户偏好，注入 prompt
        preferences = self._format_preferences()
        system_prompt = PARSER_SYSTEM_PROMPT.format(
            today=today, yesterday=yesterday, tomorrow=tomorrow, now=now, weekday=weekday,
            preferences=preferences,
        )

        try:
            result = await self.call_llm_json(
                user_message=text,
                system_prompt=system_prompt,
            )
            # 兼容：LLM 可能返回 {"items":[...]} 或直接返回单个对象
            if "items" in result:
                for raw in result["items"]:
                    _expand_time(raw)
                parsed = ParseResults(**result)
                return parsed.items
            else:
                _expand_time(result)
                return [ParseResult(**result)]
        except Exception as e:
            logger.error(f"[parser] 解析失败: {e}, 原始输入: {text}")
            raise ParseError(f"无法把这句话解析成结构化记录（{e}）") from e

    def _format_preferences(self) -> str:
        """读取 profile.json 中的偏好，格式化为 prompt 文本"""
        try:
            prefs = db.load_preferences()
        except Exception as e:
            logger.warning(f"[parser] 读取偏好失败: {e}")
            return "（暂无偏好记录）"

        if not prefs:
            return "（暂无偏好记录）"

        # manual 优先于 auto
        prefs.sort(key=lambda p: 0 if p.source == "manual" else 1)

        lines = []
        for p in prefs:
            # parse 只消费"默认时间"——唯一能填空白（缺时刻）的偏好类型。
            # fact 型（"吃饭时喜欢配咖啡"）没有可填的字段、也解不了指代，
            # 进 parse prompt 只是噪声；它是决策侧（检索/回复/总览）的素材，不在这注入
            if not p.rules.get("default_time"):
                continue
            source_label = "用户声明" if p.source == "manual" else "系统发现"
            lines.append(f"- {p.pattern}: 默认时间 {p.rules['default_time']}（{source_label}）")

        guidance = "\n".join(lines)
        if not guidance:
            return "（暂无偏好记录）"
        return (
            f"{guidance}\n"
            f"应用规则：当用户提到上述活动但未指定时间时，使用偏好中的默认时间。"
        )

    async def parse_and_create(self, text: str) -> List[Union[Event, MoodRecord, Preference]]:
        """解析并创建 Event/MoodRecord/Preference 对象列表（不存储）"""
        parsed_list = await self.parse(text)
        now = datetime.now().isoformat()
        results = []

        for parsed in parsed_list:
            if parsed.type == "preference":
                # 偏好：pattern=title；规则分型——默认时刻 or 生活事实短句。
                # 事实走"写时压缩"：note 是 LLM 规范化复述（≤20字），source_text 只存证不进 prompt
                rules = {}
                if parsed.start_time:
                    rules["default_time"] = parsed.start_time
                if parsed.note:
                    rules["fact"] = parsed.note
                results.append(Preference(
                    id=str(uuid.uuid4()),
                    pattern=parsed.title or text[:5],
                    rules=rules,
                    source="manual",
                    source_text=text,
                    evidence={"recent_mismatches": 0},
                    created_at=now,
                    updated_at=now,
                ))
            elif parsed.type == "mood":
                # 心情不变量（plan"不落过去"的镜像）：只记过去和当下——
                # 对未来的预感/期待是"现在感受到的情绪"，属于今天。LLM 会把输入里的
                # 未来词照抄进 time（实测"明天一定很糟糕"→记到明天），代码统一收拢为当下
                mood_date = parsed.date or date.today().isoformat()
                mood_time = parsed.start_time
                now_hm = datetime.now().strftime("%H:%M")
                if (mood_date > date.today().isoformat()
                        or (mood_date == date.today().isoformat() and mood_time and mood_time > now_hm)):
                    logger.info(f"[parser] mood 落在未来（{mood_date} {mood_time or ''}），收拢为当下: {parsed.note}")
                    mood_date, mood_time = date.today().isoformat(), now_hm
                results.append(MoodRecord(
                    id=str(uuid.uuid4()),
                    date=mood_date,
                    time=mood_time,
                    # 分值收拢进 [1,5]：LLM 偶发越界（0/6/7），clamp 保住记录而不是整条丢弃。
                    # 不能写 `parsed.mood or 3`——0 是合法的"很糟"信号，or 会把它误抬成 3
                    mood=max(1, min(5, parsed.mood)) if parsed.mood is not None else 3,
                    content=parsed.note or parsed.title or text,
                    strength=1.0,
                    created_at=now,
                ))
            else:
                event_date = parsed.date or date.today().isoformat()
                # 不变量兜底：plan 不能落在过去。歧义时刻（如"九点"）LLM 可能选到已过时刻 → 顺延到明天同一时刻
                if (parsed.type == "plan" and parsed.start_time
                        and event_date == date.today().isoformat()
                        and parsed.start_time <= datetime.now().strftime("%H:%M")):
                    event_date = (date.today() + timedelta(days=1)).isoformat()
                    logger.info(
                        f"[parser] plan 落在过去（今天 {parsed.start_time}），顺延到明天: {parsed.title}")
                # 不变量兜底：未来无已完成。裸日期的增强文本（"2026-09-01 06:30 刷牙"）
                # 缺"明天/要"这类将来标记时 LLM 会误判 done → 强制归 plan，账目归代码不归 prompt
                item_type = parsed.type
                if item_type == "done" and event_date > date.today().isoformat():
                    item_type = "plan"
                    logger.info(
                        f"[parser] done 落在未来（{event_date}），纠正为 plan: {parsed.title}")
                results.append(Event(
                    id=str(uuid.uuid4()),
                    title=parsed.title or text[:5],
                    type=item_type,
                    date=event_date,
                    start_time=parsed.start_time,
                    end_time=parsed.end_time,
                    note=parsed.note,
                    tags=parsed.tags,
                    source_text=text,
                    strength=1.0,
                    is_moment=parsed.is_moment,
                    created_at=now,
                ))

        return results
