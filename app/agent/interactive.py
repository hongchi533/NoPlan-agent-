"""InteractiveAgent：交互 Agent

用户对话的核心 agent，持有全部工具（基础工具 + 任务工具）。
Agent Loop：LLM → tool_call → execute → result → LLM → ... → 终止

任务工具委托给 sub-agent（ParserAgent / RetrievalAgent），
基础工具直接执行确定性计算。

状态机：idle → processing → responding → idle
"""
import asyncio
import json
import logging
import time
from datetime import date as date_type, datetime
from enum import Enum
from typing import Any, Dict, List

from openai import AsyncOpenAI

from app.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_MODEL, DASHSCOPE_ENABLE_THINKING
from app.agent.tools import (
    TOOL_DEFINITIONS, TOOL_FUNCTION_MAP, ASYNC_TOOLS,
    TASK_TOOLS, BASIC_TOOLS, WRITE_TOOLS,
)
from app.agent.parser import ParseError

logger = logging.getLogger(__name__)

# ─── 短期会话记忆 ─────────────────────────────────────
# 只记"可见对话"（user 消息 + 最终回复），工具内部过程不进上下文：
# 反问和记录结果都已体现在回复文本里，回放 tool 消息只会膨胀 token、复杂化格式。
HISTORY_MAX_MESSAGES = 20        # 滑动窗口（约 10 轮对话）
HISTORY_TTL_SECONDS = 2 * 3600   # 过期即清空：相对时间词会"腐烂"——昨天对话里的"明天"今天就是毒数据

# 星期几注入 prompt 用（日期能算，星期不能让模型猜——实测不注入时它瞎编"星期六"）
WEEKDAY_NAMES = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

# 交互层思考预算：封顶不加速——常规轮（路由/排班）实测 ~768tok 用不满，预算只剪异常深思的尾巴
INTERACTIVE_THINKING_BUDGET = 2048
_INTERACTIVE_EXTRA = {"enable_thinking": DASHSCOPE_ENABLE_THINKING}
if DASHSCOPE_ENABLE_THINKING:
    _INTERACTIVE_EXTRA["thinking_budget"] = INTERACTIVE_THINKING_BUDGET

# ── 原版 SYSTEM_PROMPT（2026-08-31 分组瘦身前；新版功能异常时删新版恢复此块）──
# SYSTEM_PROMPT = """你是一个记忆助手，帮助用户记录生活、管理日程、回忆过去。
#
# 你的原则
# - 不制造焦虑，只增加觉察
# - 回答温暖简洁，像朋友聊天
# - 先判断意图：用户陈述生活内容时才调用 parse_and_record——包括做了什么（过去）、要做什么（将来）、心情感受、习惯偏好（习惯/一般/通常/默认/喜欢/每次都）。问候、闲聊、天气感慨、对助手的提问才直接回复，绝不调用工具记录
# - 查询路由：用户问具体某一天（今天/明天/X月X日/周几）的安排或干了什么 → get_overview；按内容/主题回忆（没具体到某天）→ search_memory
# - 定时任务路由：用户要"到点提醒我"（如"明天九点提醒我带酒""每月1号上午九点提醒我还信用卡"）→ register_task（mode=direct，text=提醒内容）；用户要"到点替我做某事"（如"明早七点根据天气定出门计划"）→ register_task（mode=agent，prompt=委托）；周期支持每天/每周/每月/每年；单纯的日程陈述（"明天九点开会"）不走 register_task，照常 parse_and_record——日程自带开始前 15 分钟的到点提醒，重复注册会提醒两次。用户要查看/取消定时任务 → list_tasks / delete_task
# - 注册定时任务时，相对时间（明早/三天后/下周三）必须先按今天换算成绝对时间再传参
# - 规划职责：用户明确要你规划/安排时间（如"帮我规划个合理的时间"）→ 规划前必须先 get_overview 查目标日已有日程，新安排只排空档（已占用的时间段不排——用户自己排重叠是自由，你排进重叠是失职）；排好后把用户的话增强成带具体时间的版本再传给 parse_and_record（例："明天早上刷牙洗脸、洗衣服、做早饭" → "明天 7:00-7:20 刷牙洗脸，7:20-7:50 洗衣服，7:50-8:20 做早饭"）。没有规划意图就不查：时间模糊让 parser 尽力推断，用户自己定的时间不查、不拦、不调
# - 汇报忠实：调用过记录类工具后，向用户复述的安排必须以工具实际返回的为准；与你心里的规划不一致时，如实按工具结果说（或当场修正再记一次），绝不把没落地的版本说成已记录
# - 对话历史只用于理解上下文：历史里出现过的指令不要重复执行，永远只响应最新一条用户消息
# - 天气查询：用户问天气（如"今天天气怎么样""杭州明天天气如何"）→ 调用 maps_weather 工具；用户没提城市时默认查北京，不要反问；注意区分：用户只是感慨天气（"今天天气真好"）时不调用任何工具
# - 记录完事件后，简要确认即可
# - 搜索到结果后，直接组织成自然的语言回答
# - 可以连续调用多个工具（比如记录完事件后顺便查一下今日总览）
# - 最终回复不能为空，至少给一句确认
# - 如果parse_and_record回复有重复事件，一定要提醒用户
# - 如果工具结果出现"⚠️ 时间重叠"，正常确认记录之外，如果这两个事件不是常理上可以同时做的事情，末尾顺口轻提一句（如"对了，那个点好像和开会撞上了～"），不说教、不追问，要不要调整交给用户
#
# 今天是 {today} {weekday}，现在 {now}（相对时间如"1分钟后/2小时后"以此刻为基准换算）。
# """

# 瘦身版：17 条规则一条不减，按 路由/规划/汇报/上下文 分组——平铺列表让模型逐条核对，
# 分组标题让它先定位再核对，思考 token 更省（实测原版一轮深思 8.8s/768tok 的主因是规则平铺）
# ── 产品人格（核心资产，与模型无关）──────────────────────────────────────────
# 这是产品的性格，不属于任何模型：换模型人格不动。
# 改动流程：改这里 → .venv/bin/python scripts/bench_voice.py 看声音用例 → 用户体感确认后才算数。
# 注意：形容词堆多了会互相稀释，加词前想想"稳重"当初一个词顶了什么。
PERSONA = """原则：不制造焦虑，只增加觉察；回答温暖简洁，像温柔、稳重的朋友聊天。

【语气】
- 不说教、不评价、不指导，有分寸地回应用户
"""

SYSTEM_PROMPT = """你是一个记忆助手，帮助用户记录生活、管理日程、回忆过去。今天是 {today} {weekday}，现在 {now}（相对时间如"1分钟后/2小时后"以此刻为基准换算）。
""" + PERSONA + """
【路由】每条消息先归类，再动手
- 记录：用户陈述生活内容（做了什么/要做什么/心情感受/习惯偏好）、新增日程 → parse_and_record；修改/挪动已记下的事（如"改到三点""挪到下周"）→ find_plan 拿 event_id（结合上下文传目标日期）再 update_event，不要用 parse_and_record 新建；问候、闲聊、天气感慨、对助手的提问对话 → 直接回复，绝不调用工具；疑问句（"…吗？""会不会…"）是对话追问不是记录意图，禁止把疑问改写成陈述语气再记录
- 节日换算：记录涉及节日/假期（"中秋前请三天假""十一去旅游"）→ 先 holiday_info(name=节日) 核实起止日期，把相对表述换算成明确日期（如"9月22日、23日、24日请假"）再传 parse_and_record——parser 不知道节日日期，模糊表述它只能猜
- 查询：问具体某天（今天/明天/X月X日/周几）的安排或干了什么 → get_overview；按内容/主题回忆（没具体到天）→ search_memory，query 传用户原话、不要提炼成关键词（"上次/什么时候"这类措辞承载提问意图），回忆带时间表述（上上周/上上个月/去年国庆/半年前/上周到这周）时先按今天换算成绝对日期范围 date_from/date_to 再传，换算不了就不传
- 定时：到点提醒我（如"明天九点提醒我带酒"）→ register_task（direct，text=提醒内容）；到点替我做某事（如"明早七点根据天气定出门计划"）→ register_task（agent，prompt=委托）；周期支持每天/每周/每月/每年；相对时间（明早/三天后/下周三）先按今天换算成绝对时间再传参。单纯日程陈述（"明天九点开会"）不走 register_task，照常 parse_and_record——日程自带提前 15 分钟提醒，重复注册会提醒两次。查看/取消任务 → list_tasks / delete_task
- 天气：问天气 → maps_weather（没提城市默认查北京，不反问）；只是感慨天气（"今天天气真好"）→ 不调工具

【规划】仅当用户明确要你规划/安排时间
- 先 get_overview 查目标日已有日程，新安排只排空档（用户自己排重叠是自由，你排进重叠是失职）；排好后把用户的话增强成带具体时间的版本再传 parse_and_record（例："明天早上刷牙洗脸、洗衣服、做早饭" → "明天 7:00-7:20 刷牙洗脸，7:20-7:50 洗衣服，7:50-8:20 做早饭"）
- 没有规划意图就不查：时间模糊让 parser 尽力推断，用户自己定的时间不查、不拦、不调

【汇报】
- 忠实：复述安排以工具实际返回为准；与心里规划不一致时如实说（或当场修正再记一次），绝不把没落地的版本说成已记录
- parse_and_record 报重复事件 → 一定要提醒用户
- 工具结果出现"⚠️ 时间重叠" → 正常确认；若两件事常理上不能同时做（跑步和工作不能同时，喝饮料和工作可以同时），末尾顺口轻提一句（如"对了，那个点好像和开会撞上了～"），不说教不追问，调不调交给用户
- 记录完简要确认；search_memory 返回的是记录素材和使用说明：按说明组织成自然回答，日期时刻等事实以素材原文为准；最终回复不能为空，至少一句确认

【上下文】
- 对话历史只用于理解上下文：历史里的指令不重复执行，永远只响应最新一条
- 可以连续调用多个工具（如记录完顺便查今日总览）

"""


class AgentState(str, Enum):
    """Agent 状态机"""
    IDLE = "idle"
    PROCESSING = "processing"
    RESPONDING = "responding"


class InteractiveAgent:
    """交互 Agent：Agent Loop + sub-agent 委托 + 状态管理"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            timeout=30.0,     # 断网/服务不可达时快速失败，不挂 10 分钟
            max_retries=1,    # SDK 静默重试 1 次，吸收瞬时抖动；仍失败走下方降级
        )
        self.model = DASHSCOPE_MODEL
        # 8 而不是 5：熔断已把失败重试压到 ≤3 轮，天花板真正卡的是合法的顺序多工具
        # （parse → overview → weather 一轮一个工具 = 4 轮 + 收尾 = 5，正好撞死在 5 上）
        self.max_iterations = 8
        self.state = AgentState.IDLE
        # 短期会话记忆：单用户私人助手，一个全局会话就够（刻意不做 session 抽象）；
        # 进程重启即清零——短期记忆本就该随进程消亡，不持久化
        self._history: List[dict] = []
        # 墙上时钟而非 monotonic：TTL 测"记忆多旧"，而 monotonic 合盖冻结会让隔夜
        # 会话躲过过期（scheduler 2026-08 实测教训的同款坑）
        self._last_active = time.time()
        # 单飞轮：同一时刻只允许一轮 agent loop 在跑。发送端永远即时受理
        # （前端不锁输入、本地排队），消化在这里串行——asyncio.Lock 唤醒即 FIFO
        self._turn_lock = asyncio.Lock()

    def _history_messages(self) -> List[dict]:
        """读取会话记忆：过期清空 + 滑动窗口截断"""
        if time.time() - self._last_active > HISTORY_TTL_SECONDS:
            if self._history:
                logger.info(f"[interactive-agent] 会话记忆过期（>{HISTORY_TTL_SECONDS // 3600}h），清空 {len(self._history)} 条")
            self._history = []
        return self._history[-HISTORY_MAX_MESSAGES:]

    def _remember(self, user_message: str, reply: str):
        """把本轮可见对话写入会话记忆（失败轮也记：下一轮才知道用户在补充什么）"""
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": reply})
        self._history = self._history[-HISTORY_MAX_MESSAGES:]
        self._last_active = time.time()

    async def handle(self, message: str) -> Dict[str, Any]:
        """Agent Loop 主入口（非流式包装）：消费事件流，取 done 事件"""
        result = None
        async for ev in self.handle_stream(message):
            if ev.get("type") == "done":
                result = ev
        # 所有出口都产出 done；这里兜底纯属防御（生成器被异常截断时）
        return result or {"reply": "（无响应）", "iterations": 0, "tool_calls_log": [], "state": "idle"}

    async def handle_stream(self, message: str):
        """串行化外壳：单飞轮锁，后来者等前一轮完整落地（含 _remember 写
        会话记忆）才进 loop——会话历史的入账顺序 = 用户真实发送顺序。
        前端本地排队是第一道（同 tab 只有一条请求在飞），这把锁兜住
        多标签页 / curl / Enter 连击等绕过路径。
        """
        async with self._turn_lock:
            async for ev in self._run_turn(message):
                yield ev

    async def _run_turn(self, message: str):
        """Agent Loop 流式版：产出事件流，SSE 端点直通前端

        事件协议（前端渲染依据）：
          status(stage=thinking)      每轮 LLM 调用前——前端转圈"正在思考"
          status(stage=tool, tool=名) 每次工具执行前——前端换工具专属文案（正在记下/翻翻日程…）
          delta(text)                 最终回复的增量分片（只有文本轮产生）
          reset                       罕见：先流了文本又转工具调用——前端清回等待态
          done(reply,...)             权威回复：诚实不变量的覆写发生在 done 之前，前端以它为准

        状态流转：idle → processing → responding → idle
        """
        self.state = AgentState.PROCESSING
        t_start = time.monotonic()
        today = date_type.today().isoformat()
        weekday = WEEKDAY_NAMES[date_type.today().weekday()]
        now = datetime.now().strftime("%H:%M")
        system_content = SYSTEM_PROMPT.format(today=today, weekday=weekday, now=now)

        # 短期记忆插在 system 与本轮 user 之间——反问后的补充信息才有上下文
        history = self._history_messages()
        messages = [
            {"role": "system", "content": system_content},
            *history,
            {"role": "user", "content": message},
        ]
        # 请求全景日志（对齐 parser 的日志规范）：用户请求 + 注入的会话记忆 + system（截断）。
        # 排查"记忆有没有带上、带的是什么"一眼可见
        hist_str = " | ".join(f"{m['role']}({len(m['content'])}字): {m['content'][:80]}" for m in history) or "（空）"
        logger.info(
            f"[interactive-agent] 请求上下文 | user: {message} | "
            f"history({len(history)}条): {hist_str} | "
            f"system({len(system_content)}字): {system_content[:300]}..."
        )

        tool_calls_log = []
        iteration = 0
        empty_retries = 0
        # 诚实不变量的判断依据：写工具本次请求是否出现过 成功/失败。
        # 失败分两类，覆写文案不同：没听懂（请补充信息）vs 存不住（稍后再试）
        write_ok = False
        write_fail_parse = write_fail_other = False
        # 同工具连续失败计数：≥2 次就在 tool 消息里注入"停止重试"指令（盲重试熔断）
        fail_streak = {}
        # 请求级"首字"UX 指标只打一次（跨轮，最终轮才出现文本）
        first_text_logged = False

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[interactive-agent] iteration {iteration}, state={self.state.value}, messages={len(messages)}")

            # 调用 LLM（带 tools 定义），记录 prompt 日志
            logger.info(
                f"[interactive-agent] LLM 调用 iteration {iteration} | "
                f"messages({len(messages)}条): "
                + " | ".join(
                    f"{m['role']}({len(str(m.get('content', '')))}字)"
                    for m in messages
                )
            )
            t0 = time.monotonic()
            yield {"type": "status", "stage": "thinking"}
            # 全程流式：工具轮只是不产生 delta，文本轮的增量直接透传前端（乐观直出，
            # 先文本后工具的罕见乱序由 reset 事件兜底）
            content_parts: List[str] = []
            streamed_text = False
            tool_acc: Dict[int, Dict[str, str]] = {}
            usage = None
            first_chunk_at = None   # 首 chunk 到达：≈prefill+排队（思考型模型此刻思维链已开始流）
            first_text_at = None    # 首个可见文本分片：= 本轮思考结束点（仅文本轮有意义）
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    temperature=0.3,
                    stream=True,
                    stream_options={"include_usage": True},
                    # qwen3 思考型模型：交互层开思考（路由/排班要推理），但封预算防单轮深思拖响应
                    # （parser 走 base.py 固定不思考，不受此影响）
                    extra_body=_INTERACTIVE_EXTRA,
                )
                async for chunk in stream:
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if not chunk.choices:
                        continue  # include_usage 时最后一个块 choices 为空、只带用量
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue
                    if delta.content:
                        if first_text_at is None:
                            first_text_at = time.monotonic()
                            if not first_text_logged:
                                first_text_logged = True
                                logger.info(
                                    f"[interactive-agent] 首字 {(first_text_at - t_start) * 1000:.0f}ms"
                                    f"（距请求开始，用户此刻看到第一个字）")
                        content_parts.append(delta.content)
                        streamed_text = True
                        yield {"type": "delta", "text": delta.content}
                    for tc in (delta.tool_calls or []):
                        acc = tool_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["name"] += tc.function.name
                            if tc.function.arguments:
                                acc["arguments"] += tc.function.arguments
                    # 思考增量（reasoning_content）不产出：那是后台思考，不是给用户的回复
            except Exception as e:
                # LLM 调用失败（断网/超时/服务端错误）：降级退出，绝不向 FastAPI 抛异常。
                # 文案按是否已执行过工具区分——记忆应用最怕用户不知道"记没记住"。
                logger.error(
                    f"[interactive-agent] LLM 调用失败({(time.monotonic()-t0)*1000:.0f}ms, "
                    f"iteration {iteration}): {type(e).__name__}: {e}"
                )
                self.state = AgentState.IDLE
                if tool_calls_log:
                    reply = "刚才的记录已经保存了，但回复生成时网络出了点问题——可以稍后再问我确认一下～"
                else:
                    reply = "网络好像不太稳定，这条我还没记住，稍后再发一遍好吗？"
                self._remember(message, reply)  # 降级轮也进记忆：用户重发时才知道上一轮发生了什么
                yield {
                    "type": "done",
                    "reply": reply,
                    "iterations": iteration,
                    "tool_calls_log": tool_calls_log,
                    "state": self.state.value,
                    "degraded": True,
                }
                return

            # 流结束：拼出本轮完整响应（content / tool_calls 二者通常只有其一）
            elapsed_ms = (time.monotonic() - t0) * 1000
            tok = (f", prompt={usage.prompt_tokens}tok, completion={usage.completion_tokens}tok"
                   if usage else "")
            ttft = f", 首chunk={(first_chunk_at - t0) * 1000:.0f}ms" if first_chunk_at else ""
            ttt = f", 首字={(first_text_at - t0) * 1000:.0f}ms" if first_text_at else ""
            content = "".join(content_parts)
            tool_calls_sorted = [tool_acc[i] for i in sorted(tool_acc)]
            if tool_calls_sorted:
                tc_names = [t["name"] for t in tool_calls_sorted]
                logger.info(f"[interactive-agent] LLM 响应({elapsed_ms:.0f}ms, model={self.model}{tok}{ttft}) → tool_calls: {tc_names}")
                if streamed_text:
                    yield {"type": "reset"}
            else:
                logger.info(f"[interactive-agent] LLM 响应({elapsed_ms:.0f}ms, model={self.model}{tok}{ttft}{ttt}, 文本): {content[:200]}")

            # 构建 assistant message dict（可能包含 tool_calls）
            msg_dict = {"role": "assistant", "content": content}
            if tool_calls_sorted:
                msg_dict["tool_calls"] = [
                    {
                        "id": t["id"],
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "arguments": t["arguments"],
                        },
                    }
                    for t in tool_calls_sorted
                ]
            messages.append(msg_dict)

            # 退化响应重试：思考型模型偶发把答案留在思维链里，content 为空且无 tool_call
            if not tool_calls_sorted and not content.strip():
                if empty_retries < 1:
                    empty_retries += 1
                    messages.pop()  # 不把空 assistant 消息留在历史里
                    logger.warning(f"[interactive-agent] LLM 返回空响应（思维链吞了答案），重试第 {empty_retries} 次")
                    continue

            # 终止条件：无 tool_calls → 进入 responding 状态
            if not tool_calls_sorted:
                self.state = AgentState.RESPONDING
                final_reply = content
                if not final_reply.strip():
                    # 兜底：LLM 返回空文本时，根据是否执行过工具给出基本回复
                    final_reply = "搞定啦～" if tool_calls_log else "我在的，想聊点什么？"
                    logger.warning(f"[interactive-agent] LLM 返回空文本，使用兜底回复")
                # 诚实不变量：写工具只失败未成功时，"已记录"式确认必为幻觉，代码直接覆写——
                # 是否落库不交给概率（实测：parse_and_record 连败 4 次后，模型仍回复"已为你记下 9月1日例会"）
                if (write_fail_parse or write_fail_other) and not write_ok:
                    logger.warning(f"[interactive-agent] 写工具全部失败，覆写疑似成功幻觉的回复: {final_reply[:60]}")
                    if write_fail_other:
                        final_reply = "抱歉，这条没能保存成功——存储好像出了点问题，我没有把它记下来，稍后再试一次好吗？"
                    else:
                        final_reply = "这句我没太听懂，为了不记错就先没有保存——可以补充一下时间和具体事情，我们再试一次吗？"
                logger.info(f"[interactive-agent] 终止于 iteration {iteration}，无 tool_call，进入 responding")
                logger.info(f"[interactive-agent] 请求完成，总耗时 {(time.monotonic()-t_start)*1000:.0f}ms，{iteration} 轮迭代")

                self.state = AgentState.IDLE
                self._remember(message, final_reply)
                yield {
                    "type": "done",
                    "reply": final_reply,
                    "iterations": iteration,
                    "tool_calls_log": tool_calls_log,
                    "state": self.state.value,
                }
                return

            # 有 tool_calls → 逐个执行
            for tc in tool_calls_sorted:
                tool_name = tc["name"]
                tool_args_str = tc["arguments"]
                tool_call_id = tc["id"]
                yield {"type": "status", "stage": "tool", "tool": tool_name}

                # 标注工具类型
                tool_kind = "任务工具(sub-agent)" if tool_name in TASK_TOOLS else "基础工具"
                logger.info(f"[interactive-agent] 调用 {tool_kind}: {tool_name}({tool_args_str})")

                try:
                    tool_args = json.loads(tool_args_str)
                    func = TOOL_FUNCTION_MAP.get(tool_name)

                    if func is None:
                        result = f"错误：未知工具 {tool_name}"
                    elif tool_name in ASYNC_TOOLS:
                        result = await func(**tool_args)
                    else:
                        result = func(**tool_args)

                    result_flat = " ".join(str(result).split())  # 多行结果压成一行，保持日志行格式
                    # 1000 字：工具返回基本可见（记忆素材的全文另有 [search] 素材 行）
                    logger.info(f"[interactive-agent] {tool_name} 执行完成: {result_flat[:1000]}")
                    fail_streak[tool_name] = 0
                    if tool_name in WRITE_TOOLS:
                        write_ok = True

                except Exception as e:
                    result = f"工具执行错误: {str(e)}"
                    logger.error(f"[interactive-agent] {tool_name} 执行失败: {e}")
                    if tool_name in WRITE_TOOLS:
                        if isinstance(e, ParseError):
                            write_fail_parse = True
                        else:
                            write_fail_other = True
                    # 盲重试熔断：确定性故障（如存储损坏）重试必然再败
                    fail_streak[tool_name] = fail_streak.get(tool_name, 0) + 1
                    if fail_streak[tool_name] >= 2:
                        result += "\n（该工具已连续失败 2 次以上，此错误重试无效。不要再调用它，直接如实告知用户本次操作没有成功。）"

                tool_calls_log.append({
                    "tool": tool_name,
                    "kind": tool_kind,
                    "args": tool_args_str,
                    "result_summary": result[:100],
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                })

        # 超过最大迭代
        logger.warning(f"[interactive-agent] 达到最大迭代 {self.max_iterations}，强制终止")
        self.state = AgentState.IDLE
        reply = "抱歉，处理步骤太多了，请简化一下你的请求。"
        self._remember(message, reply)
        yield {
            "type": "done",
            "reply": reply,
            "iterations": iteration,
            "tool_calls_log": tool_calls_log,
            "state": self.state.value,
        }
