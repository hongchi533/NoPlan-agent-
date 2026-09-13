"""FastAPI 入口

架构：
  InteractiveAgent — 交互 agent，持有基础工具+任务工具，agent loop
  BackgroundAgent  — 后台 agent：夜间记忆维护 + 晨览/周结组稿 + 定时委托执行
  Scheduler        — 统一调度器：内置任务 + 用户任务(tasks.json) + 派生日程提醒三源，
                     到点 fire 写信箱(outbox)，送达由前端轮询领取

生命周期：
  startup  = 校验配置 → 登记 LLM 连接池清理 → 连接高德 MCP（白名单挂载）
             → 启动统一调度器（接管原独立夜间任务）
  shutdown = 逆序释放：停调度器 → 关 MCP → 关 LLM 连接池

端点：
  GET  /              → UI 页面
  POST /api/chat      → InteractiveAgent 非流式入口（调试/兼容，无前端调用方）
  POST /api/chat/stream → 流式聊天入口（SSE：等待状态 + 回复逐字直出）
  GET  /api/init      → 前端初始化数据（含主动推送）
  GET  /api/calendar  → 日历视图数据
  GET/POST/DELETE /api/tasks       → 用户定时任务（提醒页表单与 agent 工具共用入口）
  GET  /api/reminders/due          → 前端 tick 领取信箱（保鲜期内返回，超期静默）
  POST /api/scheduler/run/{job_id} → 手动触发任意任务（开发/演示）
  DELETE /api/events/{id} → 前端手动删除事件（UI 点选=用户亲自指认，不过 agent loop）
  DELETE /api/moods/{id}  → 前端手动删除心情（同上）
"""
import os
import json
import logging
import calendar
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import date, timedelta
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.responses import Response

from app.agent.interactive import InteractiveAgent
from app.agent.background import BackgroundAgent
from app.agent import tools
from app.agent.mcp_client import McpHttpClient
from app.scheduler import Scheduler
from app.config import DASHSCOPE_API_KEY, MCP_AMAP_ENABLED, AMAP_MCP_URL, MCP_AMAP_TOOL_WHITELIST
from app.store import db
from app.models.schemas import MOOD_EMOJI

# ─── 日志配置 ─────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],  # 由启动命令重定向到 log/server.log
)

# mcp SDK 内部的 httpx2 会在 INFO 级打请求行（含完整 URL）——
# 高德 key 拼在 URL 上，压到 WARNING 防止 key 泄露进 server.log
logging.getLogger("httpx2").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ─── Agent 实例 ─────────────────────────────────────
interactive_agent = InteractiveAgent()
background_agent = BackgroundAgent()

# 高德 MCP 客户端（startup 连接挂载 / shutdown 关闭；连接失败仅天气功能降级）
amap_mcp = McpHttpClient("amap", AMAP_MCP_URL, MCP_AMAP_TOOL_WHITELIST, service_label="天气服务")

# 统一调度器：所有"时间到"类任务的计时大脑（app/scheduler.py）。
# fire 后的 handler 委托 background_agent；送达统一写信箱，由前端轮询领取
scheduler = Scheduler(background_agent)
# 注入工具层：对话里"明早七点提醒我…"走 register_task 工具进同一调度器
# （依赖注入而非 import，tools 与 scheduler 不在 import 期互相拉起）
tools.set_task_scheduler(scheduler)


# ─── 应用生命周期：lifespan 一个函数管 startup + shutdown ───
# yield 之前 = startup（按依赖顺序建立并登记资源），
# yield 之后 = shutdown（栈逆序释放：停夜间任务 → 关 MCP → 关 LLM 连接池）。

@asynccontextmanager
async def lifespan(app: FastAPI):
    stack = AsyncExitStack()
    try:
        logger.info("[startup] 记忆助手启动")
        if not DASHSCOPE_API_KEY:
            logger.warning("[startup] DASHSCOPE_API_KEY 为空：LLM 功能将不可用（检查 .env）")
        logger.info(f"[startup] InteractiveAgent 就绪，工具: {len(tools.TOOL_DEFINITIONS)} 个")
        logger.info(f"[startup]   基础工具: {tools.BASIC_TOOLS}")
        logger.info(f"[startup]   任务工具: {tools.TASK_TOOLS}")
        logger.info("[startup] BackgroundAgent 就绪")

        # ① LLM 连接池（3 个 AsyncOpenAI 各持一个 httpx 连接池）：
        #    登记最早 → 拆除最晚（夜间任务和 MCP 都不依赖它先关）
        #    （检索层 2026-09-04 起无 LLM，已不在名单里）
        async def _close_llm_clients():
            await interactive_agent.client.close()
            await background_agent.client.close()
            await tools.parser_agent.client.close()
            logger.info("[shutdown] LLM 连接池已关闭")

        stack.push_async_callback(_close_llm_clients)

        # ② 高德 MCP：连接 + 白名单工具挂载（失败只影响天气功能，不阻塞启动）
        if MCP_AMAP_ENABLED:
            try:
                kept = await amap_mcp.connect()
                for t in kept:
                    tools.register_dynamic_tool(
                        name=t.name,
                        description=t.description or "",
                        parameters=t.input_schema,
                        func=amap_mcp.make_caller(t.name),
                    )
                logger.info(f"[startup] 高德 MCP 就绪，挂载 {len(kept)} 个白名单工具")
            except Exception as e:
                logger.warning(f"[startup] 高德 MCP 连接失败，天气工具本次不挂载: {e}")
        stack.push_async_callback(amap_mcp.close)  # 幂等：未连接时空操作

        # ③ 统一调度器（原独立的夜间调度已收编：语义不变，旧 marker 首启自动迁移）。
        #    内置夜间/晨览/周结 + 用户任务 + 派生日程提醒，见 app/scheduler.py。
        #    调度器内部自持 asyncio 任务强引用（弱引用会被 GC，CPython 文档警告）
        await scheduler.start()

        async def _stop_scheduler():
            await scheduler.stop()

        stack.push_async_callback(_stop_scheduler)

        yield  # ← 应用运行期（uvicorn 在此期间服务请求）
    finally:
        # shutdown：逆序执行账本 → _stop_scheduler → amap_mcp.close → _close_llm_clients
        await stack.aclose()
        logger.info("[shutdown] 记忆助手已关闭")


app = FastAPI(title="记忆助手", lifespan=lifespan)


# ─── 全局异常兜底 ───────────────────────────────────
# 任何未捕获异常（DB 文件损坏、代理层意外等）都返回友好 JSON：
# 前端永远拿不到 500 错误页。返回 200 + ok:false 而非 500，
# 是让前端走统一的 reply 渲染通道，不用为错误单独建分支。
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"[api] 未捕获异常 {request.url.path}: {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=200, content={"ok": False, "reply": "服务开小差了，请稍后再试～"})


# ─── 定时任务与提醒（统一调度器的 API 面）────────────
# 调度本体与分片睡眠/补跑/信箱语义都在 app/scheduler.py，此处只是 HTTP 面。
# 表单注册（提醒页）与对话注册（agent 的 register_task 工具）殊途同归，
# 都进 scheduler.register_user_task 同一入口。

class TaskCreateRequest(BaseModel):
    mode: str = "direct"            # direct=到点提醒（零 LLM） / agent=到点起 loop
    at: str                         # ISO 本地时间（表单只支持一次性任务）
    text: Optional[str] = None      # direct 模式：提醒文案
    prompt: Optional[str] = None    # agent 模式：自然语言委托


@app.get("/api/tasks")
async def list_tasks():
    """提醒事项页数据：用户注册的任务（含下次触发时间）"""
    return {"ok": True, "tasks": scheduler.list_user_tasks()}


@app.post("/api/tasks")
async def create_task(req: TaskCreateRequest):
    """表单注册（仅一次性 direct/agent）。重复规则（"每周三早上"）走对话注册——LLM 才懂"""
    try:
        payload = ({"text": (req.text or "").strip()} if req.mode == "direct"
                   else {"prompt": (req.prompt or "").strip()})
        task = scheduler.register_user_task(req.mode, {"type": "once", "at": req.at}, payload)
        logger.info(f"[api] 表单注册定时任务: {task['id']} {task['mode']} {task['schedule']}")
        return {"ok": True, "task": task}
    except ValueError as e:
        return {"ok": False, "reply": f"没有注册成功：{e}"}


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """取消提醒。一次性任务触发即删（存在=未触发），取消和完成是同一个动作"""
    deleted = scheduler.delete_user_task(task_id)
    if deleted is None:
        return {"ok": False, "reply": "这条提醒不存在，可能已经触发或删除过了～"}
    logger.info(f"[api] 取消定时任务: {deleted['id']} {deleted['payload']}")
    return {"ok": True, "deleted": deleted}


@app.get("/api/reminders/due")
async def reminders_due():
    """前端 30s tick 领取信箱（确认式投递，at-least-once）：
    领取=标记 pending 不删除；展示成功后前端调 /api/reminders/ack 才清账；
    90s 未确认自动重领（HTTP 响应丢失时用户仍收得到），前端按 id 去重防重弹；
    超保鲜期（提醒 30min / 报告 12~24h）的静默作废——页面关着期间错过的，
    打开时不被旧通知轰炸（用户定稿语义）"""
    return {"ok": True, "items": scheduler.claim_due()}


class AckRequest(BaseModel):
    ids: List[str]


@app.post("/api/reminders/ack")
async def reminders_ack(req: AckRequest):
    """前端展示成功后确认送达（s12 pending_delivery 的确认半边）"""
    return {"ok": True, "acked": scheduler.ack(req.ids)}


@app.post("/api/scheduler/run/{job_id}")
async def scheduler_run(job_id: str):
    """手动触发任意任务（开发/演示用）。记账语义不变：成功同样记当天 marker"""
    logger.info(f"[api] 手动触发任务: {job_id}")
    return await scheduler.fire_now(job_id)


@app.post("/api/nightly")
async def run_nightly_manual():
    """手动触发夜间整理（保留旧路径兼容；内部已走统一调度器，成功记当天 marker）"""
    logger.info("[api] 手动触发夜间任务")
    return await scheduler.fire_now("builtin:nightly")


# ─── UI 静态文件 ─────────────────────────────────────

UI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(UI_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/style.css")
async def style():
    with open(os.path.join(UI_DIR, "style.css"), "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/css")


@app.get("/app.js")
async def script():
    with open(os.path.join(UI_DIR, "app.js"), "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="application/javascript")


@app.get("/journal.css")
async def journal_skin():
    # 手帐模式皮肤（body.journal 作用域，默认关；开关在 index.html 的 📔 按钮）
    with open(os.path.join(UI_DIR, "journal.css"), "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="text/css")


# ─── InteractiveAgent 入口 ───────────────────────────

class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """InteractiveAgent 非流式入口（调试/兼容用：curl 调试最方便，也是流式翻车的回退路径）。
    正常前端流量走 /api/chat/stream；此端点当前无调用方，保留作调试面"""
    result = await interactive_agent.handle(req.message)
    return {
        "ok": True,
        "reply": result["reply"],
        "iterations": result["iterations"],
        "tool_calls_log": result["tool_calls_log"],
        "state": result["state"],
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式版：SSE 直通 agent 事件流（status/delta/reset/done，见 handle_stream 协议）。
    前端聊天走这里；"""
    async def gen():
        try:
            async for ev in interactive_agent.handle_stream(req.message):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            # 响应已开始（200 + text/event-stream），全局异常 handler 接管不了，
            # 必须在流内兜底：给一个 done 让前端正常收尾
            logger.error(f"[api] 流式响应异常: {type(e).__name__}: {e}")
            fallback = {"type": "done", "ok": False, "reply": "服务开小差了，请稍后再试～",
                        "iterations": 0, "tool_calls_log": [], "state": "idle"}
            yield f"data: {json.dumps(fallback, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # 反代不缓冲，分片直达
    )


# ─── 前端初始化数据（含主动推送）──────────────────────

@app.get("/api/init")
async def init():
    """前端初始化：今日事件 + 未来事件 + 心情 + 主动推送 + mood emoji"""
    today_str = date.today().isoformat()

    # 今日事件
    today_events = db.get_events_by_date(today_str)

    # 未来事件（plan 且 date > 今天）
    all_events = db.load_events()
    future_events = [e for e in all_events if e.date > today_str and e.type == "plan"]
    future_events.sort(key=lambda e: (e.date, e.start_time or "99:99"))

    # 今日心情
    moods = db.get_moods_by_date(today_str)

    # 主动推送：时光胶囊 + 去年今天。优先读 07:00 与总览同批预组稿的叙述文案
    # （scheduler 落盘、当日有效）；今天还没组稿过（没跑到/组稿失败）才退回
    # 原文碎片，开屏永远不为卡片等 LLM。已叙述但整卡弃说的，缓存里就是空——
    # 不说权当日生效，不退原文。两条路对前端合同一致：文本可直接上屏、无标题行
    proactive = {}
    cached_cards = scheduler.today_proactive()
    if cached_cards is not None:
        if cached_cards.get("capsule"):
            proactive["capsule"] = cached_cards["capsule"]
        if cached_cards.get("last_year"):
            proactive["last_year"] = cached_cards["last_year"]
    else:
        def _body(text: str) -> str:
            # 剥首行"📅 去年今天："式标题：卡片 h3 已有，重复即累赘
            return "\n".join(text.split("\n")[1:]).strip()

        capsule_result = tools.recall_capsule()
        if "没有" not in capsule_result:
            proactive["capsule"] = _body(capsule_result)

        last_year_result = tools.recall_last_year()
        if "没有" not in last_year_result:
            proactive["last_year"] = _body(last_year_result)

    return {
        "ok": True,
        "events": [e.model_dump() for e in today_events],
        "future_events": [e.model_dump() for e in future_events],
        "moods": [m.model_dump() for m in moods],
        "proactive": proactive,
        # 今日总览（07:00 触发后落盘；今天还没生成为 null，前端不占位）
        "overview": scheduler.today_overview(),
        # mood_emoji 后端生成回复/检索文本时也要用（真正的跨层共享常量），随 init 下发避免两份；
        # 标签配色是纯视图关注点，归前端 app.js 所有，后端不下发
        "mood_emoji": MOOD_EMOJI,
    }


@app.get("/api/overview")
async def overview():
    """今日总览：页面开着时轮询用（到点生成后首页栏目自动补上，无需刷新）"""
    return {"ok": True, "overview": scheduler.today_overview()}


# ─── 手动删除（事件/心情）─────────────────────────────

@app.delete("/api/events/{event_id}")
async def delete_event(event_id: str):
    """前端手动删除：纯确定性存储操作，不经过 agent loop（无模糊指代，无需 LLM）"""
    deleted = db.delete_event(event_id)
    if deleted is None:
        # 不静默成功：事件不存在也是一种要告知的结果（可能刚在别处删过）
        return {"ok": False, "reply": "这条事件不存在，可能刚刚已经删除过了～"}
    logger.info(f"[api] 手动删除事件: {deleted.date} {deleted.start_time or ''} {deleted.title}")
    return {"ok": True, "deleted": deleted.model_dump()}


@app.post("/api/events/{event_id}/complete")
async def complete_event(event_id: str):
    """前端打勾完成：UI 点选=用户亲自指认，不过 agent loop（同手动删除的原则）。
    与 complete_plan 工具走同一个 db.update_event，语义完全一致；
    会话路径（"我跑完步了"式的模糊指代）仍归 agent 的 find_plan → complete_plan。"""
    event = db.update_event(event_id, {"type": "done"})
    if event is None:
        return {"ok": False, "reply": "这条事件不存在，可能刚刚已经完成或删除过了～"}
    logger.info(f"[api] 打勾完成事件: {event.date} {event.start_time or ''} {event.title}")
    return {"ok": True, "event": event.model_dump()}


@app.delete("/api/moods/{mood_id}")
async def delete_mood(mood_id: str):
    """前端手动删除心情（同事件：确定性操作，不过 agent loop）"""
    deleted = db.delete_mood(mood_id)
    if deleted is None:
        return {"ok": False, "reply": "这条心情不存在，可能刚刚已经删除过了～"}
    logger.info(f"[api] 手动删除心情: {deleted.date} {deleted.time or ''} {deleted.content}")
    return {"ok": True, "deleted": deleted.model_dump()}


# ─── 日历视图数据 ─────────────────────────────────────

@app.get("/api/calendar")
async def calendar_view(view: str = "month", d: Optional[str] = None):
    """日历视图数据：month / week / day"""
    today = date.today()

    if view == "month":
        return _month_view(d or today.strftime("%Y-%m"))
    elif view == "week":
        return _week_view(d or today.isoformat())
    else:
        return _day_view(d or today.isoformat())


def _month_view(month_str: str):
    """月视图：每天的事件概要"""
    year, month = [int(x) for x in month_str.split("-")]
    _, days_in_month = calendar.monthrange(year, month)

    date_from = date(year, month, 1).isoformat()
    date_to = date(year, month, days_in_month).isoformat()

    events = db.get_events_by_date_range(date_from, date_to)
    moods = db.get_moods_by_date_range(date_from, date_to)

    days = []
    for day_num in range(1, days_in_month + 1):
        d = date(year, month, day_num).isoformat()
        day_events = [e for e in events if e.date == d]
        day_moods = [m for m in moods if m.date == d]

        tags_set = set()
        for e in day_events:
            tags_set.update(e.tags)

        days.append({
            "date": d,
            "weekday": date(year, month, day_num).weekday(),
            "event_count": len(day_events),
            "has_plan": any(e.type == "plan" for e in day_events),
            "has_done": any(e.type == "done" for e in day_events),
            "tags": list(tags_set),
            "mood": day_moods[0].mood if day_moods else None,
        })

    return {
        "ok": True,
        "view": "month",
        "month": month_str,
        "days": days,
        "weekday_start": date(year, month, 1).weekday(),
    }


def _week_view(date_str: str):
    """周视图：7天的事件+心情"""
    target = date.fromisoformat(date_str)
    monday = target - timedelta(days=target.weekday())

    days = []
    for i in range(7):
        d = (monday + timedelta(days=i)).isoformat()
        day_events = db.get_events_by_date(d)
        day_moods = db.get_moods_by_date(d)

        days.append({
            "date": d,
            "weekday": i,
            "is_today": d == date.today().isoformat(),
            "events": [e.model_dump() for e in day_events],
            "moods": [m.model_dump() for m in day_moods],
        })

    return {
        "ok": True,
        "view": "week",
        "week_start": monday.isoformat(),
        "days": days,
    }


def _day_view(date_str: str):
    """日视图：一天的事件+心情详情"""
    events = db.get_events_by_date(date_str)
    moods = db.get_moods_by_date(date_str)

    return {
        "ok": True,
        "view": "day",
        "date": date_str,
        "is_today": date_str == date.today().isoformat(),
        "events": [e.model_dump() for e in events],
        "moods": [m.model_dump() for m in moods],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
