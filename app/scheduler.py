"""统一调度器：所有"时间到"类任务的计时大脑

设计原则（docs 方案定稿）：
  - 触发跟着送达通道和进程寿命走。本模块只管"何时 fire、防重、过期"，
    送达统一写入 outbox，由前端轮询领取（将来桌面 app 形态换成 OS 通知，换脸不动脑）。
  - 任务三源，各归各的事实来源：
      内置   —— 代码注册（夜间整理/晨间总览/周总结），升级即变
      用户   —— data/tasks.json（"明天九点提醒我带酒"），用户资产
      派生   —— 从今日 plan 每周期重算（前 15 分/到点/过 5 分三段提醒），不落盘：
               删/改/完成 plan 自动生效，不存在两处数据打架
  - 语义二分（注册时确定，非运行时猜）：
      durable（nightly/overview/weekly）—— 过了今天就补跑一次（重算型任务补一次即收敛），
        失败 10 分钟退避重试，成功才记账（at-least-once）
      perishable（remind/agent/plan_reminder）—— 容忍窗内补跑，超窗静默过期（错过就错过），
        fire 前先记账/先删（at-most-once，宁可漏不可轰——前端 30s 重弹事故的服务端防线）
  - 用户任务的去重按形态分两套：
      一次性：存在性=去重，先删再触发（取消和完成是同一个动作）
      周期（每天/每周/每月/每年）：按"发生时刻"记 marker，任务永续存活——
        月/年的日号超出当月天数时收敛到月末（1月31日的月任务在2月=2月28日）
  - 借自 s12 课程：意图即载荷（agent 任务存 prompt，到点才解释）、加载时响亮跳过坏条目、
    确认式投递（信箱领取只标 pending 不删，前端展示成功后 ack 才清——网络层丢了响应，
    90s 后重领，用户仍收得到；前端按 id 去重防重弹）、记账先于派发且写盘失败即中止
    （fail-closed：绝不出现"没记上账却已触发"的窗口）。

调度循环：≤60s 分片对墙上时钟——实测教训（2026-08）：macOS 合盖时单调时钟冻结
（开机 899h、monotonic 只走 101h），一次 sleep(3.5h) 的死线需要 3.5 小时"醒着的"时间，
永不触发；uvicorn --reload 每次改码还重置倒计时。分片 + 墙上时钟重对表 +
错过补跑 + 当日标记，四件套缺一不可。
"""
import asyncio
import calendar as _cal
import json
import logging
import os
import secrets
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from filelock import FileLock

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

# ─── 文件 ─────────────────────────────────────────────
TASKS_FILE = os.path.join(DATA_DIR, "tasks.json")            # 用户注册任务（唯一事实源）
STATE_FILE = os.path.join(DATA_DIR, "scheduler_state.json")  # 记账：{markers: {job_id: {fired, attempt}}}
OUTBOX_FILE = os.path.join(DATA_DIR, "notify_outbox.json")   # 待送达信箱：[{id, kind, text, created_at}]
OVERVIEW_FILE = os.path.join(DATA_DIR, "overview.json")      # 每日总览：{date, text}，当日覆盖
PROACTIVE_FILE = os.path.join(DATA_DIR, "proactive.json")    # 回忆卡片文案：{date, capsule, last_year}，当日覆盖
LEGACY_NIGHTLY_FILE = os.path.join(DATA_DIR, "nightly_last_run.txt")  # 旧夜间 marker，首启迁移

# ─── 参数 ─────────────────────────────────────────────
SLICE_SECONDS = 60          # 分片睡眠步长（提醒精度 = 1 分钟，足够）
RETRY_SECONDS = 600         # durable 任务失败退避
ONCE_TOLERANCE_MIN = 30     # 一次性用户任务的迟到容忍窗（进程没开，开醒后 30 分钟内仍补跑）
# 派生提醒三阶段：(阶段, 触发点相对开始时间的偏移分钟, 出窗偏移分钟)。
# 每阶段独立 id、独立记账、独立容忍窗 [触发点, 出窗点)——正常 tick 到三次全响；
# 窗口外静默跳过（perishable：错过就错过，不补轰）。到点阶段 5 分钟、过点阶段
# 10 分钟补跑容忍，盖住笔记本睡眠/进程重启的场景
PLAN_REMIND_PHASES = [
    ("pre", -15, -1),   # 「还有 15 分钟到点」；出窗提前 1 分钟——临近让位给 due，不背靠背连响
    ("due", 0, 5),      # 「到点啦，可以开始」
    ("post", 5, 15),    # 「已经过去 5 分钟啦」
]
OUTBOX_TTL_SECONDS = {      # 信箱保鲜期：超时被领取不弹（3 小时前的"该带酒了"是噪音）
    "remind": 30 * 60,
    "plan_reminder": 30 * 60,
    "agent": 12 * 3600,     # agent 任务结果更像报告，当天可见
    "overview": 12 * 3600,
    "weekly": 24 * 3600,
}
DEFAULT_OUTBOX_TTL = 12 * 3600
RECLAIM_SECONDS = 90       # 领取后未确认的重领等待（3 个前端 tick，覆盖"响应在网络层丢了"）

DURABLE_KINDS = {"nightly", "overview", "weekly"}   # 补跑 + 退避重试
PERISHABLE_KINDS = {"remind", "agent", "plan_reminder"}  # 容忍窗 + at-most-once

# 内置任务：代码即事实源（Phase 3 的 overview/weekly 由 background agent 提供内容）
BUILTIN_JOBS: List[dict] = [
    {"id": "builtin:nightly", "kind": "nightly",
     "schedule": {"type": "daily", "hour": 2, "minute": 0}},
    {"id": "builtin:overview", "kind": "overview",
     "schedule": {"type": "daily", "hour": 7, "minute": 0}},
    {"id": "builtin:weekly", "kind": "weekly",
     "schedule": {"type": "weekly", "weekday": 6, "hour": 20, "minute": 0}},  # 周日 20:00
]


# ─── 文件小助手（原子写 + 损坏隔离，纪律同 store/db.py）────

def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError) as e:
        corrupt = f"{path}.corrupt.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            os.replace(path, corrupt)
            logger.error(f"[scheduler] {os.path.basename(path)} 损坏（{e}），已隔离为 {os.path.basename(corrupt)}，使用默认值重开")
        except OSError:
            logger.error(f"[scheduler] {os.path.basename(path)} 损坏且无法隔离: {e}")
        return default


def _write_json(path: str, data: Any) -> None:
    """原子写 + 失败重试一次。持久化失败必须向上抛（借 s12：写盘失败不能让内存态
    越过磁盘态）——调度器的记账写都发生在派发之前，抛异常=本次不触发，下一片重试。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last_err: Exception = RuntimeError("unreachable")
    for attempt in range(2):
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with FileLock(f"{path}.lock"):
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)  # 同目录原子替换，读方永远看到完整文件
            return
        except Exception as e:
            last_err = e
            logger.warning(f"[scheduler] 写 {os.path.basename(path)} 失败 (attempt {attempt + 1}): "
                           f"{type(e).__name__}: {e}")
    raise last_err


def _parse_dt(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _clamp_day(year: int, month: int, day: int) -> date:
    """月/年任务的实际日期：日号超出当月天数时收敛到月末（1月31日的月任务在2月=2月28日）"""
    return date(year, month, min(day, _cal.monthrange(year, month)[1]))


def _at_hm(d: date, hour: int, minute: int) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute)


class Scheduler:
    """统一调度器。main.py 的 lifespan 负责启停；handler 委托 background_agent。"""

    def __init__(self, background_agent):
        self._bg = background_agent
        self._task: Optional[asyncio.Task] = None
        self._firing: set = set()   # 正在派发中的 job_id（防重复入队，见 _fire）
        self._migrate_legacy_marker()

    # ─── 生命周期 ─────────────────────────────────────

    async def start(self):
        # 持有强引用：事件循环对 task 只持弱引用，无引用可能被 GC（CPython 文档警告）
        self._task = asyncio.create_task(self._run())
        logger.info(f"[scheduler] 启动：内置 {len(BUILTIN_JOBS)} 项 + 用户任务 + 派生提醒")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            logger.info("[scheduler] 已停止")

    def _migrate_legacy_marker(self):
        """旧 nightly_last_run.txt → state.markers（不迁移会导致当晚夜间任务重复跑）"""
        if os.path.exists(STATE_FILE) or not os.path.exists(LEGACY_NIGHTLY_FILE):
            return
        legacy = _read_json(LEGACY_NIGHTLY_FILE, "")
        if isinstance(legacy, str) and legacy.strip():
            state = {"markers": {"builtin:nightly": {"fired": f"{legacy.strip()}T02:00:00"}}}
            _write_json(STATE_FILE, state)
            logger.info(f"[scheduler] 已迁移旧夜间标记: {legacy.strip()}")

    # ─── 主循环 ───────────────────────────────────────

    async def _run(self):
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # tick 级兜底：调度循环本身绝不能死（死 = 所有定时任务全灭）
                logger.error(f"[scheduler] tick 异常（已吞，下一片继续）: {type(e).__name__}: {e}")
            await asyncio.sleep(SLICE_SECONDS)

    async def _tick(self):
        now = datetime.now()
        due: List[tuple] = []   # (fire_at, job)
        for job in self._collect_jobs(now):
            fire_at = self._due_at(job, now)
            if fire_at is not None:
                due.append((fire_at, job))
        due.sort(key=lambda x: x[0])   # 多个同时到期按应触发时间先后
        for fire_at, job in due:
            await self._fire(job, now, force=False, occurrence=fire_at)

    # ─── 任务收集（三源）──────────────────────────────

    def _collect_jobs(self, now: datetime) -> List[dict]:
        jobs: List[dict] = [dict(j) for j in BUILTIN_JOBS]
        jobs.extend(self._load_user_tasks())
        jobs.extend(self._derived_jobs())
        return jobs

    def _load_user_tasks(self) -> List[dict]:
        tasks = _read_json(TASKS_FILE, [])
        valid = []
        for t in tasks if isinstance(tasks, list) else []:
            err = self._validate_task(t)
            if err:
                logger.warning(f"[scheduler] 跳过无效用户任务 {t.get('id', '?')}: {err}")
                continue
            valid.append(t)
        return valid

    def _derived_jobs(self) -> List[dict]:
        """今日 plan 的到点提醒（三阶段）——每周期重算，plan 删除/完成/改期自动生效"""
        from app.store import db  # 延迟导入：scheduler 不应在 import 期拉起存储层
        today = date.today().isoformat()
        out = []
        for e in db.get_events_by_date(today):
            if e.type == "plan" and e.start_time:
                for phase, _offset, _leave in PLAN_REMIND_PHASES:
                    out.append({
                        "id": f"planrm:{e.id}:{today}:{phase}",
                        "kind": "plan_reminder",
                        "schedule": {"type": "window", "date": today,
                                     "start_time": e.start_time, "phase": phase},
                        "payload": {"title": e.title, "start_time": e.start_time},
                    })
        return out

    # ─── due 判定：返回应触发时刻（None = 不到点）──────

    def _occurrence(self, job: dict, now: datetime) -> datetime:
        """job 当前对应的"发生时刻"（最近一次 ≤ now 的目标点，本周期未到则为未来时刻）。
        marker 记发生时刻而非 fire 时刻——周结在周一补跑周日那次时，若记 fire 时刻（周一），
        下一片重算发生时刻仍是周日，日期比对不上会无限重发。"""
        sched = job["schedule"]
        stype = sched.get("type")
        h, m = int(sched.get("hour", 0)), int(sched.get("minute", 0))

        if stype == "weekly":
            days_back = (now.weekday() - int(sched["weekday"])) % 7
            return _at_hm((now - timedelta(days=days_back)).date(), h, m)

        if stype == "monthly":
            this = _at_hm(_clamp_day(now.year, now.month, int(sched["day"])), h, m)
            if this <= now:
                return this
            py, pm = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
            return _at_hm(_clamp_day(py, pm, int(sched["day"])), h, m)

        if stype == "yearly":
            this = _at_hm(_clamp_day(now.year, int(sched["month"]), int(sched["day"])), h, m)
            if this <= now:
                return this
            return _at_hm(_clamp_day(now.year - 1, int(sched["month"]), int(sched["day"])), h, m)

        # daily（以及兜底）：今天的目标点
        return _at_hm(now.date(), h, m)

    def _due_at(self, job: dict, now: datetime) -> Optional[datetime]:
        kind, sched = job["kind"], job["schedule"]
        stype = sched.get("type")

        if stype in ("daily", "weekly", "monthly", "yearly"):
            # 周期通用：算"最近一次发生时刻"，已记账跳过；perishable 超容忍窗跳过本次
            target = self._occurrence(job, now)
            if target > now:                      # 本周期还没到点
                return None
            if self._fired_for(job["id"], target):
                return None
            if self._in_backoff(job["id"], now):
                return None
            if kind in PERISHABLE_KINDS and (now - target) > timedelta(minutes=ONCE_TOLERANCE_MIN):
                # 周期版"宁可漏"：进程 15 点才醒就不补早上 8 点的药——
                # 给本次 occurrence 记账（任务保留），下一个周期照常
                self._set_marker(job["id"], target)
                logger.info(f"[scheduler] 周期任务错过容忍窗，跳过本次: {job['id']} @ {target}")
                return None
            return target                         # durable 只补跑最近一次错过（更早的自然跳过）

        if stype == "once":
            at = _parse_dt(sched.get("at", ""))
            if at is None:
                return None
            tolerance = timedelta(minutes=ONCE_TOLERANCE_MIN)
            if now < at:
                return None
            if now > at + tolerance:   # 超窗：静默过期（存在即待办，过期先清掉再走人）
                self._remove_user_task(job["id"], reason="expired")
                return None
            return at                  # 存在性即去重：fire 时先删

        if stype == "window":          # 派生提醒三阶段，半开窗 [触发点, 出窗点)
            start = _parse_dt(f"{sched['date']}T{sched['start_time']}")
            if start is None:
                return None
            ph = next((p for p in PLAN_REMIND_PHASES if p[0] == sched.get("phase")), None)
            if ph is None:
                logger.warning(f"[scheduler] planrm 阶段缺失: {job['id']}")
                return None
            enter = start + timedelta(minutes=ph[1])
            leave = start + timedelta(minutes=ph[2])
            if now < enter or now >= leave:
                return None            # 超窗自然出局，无需记账（错过就错过）
            if self._fired_for(job["id"], now):
                return None
            return enter

        logger.warning(f"[scheduler] 未知 schedule 类型: {stype} ({job['id']})")
        return None

    # ─── fire ─────────────────────────────────────────

    async def _fire(self, job: dict, now: datetime, force: bool = False,
                    occurrence: Optional[datetime] = None):
        kind, jid = job["kind"], job["id"]
        # 防重复入队（借 s12 的 minute_marker 同位防线）：tick 与手动触发并发时，
        # 同一 job 只允许一个派发在飞；二次请求直接拒，语义交给下一次 tick 收敛
        if jid in self._firing:
            logger.warning(f"[scheduler] {jid} 已在触发中，拒绝并发触发")
            return {"ok": False, "error": "已在触发中"}
        self._firing.add(jid)
        try:
            logger.info(f"[scheduler] fire {jid} (kind={kind})")
            # perishable：先记账（at-most-once；宁可漏，不可轰）。
            # 记账写盘失败会抛异常 → 不派发 → 下一片重试（fail-closed，
            # 绝不出现"没记上账却已触发"的窗口）
            if kind == "remind" or kind == "agent":
                if job["schedule"].get("type") == "once":
                    # 一次性：存在性=去重，先删再触发（取消和完成是同一个动作）
                    self._remove_user_task(jid, reason="fired")
                else:
                    # 周期：按本次发生时刻记账，任务存活到下一周期
                    self._set_marker(jid, occurrence or now)
            elif kind == "plan_reminder":
                self._set_marker(jid, now)
            if kind == "overview":
                # overview 双轨产出：横幅版进信箱（现状不变），栏目版落盘首页卡片
                banner, card = await self._bg.run_overview()
                text = banner
                if card:
                    self._save_overview(card, now)   # 首页栏目读它：当日覆盖 = 每日刷新
                # 第三轨：回忆卡片（时光胶囊/去年今天）预组稿。失败只丢今天的卡片、
                # 退回原文展示，不连累横幅重发（durable 重试会把已送达的横幅再弹一次）；
                # 组稿成功则整份判定落盘——整卡弃说也是判定，不退原文
                try:
                    proactive_cards = await self._bg.run_proactive_cards()
                    if proactive_cards is not None:
                        self._save_proactive(proactive_cards, now)
                except Exception as e:
                    logger.error(f"[scheduler] 回忆卡片组稿失败，今日退回原文展示: {type(e).__name__}: {e}")
            else:
                text = await self._dispatch(job, now)
            if kind in DURABLE_KINDS:
                # durable：成功才记账（at-least-once），记发生时刻
                self._set_marker(jid, occurrence or now)
            if text:
                self._outbox_push(kind, text)
            logger.info(f"[scheduler] fire 完成 {jid}" + (f" → 信箱: {text[:60]}" if text else ""))
            return {"ok": True, "text": text}
        except Exception as e:
            logger.error(f"[scheduler] fire 失败 {jid}: {type(e).__name__}: {e}")
            if kind in DURABLE_KINDS:
                self._set_attempt(jid, now)  # 退避后重试
            elif kind == "agent":
                # 不静默：agent 任务失败要让用户知道（结果预期落空比没结果更糟）
                self._outbox_push("agent", f"⚠️ 定时任务执行失败了（{job.get('payload', {}).get('prompt', '')[:40]}…），可以稍后再试一次")
            return {"ok": False, "error": str(e)}
        finally:
            self._firing.discard(jid)

    async def _dispatch(self, job: dict, now: datetime) -> Optional[str]:
        """handler 注册表：kind → 执行体。返回文本则进信箱，None = 纯内部维护"""
        kind = job["kind"]
        if kind == "nightly":
            self._sweep_outbox()                       # 顺手清陈年信箱
            await self._bg.run_nightly()
            return None
        if kind == "weekly":
            return await self._bg.run_weekly()
        if kind == "overview":
            # 双轨在 _fire 里展开（横幅+栏目两份产出），不走这里的单文本通道
            raise ValueError("overview 双轨派发在 _fire 处理，不应走到 _dispatch")
        if kind == "remind":
            return job["payload"].get("text") or "（空提醒）"
        if kind == "agent":
            return await self._bg.run_task(job["payload"]["prompt"])
        if kind == "plan_reminder":
            return self._plan_reminder_text(job, now)
        raise ValueError(f"未注册的 kind: {kind}")

    def _plan_reminder_text(self, job: dict, now: datetime) -> str:
        """三阶段各说各的话：pre 报倒计时、due 报到点、post 轻描淡写提一句"""
        p = job["payload"]
        phase = job["schedule"].get("phase", "pre")
        start_min = _parse_dt(f"2000-01-01T{p['start_time']}").hour * 60 + _parse_dt(f"2000-01-01T{p['start_time']}").minute
        now_min = now.hour * 60 + now.minute
        diff = start_min - now_min
        if phase == "pre":
            ahead = max(diff, 1)   # fire_now 手动触发绕过 due 窗口，diff 可能 ≤0，不报"还有 0/-3 分钟"
            return f"「{p['title']}」还有 {ahead} 分钟到点哦～（{p['start_time']}）"
        if phase == "due":
            return f"「{p['title']}」到点啦，可以开始～（{p['start_time']}）"
        elapsed = max(-diff, 0)
        return f"「{p['title']}」已经过去 {elapsed} 分钟啦，现在开始也不晚～"

    async def fire_now(self, job_id: str) -> dict:
        """手动触发（/api/scheduler/run/{id}）：绕过 due 判定，记账语义不变"""
        now = datetime.now()
        for job in self._collect_jobs(now):
            if job["id"] == job_id:
                return await self._fire(job, now, force=True, occurrence=self._occurrence(job, now))
        return {"ok": False, "error": f"任务不存在: {job_id}"}

    # ─── 用户任务 CRUD（API 端点与 agent 工具的共同入口）──

    def register_user_task(self, mode: str, schedule: dict, payload: dict) -> dict:
        # 注册幂等（防重复入队的注册侧）：Agent loop 偶发重复调工具时，
        # 同一 kind+schedule+payload 的第二次注册命中已有任务，不再新建
        def _key(kind, s, p):
            return (kind, json.dumps(s, sort_keys=True, ensure_ascii=False),
                    json.dumps(p, sort_keys=True, ensure_ascii=False))
        want = _key("remind" if mode == "direct" else "agent", schedule, payload)
        tasks = [t for t in _read_json(TASKS_FILE, []) if isinstance(t, dict)]
        dup = next((t for t in tasks
                    if _key(t.get("kind"), t.get("schedule") or {}, t.get("payload") or {}) == want), None)
        if dup:
            logger.info(f"[scheduler] 注册命中幂等去重，复用已有任务 {dup['id']}")
            return {**dup, "_existing": True}
        task = {
            "id": f"task_{secrets.token_hex(4)}",
            "kind": "remind" if mode == "direct" else "agent",
            "mode": mode,
            "schedule": schedule,
            "payload": payload,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        err = self._validate_task(task, for_register=True)
        if err:
            raise ValueError(err)
        tasks.append(task)
        _write_json(TASKS_FILE, tasks)
        logger.info(f"[scheduler] 注册用户任务 {task['id']}: {mode} {schedule} {payload}")
        return task

    def list_user_tasks(self) -> List[dict]:
        tasks = self._load_user_tasks()
        for t in tasks:
            t["next_fire"] = self._next_fire(t["schedule"])
        return tasks

    def delete_user_task(self, task_id: str) -> Optional[dict]:
        tasks = [t for t in _read_json(TASKS_FILE, []) if isinstance(t, dict)]
        found = next((t for t in tasks if t.get("id") == task_id), None)
        if found:
            _write_json(TASKS_FILE, [t for t in tasks if t.get("id") != task_id])
            logger.info(f"[scheduler] 删除用户任务 {task_id}")
        return found

    def _remove_user_task(self, task_id: str, reason: str):
        self.delete_user_task(task_id)

    def _validate_task(self, t: dict, for_register: bool = False) -> Optional[str]:
        if not isinstance(t, dict):
            return "非对象"
        if not str(t.get("id", "")).startswith("task_"):
            return "id 必须以 task_ 开头"
        kind = t.get("kind")
        if kind not in ("remind", "agent"):
            return f"kind 非法: {kind}"
        if kind == "remind" and not (t.get("payload") or {}).get("text", "").strip():
            return "remind 任务缺 text"
        if kind == "agent" and not (t.get("payload") or {}).get("prompt", "").strip():
            return "agent 任务缺 prompt"
        sched = t.get("schedule") or {}
        stype = sched.get("type")
        if stype == "once":
            at = _parse_dt(sched.get("at", ""))
            if at is None:
                return "once 任务 at 不是合法 ISO 时间"
            if for_register and at < datetime.now() - timedelta(minutes=1):
                return "once 任务不能注册过去的时间"
        elif stype == "daily":
            if not (0 <= int(sched.get("hour", -1)) <= 23 and 0 <= int(sched.get("minute", 0)) <= 59):
                return "daily 任务 hour/minute 越界"
        elif stype == "weekly":
            if not (0 <= int(sched.get("weekday", -1)) <= 6):
                return "weekly 任务 weekday 越界（0=周一 … 6=周日）"
            if not (0 <= int(sched.get("hour", -1)) <= 23):
                return "weekly 任务 hour 越界"
        elif stype == "monthly":
            if not (1 <= int(sched.get("day", 0)) <= 31):
                return "monthly 任务 day 越界（1-31，超出当月天数自动收敛到月末）"
            if not (0 <= int(sched.get("hour", -1)) <= 23):
                return "monthly 任务 hour 越界"
        elif stype == "yearly":
            if not (1 <= int(sched.get("month", 0)) <= 12):
                return "yearly 任务 month 越界（1-12）"
            if not (1 <= int(sched.get("day", 0)) <= 31):
                return "yearly 任务 day 越界"
            if not (0 <= int(sched.get("hour", -1)) <= 23):
                return "yearly 任务 hour 越界"
        else:
            return f"schedule.type 非法: {stype}"
        return None

    def _next_fire(self, schedule: dict) -> Optional[str]:
        now = datetime.now()
        stype = schedule.get("type")
        if stype == "once":
            return schedule.get("at")
        if stype == "daily":
            target = now.replace(hour=schedule["hour"], minute=schedule.get("minute", 0), second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target.isoformat(timespec="minutes")
        if stype == "weekly":
            days_ahead = (schedule["weekday"] - now.weekday()) % 7
            target = (now + timedelta(days=days_ahead)).replace(
                hour=schedule["hour"], minute=schedule.get("minute", 0), second=0, microsecond=0)
            if target <= now:
                target += timedelta(weeks=1)
            return target.isoformat(timespec="minutes")
        if stype == "monthly":
            target = _at_hm(_clamp_day(now.year, now.month, int(schedule["day"])),
                            int(schedule["hour"]), int(schedule.get("minute", 0)))
            if target <= now:
                ny, nm = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
                target = _at_hm(_clamp_day(ny, nm, int(schedule["day"])),
                                int(schedule["hour"]), int(schedule.get("minute", 0)))
            return target.isoformat(timespec="minutes")
        if stype == "yearly":
            target = _at_hm(_clamp_day(now.year, int(schedule["month"]), int(schedule["day"])),
                            int(schedule["hour"]), int(schedule.get("minute", 0)))
            if target <= now:
                target = _at_hm(_clamp_day(now.year + 1, int(schedule["month"]), int(schedule["day"])),
                                int(schedule["hour"]), int(schedule.get("minute", 0)))
            return target.isoformat(timespec="minutes")
        return None

    # ─── 记账（state.markers）─────────────────────────

    def _fired_for(self, job_id: str, occurrence: datetime) -> bool:
        """durable：同一"发生日"（daily=天 / weekly=该次周目标日）是否已成功跑过"""
        state = _read_json(STATE_FILE, {"markers": {}})
        m = (state.get("markers") or {}).get(job_id) or {}
        fired = _parse_dt(m.get("fired", "") or "")
        return fired is not None and fired.date() == occurrence.date()

    def _in_backoff(self, job_id: str, now: datetime) -> bool:
        state = _read_json(STATE_FILE, {"markers": {}})
        m = (state.get("markers") or {}).get(job_id) or {}
        attempt = _parse_dt(m.get("attempt", "") or "")
        return attempt is not None and (now - attempt).total_seconds() < RETRY_SECONDS

    def _set_marker(self, job_id: str, when: datetime):
        self._update_marker(job_id, {"fired": when.isoformat(timespec="seconds")})

    def _set_attempt(self, job_id: str, when: datetime):
        self._update_marker(job_id, {"attempt": when.isoformat(timespec="seconds")})

    def _update_marker(self, job_id: str, patch: dict):
        state = _read_json(STATE_FILE, {"markers": {}})
        markers = state.setdefault("markers", {})
        markers.setdefault(job_id, {}).update(patch)
        # 顺手清扫 8 天前的旧记账（weekly marker 最长需要保留一周）
        cutoff = datetime.now() - timedelta(days=8)
        for jid in list(markers):
            fired = _parse_dt((markers[jid].get("fired") or ""))
            attempt = _parse_dt((markers[jid].get("attempt") or ""))
            if (fired is None or fired < cutoff) and (attempt is None or attempt < cutoff):
                del markers[jid]
        _write_json(STATE_FILE, state)

    # ─── 信箱（outbox）────────────────────────────────

    def _save_overview(self, text: str, now: datetime):
        """每日总览落盘：只在当日有效，第二天触发时覆盖（每日刷新）"""
        _write_json(OVERVIEW_FILE, {"date": now.date().isoformat(), "text": text,
                                    "created_at": now.isoformat(timespec="seconds")})
        logger.info(f"[scheduler] 每日总览已落盘（{now.date()}）")

    def today_overview(self) -> Optional[dict]:
        """首页栏目数据源：只认今天的那份，昨天的不给（过期总览是误导）"""
        data = _read_json(OVERVIEW_FILE, {})
        if isinstance(data, dict) and data.get("date") == date.today().isoformat() \
                and str(data.get("text", "")).strip():
            return data
        return None

    def _save_proactive(self, cards: dict, now: datetime):
        """回忆卡片文案落盘：当日有效（overview.json 同款契约），第二天覆盖。
        落盘的是"今天已叙述"的判定——两张都交白卷（弃说）也落盘，弃说当日
        生效不再退原文；None（无素材未组稿）由调用方拦截，不落盘"""
        _write_json(PROACTIVE_FILE, {"date": now.date().isoformat(),
                                     "capsule": cards.get("capsule"),
                                     "last_year": cards.get("last_year"),
                                     "created_at": now.isoformat(timespec="seconds")})
        logger.info(f"[scheduler] 回忆卡片文案已落盘（{now.date()}）")

    def today_proactive(self) -> Optional[dict]:
        """/api/init 回忆卡片数据源：文件存在且是今天的 = 今天已叙述的判定
        （含整卡弃说，两张文案都可以是 null）；今天还没组稿过返回 None 走原文兜底"""
        data = _read_json(PROACTIVE_FILE, {})
        if isinstance(data, dict) and data.get("date") == date.today().isoformat():
            return data
        return None

    def _outbox_push(self, kind: str, text: str):
        entries = _read_json(OUTBOX_FILE, [])
        entries.append({
            "id": f"outbox_{secrets.token_hex(4)}",
            "kind": kind,
            "text": text,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
        _write_json(OUTBOX_FILE, entries)

    def claim_due(self) -> List[dict]:
        """前端领取（s12 的 pending_delivery + 确认，at-least-once 投递）：
        领取 = 标 claimed_at（不删）；前端展示成功后 ack 才删；
        领取后 RECLAIM_SECONDS 内未确认 → 允许重领（HTTP 响应在网络层丢了，
        用户仍收得到；前端按 id 去重防重弹）；
        超保鲜期 → 作废清账不返回（超时的领走不要通知，用户定稿）。
        读改写之间无 await，单事件循环下天然原子。"""
        now = datetime.now()
        entries = [e for e in _read_json(OUTBOX_FILE, []) if isinstance(e, dict)]
        out: List[dict] = []
        keep: List[dict] = []
        stale = 0
        dirty = False
        for e in entries:
            created = _parse_dt(e.get("created_at", ""))
            ttl = OUTBOX_TTL_SECONDS.get(e.get("kind", ""), DEFAULT_OUTBOX_TTL)
            if not created or (now - created).total_seconds() > ttl:
                stale += 1
                dirty = True
                continue                      # 过期作废：清账不返回
            claimed = _parse_dt(e.get("claimed_at", "") or "")
            if claimed and (now - claimed).total_seconds() < RECLAIM_SECONDS:
                keep.append(e)               # 已领走、等待确认中
                continue
            # 首领/超时重领都要把 claimed_at 落盘——写回条件若漏了这一步，
            # 下一片会当作从未领过而重复投递（每 30s 重弹一次的事故又回来了）
            keep.append({**e, "claimed_at": now.isoformat(timespec="seconds")})
            out.append(e)
            dirty = True
        if dirty:
            _write_json(OUTBOX_FILE, keep)
        if stale:
            logger.info(f"[scheduler] 信箱过期作废 {stale} 条（领取时静默清理）")
        return out

    def ack(self, ids: List[str]) -> int:
        """前端展示成功后确认送达，删除对应条目。幂等；未确认的条目在 RECLAIM 后重领"""
        idset = set(ids or [])
        if not idset:
            return 0
        entries = [e for e in _read_json(OUTBOX_FILE, []) if isinstance(e, dict)]
        rest = [e for e in entries if e.get("id") not in idset]
        n = len(entries) - len(rest)
        if n:
            _write_json(OUTBOX_FILE, rest)
            logger.info(f"[scheduler] 信箱确认送达 {n} 条")
        return n

    def _sweep_outbox(self):
        """夜间清过保条目（用户从不打开页面时防文件膨胀）。按各自 TTL 判断——
        未过保的留着等白天领取（周日晚的周报保鲜 24h，凌晨 2 点不能整箱倒）；
        判定与 claim_due 领取路径同款（无 created_at 视为过保），两处一套政策"""
        now = datetime.now()
        keep: List[dict] = []
        dropped = 0
        for e in _read_json(OUTBOX_FILE, []):
            if not isinstance(e, dict):
                continue
            created = _parse_dt(e.get("created_at", ""))
            ttl = OUTBOX_TTL_SECONDS.get(e.get("kind", ""), DEFAULT_OUTBOX_TTL)
            if not created or (now - created).total_seconds() > ttl:
                dropped += 1
                continue
            keep.append(e)
        if dropped:
            _write_json(OUTBOX_FILE, keep)
            logger.info(f"[scheduler] 夜间清扫信箱：过保清除 {dropped} 条，保留 {len(keep)} 条")
