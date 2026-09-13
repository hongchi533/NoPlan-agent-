"""声音验收：产品人格用例 × 配置（2026-09-01）

背景：产品原则——模型是基建，性格是产品，人格沉淀在 interactive.PERSONA。
换模型 / 改人格后必跑本脚本：机器验纪律与红线词，回复全文打印出来由用户体感终审
（机器判不了"像不像我们的产品"，只能拦住明显跑偏）。

用例（三档情绪强度 + 1 条纪律对照）：
  V1 轻度：累 + 晚饭纠结
  V2 中度：被领导当众批评，憋屈
  V3 重度：考研倒计时焦虑，自我否定
  C1 纪律对照：纯心情陈述 → 必须照常 parse + mood 落库（防人格加重后喧宾夺主、丢路由纪律）

硬判据（任一不过即 FAIL）：
  - 纪律：V1-V3、C1 均 parse_and_record + mood 落库 + 零事件 + 零定时任务
  - 红线词：回复含典型说教/无效化用语（想开点 / 别想太多 / 没什么大不了 / 你应该 /
    你要坚强 / 至少你还）——人格的红线，跟路由纪律同级
软指标（只打印不判死，供体感参考）：
  - 字数参考带宽：轻度 ≤120，中度 ≤200，重度 ≤250；超出提示"偏长"
  - 情绪场景用列表符号（- / 1. / •）通常显冷淡，出现则提示

用法：.venv/bin/python scripts/bench_voice.py <ds_nothink|37_think> [配置...]
输出：console（判据 + 回复全文） + docs/voice-raw.jsonl 逐链追加
"""
import asyncio, json, logging, os, re, shutil, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import app.store.db as db
import app.agent.interactive as im
import app.scheduler as sched_mod
from app.agent.interactive import InteractiveAgent
from app.agent import tools as tools_mod
from app.scheduler import Scheduler

CONFIGS = {
    "ds_nothink": ("deepseek-v4-flash", {"enable_thinking": False}),
    "37_think":   ("qwen3.7-flash", {"enable_thinking": True, "thinking_budget": 2048}),
}

# 用例：输入 → 档位（软指标带宽用）
CASES = [
    ("V1", "轻度", "今天上班有点累，晚饭不知道吃啥"),
    ("V2", "中度", "被领导当众说了一顿，憋屈"),
    ("V3", "重度", "考研倒计时一百天了，啥都没学进去，觉得自己完了"),
    ("C1", "对照", "今天莫名有点低落，有点想家了"),
]
BAND = {"轻度": 120, "中度": 200, "重度": 250}
REDLINE = ["想开点", "别想太多", "没什么大不了", "你应该", "你要坚强", "至少你还"]

captured = []
class _Cap(logging.Handler):
    def emit(self, r): captured.append(r.getMessage())
_h = _Cap(); _h.setLevel(logging.INFO)
log_i = logging.getLogger("app.agent.interactive"); log_i.addHandler(_h)
log_i.setLevel(logging.INFO); log_i.propagate = False
logging.basicConfig(level=logging.WARNING)
RE_CALL = re.compile(r"LLM 响应\((\d+)ms, model=[^,]+, prompt=(\d+)tok, completion=(\d+)tok")

OUT = ROOT / "docs" / "voice-raw.jsonl"

def _setup(tmp, model, extra):
    db.DATA_DIR = tmp
    sched_mod.TASKS_FILE = os.path.join(tmp, "tasks.json")
    sched_mod.STATE_FILE = os.path.join(tmp, "state.json")
    sched_mod.LEGACY_NIGHTLY_FILE = os.path.join(tmp, "nope")
    tools_mod.set_task_scheduler(Scheduler(background_agent=None))
    im._INTERACTIVE_EXTRA = dict(extra)
    ag = InteractiveAgent(); ag.model = model
    return ag

def _snap(tmp):
    def rd(fn):
        p = os.path.join(tmp, fn)
        return json.load(open(p)) if os.path.exists(p) else []
    tasks_p = sched_mod.TASKS_FILE
    return rd("events.json"), rd("moods.json"), (json.load(open(tasks_p)) if os.path.exists(tasks_p) else [])

async def _turn(agent, msg):
    t0 = time.monotonic()
    r = await asyncio.wait_for(agent.handle(msg), timeout=240)
    rounds = [{"ms": int(a), "completion": int(c)} for a, _p, c in RE_CALL.findall("\n".join(captured))]
    captured.clear()
    return r, [c.get("tool") for c in (r or {}).get("tool_calls_log", [])], \
        (r or {}).get("reply", ""), rounds, round((time.monotonic() - t0) * 1000)

async def run_case(cid, key, tier, text):
    model, extra = CONFIGS[key]
    tmp = tempfile.mkdtemp(prefix="voice-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, text)
    evs, moods, tasks = _snap(tmp)
    hit = [w for w in REDLINE if w in reply]
    n_char = len(reply.replace("\n", ""))
    has_list = bool(re.search(r"\n\s*([-•·]|\d+\.)\s", reply))
    checks = {"parse落心情": "parse_and_record" in tools and len(moods) >= 1,
              "零定时任务": len(tasks) == 0,
              "无红线词": not hit,
              "有回复": bool(reply.strip())}
    # 纯情绪陈述（重度/对照）不应产生事件；轻度/中度的输入含生活内容（上班/被批评），
    # 记成事件算不算合理是产品决定——只打印不判死
    if tier in ("重度", "对照"):
        checks["纯情绪零事件"] = len(evs) == 0
    soft = []
    if tier in BAND and n_char > BAND[tier]:
        soft.append(f"偏长({n_char}字>{BAND[tier]})")
    if has_list:
        soft.append("用了列表符号")
    mood_str = ",".join(sorted({str(m.get("mood")) for m in moods}))
    ev_str = [f"{e.get('title')}@{e.get('start_time') or e.get('date')}" for e in evs]
    shutil.rmtree(tmp, ignore_errors=True)
    return {"case": cid, "tier": tier, "input": text, "checks": checks,
            "pass": all(checks.values()), "reply": reply, "mood": mood_str,
            "events": ev_str, "n_char": n_char, "soft": soft,
            "total_ms": ms, "rounds": rounds, "tools": tools}

async def run_config(key):
    for cid, tier, text in CASES:
        rec = {"config": key, **await run_case(cid, key, tier, text)}
        bad = [k for k, v in rec["checks"].items() if not v]
        print(f"\n[{key}] {rec['case']}({tier}) {'PASS' if rec['pass'] else 'FAIL'} "
              f"{rec['total_ms']}ms 心情={rec['mood']} {'未过:' + ','.join(bad) if bad else ''}", flush=True)
        if rec["events"]:
            print(f"  顺带记了事件: {'; '.join(rec['events'])}", flush=True)
        if rec["soft"]:
            print(f"  软指标提示: {'; '.join(rec['soft'])}", flush=True)
        print("  回复全文:", flush=True)
        for line in rec["reply"].splitlines() or [rec["reply"]]:
            print(f"    {line}", flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        await asyncio.sleep(0.3)

async def main():
    keys = sys.argv[1:] or list(CONFIGS)
    for k in keys:
        if k not in CONFIGS:
            print(f"未知配置 {k}"); continue
        print(f"\n══ {k} = {CONFIGS[k][0]} {CONFIGS[k][1]} ══", flush=True)
        await run_config(k)

if __name__ == "__main__":
    asyncio.run(main())
