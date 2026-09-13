"""交互模型终选对照：易错用例全集 × 5 配置（2026-09-01）

背景：生产用 qwen3.7-flash+思考2048，慢（hard 17.8s）。探针初筛发现
deepseek-v4-flash 不开思考全绿且 8.2s——本脚本把"选型证据"钉成可复查的数据。

配置（model, extras）：
  ds_nothink   deepseek-v4-flash  不思考            ← 候选
  37_nothink   qwen3.7-flash      不思考            ← 对照
  37_think     qwen3.7-flash      思考2048          ← 现产
  38_nothink   qwen3.8-flash      不思考            ← 已知掉疑问句纪律
  38_think     qwen3.8-flash      思考2048          ← 回归从未跑过

用例十个（典型场景 × 历史翻车点）：A 规划hard / B 明确时间纪律 / C 闲聊路由 /
D 心情陈述只落心情 / E 查询汇报不落库 /
R1 陈述应记录+追问零记录（当天修的 bug，两轮对话两个判据）/
R2a 提醒走 register_task / R2b 日程陈述不走 register_task（双提醒坑）/
R3 天气路由（512/1024 在此翻车）/ R4 相对时间换算

控制变量：parser 固定 qwen3.7-flash 不思考（base.py 硬编码）；每链独立临时库；
温度生产值。deepseek 候选对 A、R1 各加跑一次复确认（方差钉子）。

用法：.venv/bin/python scripts/bench_final_selection.py <配置key> [配置key...]
输出：console 摘要 + docs/final-selection-raw.jsonl 逐链追加
"""
import asyncio, json, logging, os, re, shutil, sys, tempfile, time
from datetime import date, datetime, timedelta
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
from app.config import AMAP_MCP_URL, MCP_AMAP_TOOL_WHITELIST
from app.agent.mcp_client import McpHttpClient

CONFIGS = {
    "ds_nothink": ("deepseek-v4-flash", {"enable_thinking": False}),
    "37_nothink": ("qwen3.7-flash", {"enable_thinking": False}),
    "37_think":   ("qwen3.7-flash", {"enable_thinking": True, "thinking_budget": 2048}),
    "38_nothink": ("qwen3.8-flash", {"enable_thinking": False}),
    "38_think":   ("qwen3.8-flash", {"enable_thinking": True, "thinking_budget": 2048}),
}
TOMORROW = (date.today() + timedelta(days=1)).isoformat()
SEED = {"id": "seed1", "title": "晨跑", "type": "plan", "date": TOMORROW,
        "start_time": "07:10", "end_time": "07:40", "source_text": "晨跑",
        "created_at": datetime.now().isoformat(timespec="seconds")}
SEED_SPAN = ("07:10", "07:40")
B_EXPECTED = {("06:30", "06:45"), ("06:45", "07:00"), ("07:00", "07:10")}
R1_ST = "今天好累啊，想摆烂一天"
R1_Q = "可是摆烂不会拖慢进度吗？不会让原本记得的更加忘记吗"

captured = []
class _Cap(logging.Handler):
    def emit(self, r): captured.append(r.getMessage())
_h = _Cap(); _h.setLevel(logging.INFO)
log_i = logging.getLogger("app.agent.interactive"); log_i.addHandler(_h)
log_i.setLevel(logging.INFO)
log_i.propagate = False   # 根 logger 已有 handler，INFO 会重复打到 console；本脚本自建 handler 已全量捕获
logging.basicConfig(level=logging.WARNING)
RE_CALL = re.compile(r"LLM 响应\((\d+)ms, model=[^,]+, prompt=(\d+)tok, completion=(\d+)tok(?:, 首chunk=(\d+)ms)?(?:, 首字=(\d+)ms)?")

OUT = ROOT / "docs" / "final-selection-raw.jsonl"

def _to_min(t): h, m = t.split(":"); return int(h) * 60 + int(m)
def _norm_times(text): return {f"{int(m.group(1)):02d}:{m.group(2)}" for m in re.finditer(r"(\d{1,2}):(\d{2})", text or "")}
def _arg_text(c):
    a = c.get("args")
    if isinstance(a, str):
        try: a = json.loads(a)
        except ValueError: return ""
    return (a or {}).get("text", "") if isinstance(a, dict) else ""

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
    rounds = [{"ms": int(a), "completion": int(c), "first_chunk_ms": int(fc) if fc else None,
               "first_text_ms": int(ft) if ft else None}
              for a, _pt, c, fc, ft in RE_CALL.findall("\n".join(captured))]
    captured.clear()
    return r, [c.get("tool") for c in (r or {}).get("tool_calls_log", [])], \
        (r or {}).get("reply", ""), rounds, round((time.monotonic() - t0) * 1000)

async def case_A(model, extra):
    tmp = tempfile.mkdtemp(prefix="fsA-"); ag = _setup(tmp, model, extra)
    json.dump([SEED], open(os.path.join(tmp, "events.json"), "w"))
    r, tools, reply, rounds, ms = await _turn(ag, "我明天早上要刷牙、洗脸、洗衣服、做早饭，帮我规划个合理的时间记录下来")
    evs = [e for e in _snap(tmp)[0] if e.get("id") != "seed1"]
    i_ov = tools.index("get_overview") if "get_overview" in tools else 99
    i_pa = tools.index("parse_and_record") if "parse_and_record" in tools else 99
    parse_texts = [_arg_text(c) for c in (r or {}).get("tool_calls_log", []) if c.get("tool") == "parse_and_record"]
    overlap = [e["title"] for e in evs if e.get("start_time") and e.get("end_time")
               and _to_min(e["start_time"]) < _to_min(SEED_SPAN[1]) and _to_min(SEED_SPAN[0]) < _to_min(e["end_time"])]
    dbt = set()
    for e in evs: dbt |= {e.get("start_time"), e.get("end_time")} - {None}
    checks = {"先查日程再记录": i_ov < i_pa and i_pa < 99,
              "增强带时段": bool(parse_texts) and all(len(_norm_times(t)) >= 4 for t in parse_texts),
              "不与晨跑重叠": not overlap,
              "全记为plan": bool(evs) and all(e.get("type") == "plan" for e in evs),
              "汇报与库一致": dbt != set() and dbt <= _norm_times(reply)}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, [f"{e.get('title')}@{e.get('start_time')}-{e.get('end_time')}" for e in evs]

async def case_B(model, extra):
    tmp = tempfile.mkdtemp(prefix="fsB-"); ag = _setup(tmp, model, extra)
    json.dump([SEED], open(os.path.join(tmp, "events.json"), "w"))
    r, tools, reply, rounds, ms = await _turn(ag, "明天 6:30-6:45 刷牙洗脸，6:45-7:00 洗衣服，7:00-7:10 做早饭，帮我记录一下")
    evs = [e for e in _snap(tmp)[0] if e.get("id") != "seed1"]
    got = {(e.get("start_time"), e.get("end_time")) for e in evs}
    checks = {"未调日程查询": "get_overview" not in tools, "时间与用户一致": got == B_EXPECTED}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, []

async def case_C(model, extra):
    tmp = tempfile.mkdtemp(prefix="fsC-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, "今天天气真好呀，出门走走很舒服")
    checks = {"零工具": tools == [], "有回复": bool(reply.strip())}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, []

async def case_D(model, extra):
    tmp = tempfile.mkdtemp(prefix="fsD-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, "今天莫名有点低落，有点想家了")
    evs, moods, _ = _snap(tmp)
    checks = {"调了parse": "parse_and_record" in tools,
              "心情落库": len(moods) >= 1,
              "无误日程": len(evs) == 0}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, [f"mood={m.get('mood')}" for m in moods]

async def case_E(model, extra):
    tmp = tempfile.mkdtemp(prefix="fsE-"); ag = _setup(tmp, model, extra)
    json.dump([SEED], open(os.path.join(tmp, "events.json"), "w"))
    r, tools, reply, rounds, ms = await _turn(ag, "我明天早上有什么安排呀")
    evs, _, tasks = _snap(tmp)
    checks = {"查了日程": "get_overview" in tools,
              "零新增记录": "parse_and_record" not in tools and "register_task" not in tools and len(evs) == 1 and len(tasks) == 0,
              "答到点子上": "晨跑" in reply}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, []

async def case_R1(model, extra):
    tmp = tempfile.mkdtemp(prefix="fsR1-"); ag = _setup(tmp, model, extra)
    r1, t1, rep1, rd1, ms1 = await _turn(ag, R1_ST)
    ev0, md0, _ = _snap(tmp)
    r2, t2, rep2, rd2, ms2 = await _turn(ag, R1_Q)
    ev1, md1, _ = _snap(tmp)
    checks = {"陈述轮有记录": "parse_and_record" in t1 and (len(ev1) + len(md1)) > 0,
              "追问轮零工具": t2 == [],
              "追问轮零新增": len(ev1) == len(ev0) and len(md1) == len(md0)}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms1 + ms2, rd1 + rd2, t1 + t2, []

async def case_R2a(model, extra):
    tmp = tempfile.mkdtemp(prefix="fsR2a-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, "明天早上九点提醒我带酒")
    tasks = _snap(tmp)[2]
    at = tasks[0]["schedule"].get("at", "") if tasks else ""
    checks = {"register_task且无parse": tools == ["register_task"],
              "at=明天09:00": at.startswith(f"{TOMORROW}T09:00")}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, [at]

async def case_R2b(model, extra):
    tmp = tempfile.mkdtemp(prefix="fsR2b-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, "明天九点开会")
    evs, _, tasks = _snap(tmp)
    checks = {"parse且无register": "parse_and_record" in tools and "register_task" not in tools,
              "没落任务": len(tasks) == 0, "时间09:00": any(e.get("start_time") == "09:00" for e in evs)}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, []

async def case_R3(model, extra):
    tmp = tempfile.mkdtemp(prefix="fsR3-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, "明天天气怎么样")
    evs, _, _ = _snap(tmp)
    checks = {"调了天气工具": any("weather" in x for x in tools), "无误记录": len(evs) == 0}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, []

async def case_R4(model, extra):
    tmp = tempfile.mkdtemp(prefix="fsR4-"); ag = _setup(tmp, model, extra)
    t_before = datetime.now()
    r, tools, reply, rounds, ms = await _turn(ag, "十分钟后提醒我喝水")
    tasks = _snap(tmp)[2]
    at = tasks[0]["schedule"].get("at", "") if tasks else ""
    dt = (datetime.fromisoformat(at) - t_before).total_seconds() / 60 if at else -99
    checks = {"register恰一次": tools == ["register_task"], "at在8-12分": 8 <= dt <= 12}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, [f"{at} (+{dt:.1f}min)"]

CASES = {"A": case_A, "B": case_B, "C": case_C, "D": case_D, "E": case_E, "R1": case_R1,
         "R2a": case_R2a, "R2b": case_R2b, "R3": case_R3, "R4": case_R4}

async def run_config(key):
    model, extra = CONFIGS[key]
    # deepseek 候选：A、R1 复确认各加一次（钉方差）
    plan = [(n, CASES[n], 1) for n in CASES]
    if key == "ds_nothink":
        plan += [("A", CASES["A"], 2), ("R1", CASES["R1"], 2)]
    for name, fn, rep in plan:
        rec = {"config": key, "model": model, "case": name if rep == 1 else f"{name}#2"}
        try:
            checks, ms, rounds, tools, extra_out = await fn(model, extra)
            rec.update({"checks": checks, "pass": all(checks.values()),
                        "total_ms": ms, "rounds": rounds, "tools": tools, "detail": extra_out})
            bad = [k for k, v in checks.items() if not v]
            print(f"[{key:10s}] {rec['case']:5s} {'PASS' if rec['pass'] else 'FAIL'} {ms:>6}ms 轮数={len(rounds)}"
                  f"{'  未过:' + ','.join(bad) if bad else ''}", flush=True)
        except Exception as e:
            rec.update({"pass": False, "error": f"{type(e).__name__}: {e}"})
            print(f"[{key:10s}] {rec['case']:5s} ERROR {str(e)[:80]}", flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        await asyncio.sleep(0.3)

async def main():
    # 与生产 main.py lifespan 同构：先挂载 amap MCP，否则天气用例无效
    # （2026-09-01 勘误：首跑没挂载，R3 全部作废——qwen 诚实说没工具，ds 幻觉调未知工具名）
    client = McpHttpClient("amap", AMAP_MCP_URL, MCP_AMAP_TOOL_WHITELIST)
    for t in await client.connect():
        tools_mod.register_dynamic_tool(name=t.name, description=t.description or "",
                                        parameters=t.input_schema, func=client.make_caller(t.name))
    try:
        keys = sys.argv[1:] or list(CONFIGS)
        for k in keys:
            if k not in CONFIGS:
                print(f"未知配置 {k}"); continue
            print(f"\n══ {k} = {CONFIGS[k][0]} {CONFIGS[k][1]} ══", flush=True)
            await run_config(k)
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
