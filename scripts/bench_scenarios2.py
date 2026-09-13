"""第二轮场景基准：10 个经典场景 × 2 配置（2026-09-01）

背景：终选轮（bench_final_selection.py）ds 全绿但均值优势靠 R3 长尾撑着。
本轮换一批用户侧典型场景重跑，看结论是否稳态。仅对比两个配置：
  ds_nothink  deepseek-v4-flash 不思考   ← 候选
  37_think    qwen3.7-flash 思考2048     ← 现产

用例（★=易错类）：
  S1 查天气（疑问句+路由：要调天气工具，但不许落库）
  S2 明天九点开会★（陈述→parse_and_record，禁 register_task，双提醒坑）
  S3 十分钟后休息一下★（相对时间→register_task once，at≈now+10min）
  S4 天气好就记十点跑步★（条件陈述→仍应落 plan，禁 register_task）
  S5 心情描述（纯感受→只落 mood，零事件）
  S6 多步规划并罗列（hard：先查日程再排时段，不与种子重叠）
  S7 周期提醒★（每天十点半吃药→register_task daily hour=22 minute=30）
  S8 陈述后追问★（疑问句≠记录，当天修的 bug）
  S9 明确时间纪律★（时间全给了→直接记，禁多余 get_overview）
  S10 查询汇报（问安排→get_overview，零落库，答到点子上）

控制变量：parser 固定 qwen3.7-flash 不思考；每链独立临时库；温度生产值。
用法：.venv/bin/python scripts/bench_scenarios2.py <ds_nothink|37_think>
输出：console 摘要 + docs/scenarios2-raw.jsonl 逐链追加
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
    "37_think":   ("qwen3.7-flash", {"enable_thinking": True, "thinking_budget": 2048}),
}
TOMORROW = (date.today() + timedelta(days=1)).isoformat()
SEED = {"id": "seed1", "title": "晨跑", "type": "plan", "date": TOMORROW,
        "start_time": "07:10", "end_time": "07:40", "source_text": "晨跑",
        "created_at": datetime.now().isoformat(timespec="seconds")}
SEED_SPAN = ("07:10", "07:40")
S9_EXPECTED = {("06:30", "06:45"), ("06:45", "07:00"), ("07:00", "07:10")}
S8_ST = "今天好累啊，想摆烂一天"
S8_Q = "可是摆烂不会拖慢进度吗？不会让原本记得的更加忘记吗"

captured = []
class _Cap(logging.Handler):
    def emit(self, r): captured.append(r.getMessage())
_h = _Cap(); _h.setLevel(logging.INFO)
log_i = logging.getLogger("app.agent.interactive"); log_i.addHandler(_h)
log_i.setLevel(logging.INFO)
log_i.propagate = False
logging.basicConfig(level=logging.WARNING)
RE_CALL = re.compile(r"LLM 响应\((\d+)ms, model=[^,]+, prompt=(\d+)tok, completion=(\d+)tok(?:, 首chunk=(\d+)ms)?(?:, 首字=(\d+)ms)?")

OUT = ROOT / "docs" / "scenarios2-raw.jsonl"

def _to_min(t): h, m = t.split(":"); return int(h) * 60 + int(m)
def _norm_times(text): return {f"{int(m.group(1)):02d}:{m.group(2)}" for m in re.finditer(r"(\d{1,2}):(\d{2})", text or "")}

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
    return rd("events.json"), rd("moods.json"), (json.load(open(sched_mod.TASKS_FILE)) if os.path.exists(sched_mod.TASKS_FILE) else [])

def _seeded(tmp):
    json.dump([SEED], open(os.path.join(tmp, "events.json"), "w", encoding="utf-8"), ensure_ascii=False)

async def _turn(agent, msg):
    t0 = time.monotonic()
    r = await asyncio.wait_for(agent.handle(msg), timeout=240)
    rounds = [{"ms": int(a), "completion": int(c), "first_chunk_ms": int(fc) if fc else None,
               "first_text_ms": int(ft) if ft else None}
              for a, _pt, c, fc, ft in RE_CALL.findall("\n".join(captured))]
    captured.clear()
    return r, [c.get("tool") for c in (r or {}).get("tool_calls_log", [])], \
        (r or {}).get("reply", ""), rounds, round((time.monotonic() - t0) * 1000)

async def case_S1(model, extra):  # 查天气：疑问句，要调工具不许落库
    tmp = tempfile.mkdtemp(prefix="s2w-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, "明天下午会下雨吗？")
    evs, moods, _ = _snap(tmp)
    checks = {"调了天气工具": any("weather" in x for x in tools),
              "零落库": len(evs) == 0 and len(moods) == 0,
              "答了天气": bool(reply.strip())}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, []

async def case_S2(model, extra):  # 明天九点开会：陈述→parse，禁 register
    tmp = tempfile.mkdtemp(prefix="s2m-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, "明天九点开会")
    evs, _, tasks = _snap(tmp)
    checks = {"parse且无register": "parse_and_record" in tools and "register_task" not in tools,
              "没落任务": len(tasks) == 0,
              "时间09:00": any(e.get("start_time") == "09:00" for e in evs)}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, []

async def case_S3(model, extra):  # 十分钟后休息：相对时间→once at≈now+10
    tmp = tempfile.mkdtemp(prefix="s2r-"); ag = _setup(tmp, model, extra)
    t_before = datetime.now()
    r, tools, reply, rounds, ms = await _turn(ag, "提醒我十分钟后休息一下")
    tasks = _snap(tmp)[2]
    at = tasks[0]["schedule"].get("at", "") if tasks else ""
    dt = (datetime.fromisoformat(at) - t_before).total_seconds() / 60 if at else -99
    checks = {"register恰一次": tools == ["register_task"], "at在8-12分": 8 <= dt <= 12}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, [f"{at} (+{dt:.1f}min)"]

async def case_S4(model, extra):  # 条件陈述：天气好就记十点跑步→仍落 plan，禁 register
    tmp = tempfile.mkdtemp(prefix="s2c-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, "明天天气好的话帮我记一下十点跑步")
    evs, _, tasks = _snap(tmp)
    run = [e for e in evs if "跑" in str(e.get("title", ""))]
    checks = {"没落定时任务": "register_task" not in tools and len(tasks) == 0,
              "记了跑步plan": bool(run) and all(e.get("type") == "plan" for e in run),
              "时间10:00": any(e.get("start_time") == "10:00" for e in run)}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, [f"{e.get('title')}@{e.get('start_time')}" for e in evs]

async def case_S5(model, extra):  # 纯心情描述
    tmp = tempfile.mkdtemp(prefix="s2d-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, "今天有点小开心，又有点emo，说不清是什么感觉")
    evs, moods, _ = _snap(tmp)
    checks = {"调了parse": "parse_and_record" in tools,
              "心情落库": len(moods) >= 1,
              "零事件": len(evs) == 0}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, [f"mood={m.get('mood')}" for m in moods]

async def case_S6(model, extra):  # 多步规划并罗列（hard）
    tmp = tempfile.mkdtemp(prefix="s2p-"); ag = _setup(tmp, model, extra)
    _seeded(tmp)
    r, tools, reply, rounds, ms = await _turn(ag, "我明天早上要刷牙、洗脸、洗衣服、做早饭，帮我规划个合理的时间记录下来")
    evs = [e for e in _snap(tmp)[0] if e.get("id") != "seed1"]
    i_ov = tools.index("get_overview") if "get_overview" in tools else 99
    i_pa = tools.index("parse_and_record") if "parse_and_record" in tools else 99
    overlap = [e["title"] for e in evs if e.get("start_time") and e.get("end_time")
               and _to_min(e["start_time"]) < _to_min(SEED_SPAN[1]) and _to_min(SEED_SPAN[0]) < _to_min(e["end_time"])]
    dbt = set()
    for e in evs: dbt |= {e.get("start_time"), e.get("end_time")} - {None}
    checks = {"先查日程再记录": i_ov < i_pa and i_pa < 99,
              "不与晨跑重叠": not overlap,
              "全记为plan": bool(evs) and all(e.get("type") == "plan" for e in evs),
              "罗列汇报": dbt != set() and dbt <= _norm_times(reply)}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, [f"{e.get('title')}@{e.get('start_time')}-{e.get('end_time')}" for e in evs]

async def case_S7(model, extra):  # 周期提醒：每天十点半吃药
    tmp = tempfile.mkdtemp(prefix="s2y-"); ag = _setup(tmp, model, extra)
    r, tools, reply, rounds, ms = await _turn(ag, "每天晚上十点半提醒我吃药")
    evs, _, tasks = _snap(tmp)
    sc = tasks[0]["schedule"] if tasks else {}
    checks = {"register且无parse": tools == ["register_task"],
              "daily@22:30": sc.get("type") == "daily" and sc.get("hour") == 22 and sc.get("minute") == 30,
              "没落事件": len(evs) == 0}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, [f"schedule={sc}"]

async def case_S8(model, extra):  # 陈述后追问：疑问句≠记录
    tmp = tempfile.mkdtemp(prefix="s2q-"); ag = _setup(tmp, model, extra)
    r1, t1, rep1, rd1, ms1 = await _turn(ag, S8_ST)
    ev0, md0, _ = _snap(tmp)
    r2, t2, rep2, rd2, ms2 = await _turn(ag, S8_Q)
    ev1, md1, _ = _snap(tmp)
    checks = {"陈述轮有记录": "parse_and_record" in t1 and (len(ev1) + len(md1)) > 0,
              "追问轮零工具": t2 == [],
              "追问轮零新增": len(ev1) == len(ev0) and len(md1) == len(md0)}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms1 + ms2, rd1 + rd2, t1 + t2, []

async def case_S9(model, extra):  # 明确时间纪律：直接记，禁多余查询
    tmp = tempfile.mkdtemp(prefix="s2b-"); ag = _setup(tmp, model, extra)
    _seeded(tmp)
    r, tools, reply, rounds, ms = await _turn(ag, "明天 6:30-6:45 刷牙洗脸，6:45-7:00 洗衣服，7:00-7:10 做早饭，帮我记录一下")
    evs = [e for e in _snap(tmp)[0] if e.get("id") != "seed1"]
    got = {(e.get("start_time"), e.get("end_time")) for e in evs}
    checks = {"未调日程查询": "get_overview" not in tools, "时间与用户一致": got == S9_EXPECTED}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, []

async def case_S10(model, extra):  # 查询汇报：零落库
    tmp = tempfile.mkdtemp(prefix="s2e-"); ag = _setup(tmp, model, extra)
    _seeded(tmp)
    r, tools, reply, rounds, ms = await _turn(ag, "我明天早上有什么安排呀")
    evs, _, tasks = _snap(tmp)
    checks = {"查了日程": "get_overview" in tools,
              "零新增记录": "parse_and_record" not in tools and "register_task" not in tools and len(evs) == 1 and len(tasks) == 0,
              "答到点子上": "晨跑" in reply}
    shutil.rmtree(tmp, ignore_errors=True)
    return checks, ms, rounds, tools, []

CASES = {"S1": case_S1, "S2": case_S2, "S3": case_S3, "S4": case_S4, "S5": case_S5,
         "S6": case_S6, "S7": case_S7, "S8": case_S8, "S9": case_S9, "S10": case_S10}

async def run_config(key):
    model, extra = CONFIGS[key]
    for name, fn in CASES.items():
        rec = {"config": key, "model": model, "case": name}
        try:
            checks, ms, rounds, tools, extra_out = await fn(model, extra)
            rec.update({"checks": checks, "pass": all(checks.values()),
                        "total_ms": ms, "rounds": rounds, "tools": tools, "detail": extra_out})
            bad = [k for k, v in checks.items() if not v]
            print(f"[{key:10s}] {name:4s} {'PASS' if rec['pass'] else 'FAIL'} {ms:>6}ms 轮数={len(rounds)}"
                  f"{'  未过:' + ','.join(bad) if bad else ''}", flush=True)
        except Exception as e:
            rec.update({"pass": False, "error": f"{type(e).__name__}: {e}"})
            print(f"[{key:10s}] {name:4s} ERROR {str(e)[:80]}", flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        await asyncio.sleep(0.3)

async def main():
    # 与生产 main.py lifespan 同构：先挂载 amap MCP，否则 S1 天气用例无效
    # （2026-09-01 勘误：首跑没挂载，S1 两配置的结果均作废，详见 docs/model-selection）
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
