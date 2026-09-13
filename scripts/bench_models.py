"""交互模型对照实验：flash / plus / max × 开思考 / 不开思考

回答两个问题：
1. 更聪明的模型会用更少的思考 token 得到答案吗？
2. 更聪明的模型不开思考也能做对吗？

矩阵：3 模型 × 2 思考模式 × 3 场景 × 3 轮 = 54 次请求。
控制变量：parser 固定 qwen3.7-flash + 不思考（base.py 硬编码），全矩阵一致；
         温度 0.3、思考预算 2048（生产配置）、每次请求独立临时库、独立会话。

场景与评分：
  A 规划（hard）：明天有晨跑 07:10-07:40，用户要"帮我规划个合理的时间"→
     必须先 get_overview 查日程；增强文本带具体时段；排进 07:10-07:40 = 失败；
     落库全为 plan；回复里的时间与库一致（汇报忠实）
  B 用户定时（discipline）："明天 6:30-6:45 刷牙…" 明确时间只要记录 →
     不得调用 get_overview；落库时间与用户给的完全一致
  C 闲聊路由（baseline）：天气感慨 → 不调任何工具，直接回复

产出：docs/model-benchmark-data.json（逐次原始数据）+ 控制台摘要。
用法：cd myNoPlan && .venv/bin/python scripts/bench_models.py
"""
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

import app.store.db as db  # noqa: E402
import app.agent.interactive as interactive_module  # noqa: E402
from app.agent.interactive import InteractiveAgent  # noqa: E402

# 控制台安静跑，全量 INFO 进文件留档
logging.basicConfig(level=logging.WARNING, format="%(message)s")
file_handler = logging.FileHandler(ROOT / "log" / "bench-models.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logging.getLogger().addHandler(file_handler)

MODELS = ["qwen3.7-flash", "qwen3.7-plus", "qwen3.7-max"]
MODES = {"think": {"enable_thinking": True, "thinking_budget": 2048},
         "nothink": {"enable_thinking": False}}
REPEATS = 3
RUN_TIMEOUT_S = 240

TOMORROW = (date.today() + timedelta(days=1)).isoformat()
SEED_EVENT = {"id": "seed1", "title": "晨跑", "type": "plan", "date": TOMORROW,
              "start_time": "07:10", "end_time": "07:40", "source_text": "晨跑",
              "created_at": datetime.now().isoformat(timespec="seconds")}

SCENARIOS = {
    "A_plan": "我明天早上要刷牙、洗脸、洗衣服、做早饭，帮我规划个合理的时间记录下来",
    "B_user_times": "明天 6:30-6:45 刷牙洗脸，6:45-7:00 洗衣服，7:00-7:10 做早饭，帮我记录一下",
    "C_chat": "今天天气真好呀，出门走走很舒服",
}
B_EXPECTED = {("06:30", "06:45"), ("06:45", "07:00"), ("07:00", "07:10")}
SEED_SPAN = ("07:10", "07:40")


# ── 计时客户端包装：只代理 interactive 用到的 chat.completions.create ──
class _TimedCompletions:
    def __init__(self, inner_create, sink):
        self._inner = inner_create
        self._sink = sink

    async def create(self, **kw):
        t0 = time.monotonic()
        resp = err = None
        try:
            resp = await self._inner(**kw)
            return resp
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            rec = {"ms": round((time.monotonic() - t0) * 1000)}
            u = getattr(resp, "usage", None) if resp is not None else None
            if u:
                rec["prompt_tokens"] = u.prompt_tokens
                rec["completion_tokens"] = u.completion_tokens
                d = getattr(u, "completion_tokens_details", None)
                rec["reasoning_tokens"] = getattr(d, "reasoning_tokens", None)
            if err:
                rec["error"] = err
            self._sink.append(rec)


class _TimedChat:
    def __init__(self, client, sink):
        self.completions = _TimedCompletions(client.chat.completions.create, sink)


class TimedClient:
    def __init__(self, client, sink):
        self.chat = _TimedChat(client, sink)


# ── 评分 ──
def _to_min(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _norm_times(text):
    return {f"{int(m.group(1)):02d}:{m.group(2)}"
            for m in re.finditer(r"(\d{1,2}):(\d{2})", text or "")}


def _arg_text(call):
    """tool_calls_log 的 args 可能是 dict 也可能是 JSON 字符串，统一取 text 字段"""
    a = call.get("args")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return ""
    return (a or {}).get("text", "") if isinstance(a, dict) else ""


def score_run(scenario, result, new_events, reply):
    checks = {}
    tools = [c.get("tool") for c in result.get("tool_calls_log", [])]
    parse_texts = [_arg_text(c) for c in result.get("tool_calls_log", [])
                   if c.get("tool") == "parse_and_record"]

    if scenario == "A_plan":
        idx_overview = tools.index("get_overview") if "get_overview" in tools else 99
        idx_parse = tools.index("parse_and_record") if "parse_and_record" in tools else 99
        checks["先查日程再记录"] = idx_overview < idx_parse and idx_parse < 99
        checks["增强文本带时段"] = all(len(_norm_times(t)) >= 4 for t in parse_texts) if parse_texts else False
        overlap = []
        for e in new_events:
            if e.get("start_time") and e.get("end_time"):
                s, t = _to_min(e["start_time"]), _to_min(e["end_time"])
                if s < _to_min(SEED_SPAN[1]) and _to_min(SEED_SPAN[0]) < t:
                    overlap.append(f"{e['title']}({e['start_time']}-{e['end_time']})")
        checks["不与晨跑重叠"] = not overlap
        checks["全部记为计划"] = bool(new_events) and all(e.get("type") == "plan" for e in new_events)
        db_times = set()
        for e in new_events:
            db_times |= {e.get("start_time"), e.get("end_time")} - {None}
        checks["汇报时间与库一致"] = db_times != set() and db_times <= _norm_times(reply)
    elif scenario == "B_user_times":
        checks["未调用日程查询"] = "get_overview" not in tools
        got = {(e.get("start_time"), e.get("end_time")) for e in new_events}
        checks["时间与用户给的一致"] = got == B_EXPECTED
    else:  # C_chat
        checks["不调用任何工具"] = len(tools) == 0
        checks["有正常回复"] = bool(reply and reply.strip())
    return checks


async def run_once(scenario, message, model, mode):
    tmp = tempfile.mkdtemp(prefix=f"bench-{scenario}-{mode}-")
    run = {"scenario": scenario, "model": model, "mode": mode, "error": None}
    try:
        db.DATA_DIR = tmp
        with open(os.path.join(tmp, "events.json"), "w", encoding="utf-8") as f:
            json.dump([SEED_EVENT], f, ensure_ascii=False)

        interactive_module._INTERACTIVE_EXTRA = dict(MODES[mode])
        agent = InteractiveAgent()
        agent.model = model
        calls = []
        agent.client = TimedClient(agent.client, calls)

        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(agent.handle(message), timeout=RUN_TIMEOUT_S)
        finally:
            run["total_ms"] = round((time.monotonic() - t0) * 1000)
            run["llm_calls"] = calls
        reply = (result or {}).get("reply", "")
        run["reply"] = reply
        run["tools"] = [c.get("tool") for c in (result or {}).get("tool_calls_log", [])]
        with open(os.path.join(tmp, "events.json"), encoding="utf-8") as f:
            new_events = [e for e in json.load(f) if e.get("id") != "seed1"]
        run["new_events"] = [{"type": e.get("type"), "start": e.get("start_time"),
                              "end": e.get("end_time"), "title": e.get("title")} for e in new_events]
        run["checks"] = score_run(scenario, result, new_events, reply)
        run["pass"] = all(run["checks"].values())
    except Exception as e:
        run["error"] = f"{type(e).__name__}: {e}"
        run["pass"] = False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return run


async def main():
    runs = []
    t_all = time.monotonic()
    for scenario, message in SCENARIOS.items():
        for model in MODELS:
            for mode in MODES:
                for i in range(1, REPEATS + 1):
                    r = await run_once(scenario, message, model, mode)
                    r["run"] = i
                    runs.append(r)
                    think_tok = sum(c.get("reasoning_tokens") or 0 for c in r.get("llm_calls", []))
                    tag = "PASS" if r.get("pass") else "FAIL"
                    err = f" err={r['error'][:60]}" if r.get("error") else ""
                    bad = [k for k, v in r.get("checks", {}).items() if not v]
                    print(f"[{len(runs):02d}/54] {scenario[:9]:9s} {model:14s} {mode:7s} #{i} "
                          f"{tag} total={r.get('total_ms', 0):>6}ms think_tok={think_tok:>5}"
                          f"{err}{' 失败项:' + ','.join(bad) if bad else ''}", flush=True)
                    await asyncio.sleep(0.3)

    out = {"meta": {
               "date": datetime.now().isoformat(timespec="seconds"),
               "matrix": f"{len(MODELS)} models x {len(MODES)} modes x {len(SCENARIOS)} scenarios x {REPEATS} runs",
               "thinking_budget_when_on": 2048,
               "parser": "qwen3.7-flash, enable_thinking=False（base.py 硬编码，全矩阵一致）",
               "temperature": 0.3,
               "seed_event": SEED_EVENT,
               "scenarios": SCENARIOS,
               "run_timeout_s": RUN_TIMEOUT_S,
           },
           "runs": runs}
    out_path = ROOT / "docs" / "model-benchmark-data.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n完成：{len(runs)} 次，总耗时 {(time.monotonic() - t_all) / 60:.1f} 分钟")
    print(f"原始数据 → {out_path.relative_to(ROOT)}")
    print(f"过程日志 → log/bench-models.log")


if __name__ == "__main__":
    asyncio.run(main())
