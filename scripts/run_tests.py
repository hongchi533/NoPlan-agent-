"""一键跑全量测试用例（人为观察版）

用例沉淀在 docs/test-cases.md，按 ## 栏目 分组；本脚本解析该文档逐条执行：
  时间提取 → 规则兜底 _extract_time_range
  参数归一 → _resolve_range（@ 起始~结束）
  关键词清洗 → _extract_keywords
  分派模式 → _coarse_retrieve（真实数据只读，@ 参数可选）
  其余栏目（手动/自检）只列出不执行
只读、无 LLM 调用、无写入。相对时间用例以运行当天为基准（输出第一行有日期锚点）。

用法（绝对路径，任意目录可跑）：
    /path/to/myNoPlan/.venv/bin/python /path/to/myNoPlan/scripts/run_tests.py
    末尾可加栏目关键字只跑子集，如： run_tests.py 时间提取
"""
import asyncio
import datetime
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)  # config 经 load_dotenv 读 .env，依赖 cwd

DOC_PATH = os.path.join(_ROOT, "docs", "test-cases.md")


def parse_doc(path: str):
    """解析用例文档：## 栏目标题 + 其后的 `- 用例行`（其余行忽略，可作注释）"""
    sections = []  # [(title, [case, ...])]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("## "):
                sections.append((line[3:].strip(), []))
            elif line.startswith("- ") and sections:
                sections[-1][1].append(line[2:].strip())
    return sections


def split_params(case: str):
    """`query @ 起始~结束` → (query, date_from, date_to)；缺端留空"""
    if " @ " in case:
        q, rng = case.split(" @ ", 1)
        f, _, t = rng.partition("~")
        return q.strip(), f.strip(), t.strip()
    return case, "", ""


async def main():
    # 应用日志压到 WARNING 且并入 stdout（默认 stderr 会在管道/重定向时抢跑到
    # print 前面）：INFO 噪声不刷屏，参数归一的坏格式/双轨 warning 仍按序露出
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="⚠ %(message)s")
    from app.memory.retrieval import MemoryRetrieval

    r = MemoryRetrieval()
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"今天是 {datetime.date.today().isoformat()}（相对时间用例以这天为基准）")
    print(f"用例源: docs/test-cases.md\n")

    total = shown = 0
    for title, cases in parse_doc(DOC_PATH):
        if only and only not in title:
            continue
        print("━" * 10, title, "━" * 10)
        shown += 1
        for case in cases:
            total += 1
            q, f, t = split_params(case)
            if "时间提取" in title:
                print(f"  {q} → {r._extract_time_range(q)}")
            elif "参数归一" in title:
                print(f"  {q} @{f or '-'}~{t or '-'} → {r._resolve_range(q, f, t)}")
            elif "关键词清洗" in title:
                print(f"  {q} → {r._extract_keywords(q)}")
            elif "分派模式" in title:
                picked, mode = await r._coarse_retrieve(q, f, t)
                preview = "；".join(
                    (getattr(it, "title", None) or it.content or "")[:12]
                    for it in picked[:3])
                print(f"  {q} → {mode}，选中{len(picked)}：{preview or '（空）'}")
            else:
                print(f"  · {case}")
        print()

    if not shown:
        print(f"没匹配到栏目：{only}")
        return
    print(f"共 {total} 条用例（不匹配的栏目按脚本内规则执行，其余只列出）")


if __name__ == "__main__":
    asyncio.run(main())
