"""存量记录回填向量（开启 EMBEDDING_ENABLED 前后各跑一次均可）

用途：
  1. 开关开启后第一次回填存量（新记录会由写路径自动补算）
  2. 更换 EMBEDDING_MODEL 后全量重算（不同模型的向量不可互比）
  3. 检索日志出现"侧车模型 ≠ 配置"warning 时的对齐修复

用法（任意目录均可，脚本自己定位项目根）：
    /path/to/myNoPlan/.venv/bin/python /path/to/myNoPlan/scripts/backfill_embeddings.py
"""
import asyncio
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)  # config 经 load_dotenv 读 .env，依赖 cwd

from app.store import db                      # noqa: E402
from app.memory import embeddings             # noqa: E402


async def main():
    items = db.load_events() + db.load_moods()
    print(f"待回填 {len(items)} 条记录，模型 {embeddings.EMBEDDING_MODEL} …")
    n = await embeddings.backfill(items, progress=print)
    print(f"完成：{n} 条向量已写入 {embeddings._store_path()}")
    if n < len(items):
        print(f"（跳过 {len(items) - n} 条：无可嵌入文本，如空 content 的心情）")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
