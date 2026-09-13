"""向量召回：DashScope embedding 侧车存储 + 暴力余弦

开关 EMBEDDING_ENABLED（默认关，见 config.py）：关 = n-gram 主路（现状）；
开 = 向量主路，n-gram 降为旁路——补没来得及算向量的记录 + 向量路任何故障
（API 失败/侧车过期/为空）时整体接管。所有对外函数绝不抛异常，失败只 log。

关键约束：
- 向量必须同一模型产出（不同模型的向量不可互比）——侧车记录 model 名，
  与配置不一致即视为过期，向量路降级直到重跑回填
- 侧车是派生数据：任何时刻可从 events+moods 全量重算（backfill），
  缺向量的记录不参与向量召回，由 n-gram 旁路兜住，无数据丢失
- 存储 = 单位向量（写入时归一化 + 保留 6 位小数），余弦退化为点积；
  千级 × 1024 维纯 Python 全量扫毫秒级，不需要向量数据库
- 检索路径的读取按文件 mtime 缓存；ingest 每次全量重写侧车（千级 JSON
  约 1s，写路径可接受；数据上万后换 jsonl 追加式再议）
"""
import json
import logging
import math
import os
import time
from typing import Iterable, List, Optional, Tuple

from filelock import FileLock
from openai import AsyncOpenAI

from app.config import (
    DATA_DIR, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    EMBEDDING_ENABLED, EMBEDDING_MODEL,
)
from app.models.schemas import Event, MoodRecord

logger = logging.getLogger(__name__)

EMBEDDINGS_FILE = "embeddings.json"
_BATCH = 10  # DashScope embedding 单请求上限 10 条（实测 25 → 400 InvalidParameter）

_client: Optional[AsyncOpenAI] = None  # 懒建：开关关闭时永不建客户端


def enabled() -> bool:
    return EMBEDDING_ENABLED


def item_text(item) -> str:
    """记录 → 喂给 embedding 的文本：纯内容，日期/时刻不进（对语义模型是噪声，
    时间归时间桶管——向量只负责内容相似）"""
    if isinstance(item, MoodRecord):
        return (item.content or "").strip()
    parts = [item.title, item.note or "", item.source_text or ""]
    return " ".join(p.strip() for p in parts if p and p.strip())


async def embed(texts: List[str]) -> List[List[float]]:
    """文本批 → 单位向量批（内部按 _BATCH 分批请求）。失败抛异常，由调用方降级"""
    client = _get_client()
    out: List[List[float]] = []
    for i in range(0, len(texts), _BATCH):
        resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts[i:i + _BATCH])
        out.extend(d.embedding for d in resp.data)
    return [_unit(v) for v in out]


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            timeout=30.0,
            max_retries=1,
        )
    return _client


def _unit(v: List[float]) -> List[float]:
    """归一化 + 6 位小数（压文件体积；微小的非单位性不影响余弦排序）"""
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [round(x / n, 6) for x in v]


# ─── 侧车文件读写（tmp + os.replace 原子替换，同 db.py 的写纪律）───

def _store_path() -> str:
    return os.path.join(DATA_DIR, EMBEDDINGS_FILE)

_cache = {"mtime": None, "payload": None}


def _load_payload() -> dict:
    """读侧车（mtime 缓存）。损坏不隔离不致命：返回空，向量路降级，n-gram 旁路兜住"""
    path = _store_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {"model": EMBEDDING_MODEL, "dim": 0, "vectors": {}}
    if _cache["mtime"] == mtime:
        return _cache["payload"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        payload = {"model": raw.get("model", ""), "dim": raw.get("dim", 0),
                   "vectors": raw.get("vectors", {})}
    except (json.JSONDecodeError, OSError, AttributeError) as e:
        logger.warning(f"[embeddings] 侧车读取失败（{e}），向量路本轮降级")
        payload = {"model": EMBEDDING_MODEL, "dim": 0, "vectors": {}}
    _cache["mtime"] = mtime
    _cache["payload"] = payload
    return payload


def _save_payload(payload: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _store_path()
    with FileLock(path + ".lock"):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)  # 紧凑格式：机器数据不需要缩进
        os.replace(tmp_path, path)
    _cache["mtime"] = os.path.getmtime(path)
    _cache["payload"] = payload


# ─── 对外入口 ────────────────────────────────────────

async def ingest(items: Iterable) -> None:
    """写路径挂钩：记录落库后补算向量。绝不抛异常——记录本身已安全落库，
    这里失败只 log，缺的向量由 n-gram 旁路兜住，下次回填对齐"""
    if not EMBEDDING_ENABLED:
        return
    try:
        pairs = [(it.id, item_text(it)) for it in items]
        pairs = [(i, t) for i, t in pairs if t]
        if not pairs:
            return
        vecs = await embed([t for _, t in pairs])
        payload = _load_payload()
        if payload.get("model") != EMBEDDING_MODEL:
            payload = {"model": EMBEDDING_MODEL, "dim": 0, "vectors": {}}
        payload["dim"] = len(vecs[0]) if vecs else payload.get("dim", 0)
        for (id_, _), v in zip(pairs, vecs):
            payload["vectors"][id_] = v
        _save_payload(payload)
        logger.info(f"[embeddings] 已补算 {len(pairs)} 条向量（库存 {len(payload['vectors'])}）")
    except Exception as e:
        logger.warning(f"[embeddings] 向量补算失败（n-gram 旁路兜底）: {type(e).__name__}: {e}")


async def query_top(query: str, valid_ids: set, top_k: int) -> Tuple[List[Tuple[str, float]], str]:
    """查询向量 → 与库存逐一算余弦 → top_k。

    valid_ids 同时起两个作用：淘汰已删记录的陈旧向量 + 限定打分范围
    （intersect 模式传窗口 id 集，就是在窗口内做精确的向量排序）。
    返回 ([(id, score)...], status)，status: ok / disabled / stale / empty / failed；
    任何非 ok 调用方都走 n-gram 旁路。绝不抛异常。
    """
    if not EMBEDDING_ENABLED:
        return [], "disabled"
    payload = _load_payload()
    vectors = payload.get("vectors", {})
    if payload.get("model") != EMBEDDING_MODEL and vectors:
        logger.warning(f"[embeddings] 侧车模型 {payload.get('model')!r} ≠ 配置 {EMBEDDING_MODEL!r}，"
                       f"向量路过期——重跑 scripts/backfill_embeddings.py 后恢复")
        return [], "stale"
    live = {k: v for k, v in vectors.items() if k in valid_ids}
    if not live:
        return [], "empty"
    try:
        qv = (await embed([query]))[0]
    except Exception as e:
        logger.warning(f"[embeddings] 查询向量计算失败，本轮走 n-gram 旁路: {type(e).__name__}: {e}")
        return [], "failed"
    logger.info(f"[embeddings] 余弦扫描开始：{len(live)} 条 × {len(qv)} 维")
    t0 = time.monotonic()
    scored = [(id_, sum(a * b for a, b in zip(qv, v))) for id_, v in live.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    elapsed = (time.monotonic() - t0) * 1000
    top1 = f"top1={scored[0][1]:.4f}({scored[0][0]})" if scored else "无候选"
    logger.info(f"[embeddings] 余弦扫描完成：{elapsed:.1f}ms，{top1}")
    return scored[:top_k], "ok"


async def backfill(items: List, progress=None) -> int:
    """全量重建侧车（scripts/backfill_embeddings.py 与换模型后重算用）。
    显式脚本即意图，不看开关。返回成功写入的向量数"""
    pairs = [(it.id, item_text(it)) for it in items]
    pairs = [(i, t) for i, t in pairs if t]
    payload = {"model": EMBEDDING_MODEL, "dim": 0, "vectors": {}}
    dim = 0
    done = 0
    for i in range(0, len(pairs), _BATCH):
        chunk = pairs[i:i + _BATCH]
        vecs = await embed([t for _, t in chunk])
        dim = len(vecs[0]) if vecs else dim
        for (id_, _), v in zip(chunk, vecs):
            payload["vectors"][id_] = v
        done += len(chunk)
        if progress:
            progress(f"回填 {done}/{len(pairs)}")
    payload["dim"] = dim
    _save_payload(payload)
    return len(pairs)
