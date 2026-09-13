"""JSON 文件持久化存储

零依赖，人可读，MVP 够用。每个实体一个 JSON 文件，
内存中全量加载，写入时落盘。
"""
import json
import logging
import os
from filelock import FileLock
from typing import List, Optional

from app.config import DATA_DIR
from app.models.schemas import Event, MoodRecord, Summary, Preference

logger = logging.getLogger(__name__)


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_json(filename: str) -> List[dict]:
    """从 JSON 文件加载数据，文件不存在返回空列表

    损坏防护：解析失败时把坏文件改名隔离（保留现场供人工修复），
    返回空列表让服务继续可用——宁可暂时看不到旧数据，也不能让每个读请求都炸。
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        corrupt_path = path + ".corrupt"
        os.rename(path, corrupt_path)
        logger.error(f"[db] {filename} 损坏（{e}），已隔离为 {corrupt_path}，从空数据继续")
        return []


def _save_json(filename: str, data: List[dict]):
    """写入 JSON 文件，加文件锁防止并发冲突

    原子写：先写 .tmp 再 os.replace。open(path, "w") 会先截断原文件，
    进程写到一半被杀（--reload 重启/崩溃）就留下半截 JSON
    （2026-08-28 events.json 损坏事故的成因）。FileLock 只防并发写，不防这个。
    """
    _ensure_data_dir()
    path = os.path.join(DATA_DIR, filename)
    lock_path = path + ".lock"
    with FileLock(lock_path):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)  # 同目录内替换，原子生效


# ─── Event 存储 ────────────────────────────────────────

def load_events() -> List[Event]:
    """加载所有事件"""
    raw = _load_json("events.json")
    return [Event(**r) for r in raw]


def save_events(events: List[Event]):
    """保存所有事件"""
    _save_json("events.json", [e.model_dump() for e in events])


def add_event(event: Event) -> Event:
    """添加一个事件"""
    events = load_events()
    events.append(event)
    save_events(events)
    return event


def get_events_by_date(date_str: str) -> List[Event]:
    """按日期获取事件，按 start_time 排序"""
    events = load_events()
    filtered = [e for e in events if e.date == date_str]
    filtered.sort(key=lambda e: e.start_time or "99:99")
    return filtered


def get_events_by_date_range(date_from: str, date_to: str) -> List[Event]:
    """按日期范围获取事件"""
    events = load_events()
    return [e for e in events if date_from <= e.date <= date_to]


def update_event(event_id: str, updates: dict) -> Optional[Event]:
    """更新事件的部分字段"""
    events = load_events()
    for i, e in enumerate(events):
        if e.id == event_id:
            for k, v in updates.items():
                setattr(e, k, v)
            events[i] = e
            save_events(events)
            return e
    return None


def delete_event(event_id: str) -> Optional[Event]:
    """删除事件：命中返回被删的事件，不存在返回 None（不静默成功）"""
    events = load_events()
    remaining = [e for e in events if e.id != event_id]
    if len(remaining) == len(events):
        return None
    deleted = next(e for e in events if e.id == event_id)
    save_events(remaining)
    return deleted


# ─── Mood 存储 ────────────────────────────────────────

def load_moods() -> List[MoodRecord]:
    raw = _load_json("moods.json")
    return [MoodRecord(**r) for r in raw]


def save_moods(moods: List[MoodRecord]):
    _save_json("moods.json", [m.model_dump() for m in moods])


def add_mood(mood: MoodRecord) -> MoodRecord:
    moods = load_moods()
    moods.append(mood)
    save_moods(moods)
    return mood


def get_moods_by_date(date_str: str) -> List[MoodRecord]:
    moods = load_moods()
    return [m for m in moods if m.date == date_str]


def get_moods_by_date_range(date_from: str, date_to: str) -> List[MoodRecord]:
    moods = load_moods()
    return [m for m in moods if date_from <= m.date <= date_to]


def delete_mood(mood_id: str) -> Optional[MoodRecord]:
    """删除心情记录：命中返回被删记录，不存在返回 None（不静默成功）"""
    moods = load_moods()
    remaining = [m for m in moods if m.id != mood_id]
    if len(remaining) == len(moods):
        return None
    deleted = next(m for m in moods if m.id == mood_id)
    save_moods(remaining)
    return deleted


# ─── Summary 存储 ──────────────────────────────────────

def load_summaries() -> List[Summary]:
    raw = _load_json("summaries.json")
    return [Summary(**r) for r in raw]


def save_summaries(summaries: List[Summary]):
    _save_json("summaries.json", [s.model_dump() for s in summaries])


def add_summary(summary: Summary) -> Summary:
    summaries = load_summaries()
    summaries.append(summary)
    save_summaries(summaries)
    return summary


def get_summary(period_type: str, period_start: str) -> Optional[Summary]:
    summaries = load_summaries()
    for s in summaries:
        if s.period_type == period_type and s.period_start == period_start:
            return s
    return None


# ─── Preference 存储 ──────────────────────────────────

def load_preferences() -> List[Preference]:
    raw = _load_json("profile.json")
    if not isinstance(raw, dict):
        return []
    return [Preference(**r) for r in raw.get("preferences", [])]


def save_preferences(prefs: List[Preference]):
    _save_json("profile.json", {"preferences": [p.model_dump() for p in prefs]})


def add_preference(pref: Preference) -> Preference:
    prefs = load_preferences()
    prefs.append(pref)
    save_preferences(prefs)
    return pref


def update_preference(pref_id: str, updates: dict) -> Optional[Preference]:
    prefs = load_preferences()
    for i, p in enumerate(prefs):
        if p.id == pref_id:
            for k, v in updates.items():
                setattr(p, k, v)
            prefs[i] = p
            save_preferences(prefs)
            return p
    return None


def delete_preference(pref_id: str) -> bool:
    prefs = load_preferences()
    new_prefs = [p for p in prefs if p.id != pref_id]
    if len(new_prefs) == len(prefs):
        return False
    save_preferences(new_prefs)
    return True


# ─── 展示记录（主动推送的"最近展示日"）─────────────────
# 时光胶囊等推送的展示冷却账本：{记录id: 最近展示日}。
# 独立侧车而非事件字段——"什么时候给用户看过"是展示侧的
# 记账，不是记忆内容本身，别让它进 events.json

SURFACE_LOG_FILE = "surface_log.json"


def load_surface_log() -> dict:
    """加载展示记录，文件不存在/损坏返回空 dict（损坏隔离纪律同 _load_json）"""
    path = os.path.join(DATA_DIR, SURFACE_LOG_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as e:
        corrupt_path = path + ".corrupt"
        os.rename(path, corrupt_path)
        logger.error(f"[db] {SURFACE_LOG_FILE} 损坏（{e}），已隔离为 {corrupt_path}，从空记录继续")
        return {}


def save_surface_log(log: dict):
    """写入展示记录（原子写纪律同 _save_json：tmp + os.replace + FileLock）"""
    _ensure_data_dir()
    path = os.path.join(DATA_DIR, SURFACE_LOG_FILE)
    with FileLock(path + ".lock"):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
