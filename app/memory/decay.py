"""记忆衰减模型

幂律衰减（Ebbinghaus 遗忘曲线的长期行为）：
  strength(t) = initial_strength / (1 + days / half_life)

半衰期语义：经过 half_life 天，强度减半。不同类型半衰期不同
（config.DECAY_HALF_LIVES）。相比旧指数衰减（decay_rate^days），
幂律先快后慢、长时程仍有残值——旧模型下任何记录活到一年
strength ≤ 0.16，"去年今天"（门槛 0.4）永远无法触发。
每天凌晨批量重算一次，查询时直接用字段值。
"""
from datetime import datetime, date
from typing import List, Optional, Tuple

from app.config import DECAY_HALF_LIVES


def compute_strength(
    created_at: str,
    event_type: str,
    initial_strength: float = 1.0,
    rehearsals: Optional[List[float]] = None,
    now: Optional[date] = None,
) -> float:
    """计算当前记忆强度（幂律半衰期）

    strength = initial / (1 + days / half_life)

    Args:
        created_at: ISO format datetime 字符串
        event_type: "done" / "plan" / "mood" / "moment"
        initial_strength: 初始强度
        rehearsals: 历次强化值（翻阅+0.3，搜索命中+0.2，被动展示+0.1）
        now: 当前日期，默认今天

    Returns:
        记忆强度 [0, 1]
    """
    if now is None:
        now = date.today()

    created_date = datetime.fromisoformat(created_at).date()
    days = max((now - created_date).days, 0)

    half_life = DECAY_HALF_LIVES.get(event_type, 60)
    decayed = initial_strength / (1 + days / half_life)

    # 累加强化（翻阅、搜索、被动展示等）
    boost = sum(rehearsals) if rehearsals else 0.0

    return min(decayed * (1 + boost), 1.0)


def should_surface_for_last_year(strength: float) -> bool:
    """是否值得在"去年今天"中展示

    门槛 0.4 与幂律半衰期匹配：moment 一年后 strength=0.5 可以露面，
    done 一年后 0.14 不打扰。旧门槛 0.7 在旧指数模型下数学上不可达
    （衰减最慢的 moment 一年也只有 0.16）。
    """
    return strength >= 0.4


def should_surface_for_capsule(strength: float) -> bool:
    """是否适合被时光胶囊唤起（衰减区间，唤起惊喜感最强）"""
    return 0.3 <= strength <= 0.5


def should_archive(strength: float) -> bool:
    """是否应该归档"""
    return strength < 0.1


def recompute_all_strengths(events: list, moods: list) -> Tuple[list, list]:
    """批量重算所有记忆强度（夜间整理时调用）

    Returns:
        (updated_events, updated_moods)
    """
    now = date.today()

    for e in events:
        e_type = "moment" if e.is_moment else e.type
        e.strength = compute_strength(e.created_at, e_type, now=now)

    for m in moods:
        m.strength = compute_strength(m.created_at, "mood", now=now)

    return events, moods
