"""数据模型：事件、心情、解析结果、摘要"""
from datetime import date, datetime, time
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Event(BaseModel):
    """事件/活动"""
    id: str
    title: str
    type: Literal["plan", "done"]
    date: str  # YYYY-MM-DD
    start_time: Optional[str] = None  # HH:MM
    end_time: Optional[str] = None  # HH:MM
    note: Optional[str] = None
    images: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source_text: str
    strength: float = 1.0  # 记忆强度 [0, 1]
    is_moment: bool = False
    created_at: str  # ISO format datetime


class MoodRecord(BaseModel):
    """心情日记"""
    id: str
    date: str  # YYYY-MM-DD
    time: Optional[str] = None  # HH:MM，具体时刻
    mood: int = Field(ge=1, le=5)  # 😫😐🙂😊🥰
    content: Optional[str] = None
    images: list[str] = Field(default_factory=list)
    strength: float = 1.0
    created_at: str  # ISO format datetime


class Preference(BaseModel):
    """用户偏好（auto=后台发现, manual=用户声明）"""
    id: str
    pattern: str  # 匹配关键词，如"跑步"
    rules: dict = Field(default_factory=dict)  # {"default_time": "06:00"} 或 {"fact": "吃饭时喜欢配咖啡"}（写时压缩短句；source_text 仅存证）
    source: Literal["auto", "manual"] = "auto"
    source_text: Optional[str] = None  # manual 时的原文
    evidence: dict = Field(default_factory=lambda: {"recent_mismatches": 0})
    created_at: str
    updated_at: str


class ParseResult(BaseModel):
    """LLM 解析结果（单条）"""
    type: Literal["plan", "done", "mood", "preference"]
    title: Optional[str] = None
    date: Optional[str] = None  # YYYY-MM-DD
    start_time: Optional[str] = None  # HH:MM
    end_time: Optional[str] = None  # HH:MM
    note: Optional[str] = None
    mood: Optional[int] = None  # 1-5, 仅 type=mood 时
    tags: list[str] = Field(default_factory=list)
    is_moment: bool = False


class ParseResults(BaseModel):
    """LLM 解析结果（可能多条：事件+心情同时存在）"""
    items: list[ParseResult]


class Summary(BaseModel):
    """周期摘要"""
    id: str
    period_type: Literal["day", "week", "month"]
    period_start: str  # YYYY-MM-DD
    period_end: str  # YYYY-MM-DD
    content: str  # LLM 生成的摘要
    narrative: Optional[str] = None  # 叙事式总结
    stats: dict = Field(default_factory=dict)
    created_at: str


# 标签配色已迁至前端 ui/app.js 的 TAG_COLORS（纯视图关注点，后端不感知）。

# 心情 emoji 映射（后端回复/检索文本与前端展示共用，经 /api/init 下发给前端）
MOOD_EMOJI = {1: "😫", 2: "😐", 3: "🙂", 4: "😊", 5: "🥰"}
