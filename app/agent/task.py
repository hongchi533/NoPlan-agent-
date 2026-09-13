"""Task 对象与 StepRecord：执行追踪与可观测性

Task    — 一次工具调用的完整生命周期
StepRecord — agent loop 每步的结构化记录

所有记录都是结构化 dict，可 JSON 序列化，供日志和前端消费。
"""
import time
import uuid
from typing import Any, Dict, List, Optional


class TaskStatus:
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class Task:
    """一次工具调用的完整生命周期

    Attributes:
        id: 唯一标识
        tool_name: 工具名
        tool_kind: 基础工具 / 任务工具(sub-agent)
        args: 调用参数
        status: 当前状态
        result: 执行结果
        error: 错误信息
        retry_count: 已重试次数
        started_at: 开始时间戳
        finished_at: 结束时间戳
        duration_ms: 耗时毫秒
        undo_fn: 回滚函数（用于可回滚操作）
    """

    def __init__(self, tool_name: str, tool_kind: str, args: Dict[str, Any]):
        self.id = str(uuid.uuid4())[:8]
        self.tool_name = tool_name
        self.tool_kind = tool_kind
        self.args = args
        self.status = TaskStatus.PENDING
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.retry_count = 0
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.duration_ms: Optional[int] = None
        self.undo_fn: Optional[callable] = None  # 回滚函数

    def start(self):
        """标记开始执行"""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

    def succeed(self, result: str):
        """标记成功"""
        self.result = result
        self.status = TaskStatus.SUCCESS
        self.finished_at = time.time()
        self.duration_ms = int((self.finished_at - self.started_at) * 1000)

    def fail(self, error: str):
        """标记失败"""
        self.error = error
        self.status = TaskStatus.FAILED
        self.finished_at = time.time()
        self.duration_ms = int((self.finished_at - self.started_at) * 1000)

    def mark_retry(self):
        """标记正在重试"""
        self.retry_count += 1
        self.status = TaskStatus.RETRYING

    def mark_rolled_back(self):
        """标记已回滚"""
        self.status = TaskStatus.ROLLED_BACK

    def to_dict(self) -> Dict[str, Any]:
        """结构化输出，可 JSON 序列化"""
        return {
            "id": self.id,
            "tool": self.tool_name,
            "kind": self.tool_kind,
            "args": self.args,
            "status": self.status,
            "result": (self.result[:200] if self.result else None),
            "error": self.error,
            "retry_count": self.retry_count,
            "duration_ms": self.duration_ms,
        }


class StepRecord:
    """Agent loop 每步的结构化记录

    每次 LLM 调用为一轮（step），记录：
    - 调用了哪些 tool（tasks 列表）
    - LLM 是否产生了 tool_call
    - 本轮耗时
    """

    def __init__(self, step_num: int):
        self.step_num = step_num
        self.tasks: List[Task] = []
        self.has_tool_calls = False
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.duration_ms: Optional[int] = None

    def finish(self):
        self.finished_at = time.time()
        self.duration_ms = int((self.finished_at - self.started_at) * 1000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_num,
            "has_tool_calls": self.has_tool_calls,
            "tasks": [t.to_dict() for t in self.tasks],
            "duration_ms": self.duration_ms,
        }
