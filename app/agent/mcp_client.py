"""MCP 客户端：高德地图（Streamable HTTP 远程模式，mcp.amap.com 官方托管）

设计要点：
- 生命周期：AsyncExitStack 持有高层 Client（进入即完成 initialize 握手），
  app startup 连接 / shutdown 关闭，长连接复用
- 工具发现：tools/list 动态发现（服务端 15 个工具），白名单过滤只挂载天气
- 懒加载 SDK：mcp 包在 connect() 内部 import——SDK 缺失（如 Python<3.10 环境）
  只影响天气工具挂载，主服务照常启动（降级哲学）
- 韧性：call 断线自动重连一次再试；任何异常转友好文案，绝不向 agent 抛
- 脱敏：URL 携带 key，所有进日志的异常信息统一过 _scrub()（httpx 报错会带完整 URL）

新版 mcp SDK 注意（2026-08 实测）：Tool 的 schema 属性是 input_schema（下划线），
结果对象是 is_error / content[].text——与旧版 camelCase 不同。
"""
import asyncio
import logging
import time
from contextlib import AsyncExitStack
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class McpHttpClient:
    """Streamable HTTP MCP 客户端：连接 + 白名单工具发现 + 调用转发"""

    def __init__(self, name: str, url: str, whitelist: List[str], call_timeout: float = 10.0,
                 service_label: str = "外部服务"):
        self.name = name
        self.url = url
        self.whitelist = whitelist
        self.call_timeout = call_timeout
        # 用户可见的服务中文名（降级文案用）——name 是技术标识符（"amap"），不进用户文案
        self.service_label = service_label
        # URL 中 key= 后面的部分，日志脱敏用
        self._secret = url.split("key=")[-1].split("&")[0] if "key=" in url else ""
        self._stack: Optional[AsyncExitStack] = None
        self._client = None
        self._lock = asyncio.Lock()  # 防止并发调用同时触发重连

    def _scrub(self, msg: str) -> str:
        """把异常信息里的 API key 替换成 ***（httpx 报错会内嵌完整 URL）"""
        return msg.replace(self._secret, "***") if self._secret else msg

    async def connect(self) -> List[Any]:
        """建立连接 + 握手 + 工具发现（白名单过滤）。

        Returns: 白名单命中的 Tool 对象列表
        Raises: 任何连接失败（SDK 缺失/网络/Key 无效），由调用方决定降级
        """
        async with self._lock:
            try:
                await self.close()
                # 懒加载：SDK 未安装时只影响本模块，不拖垮整个应用
                from mcp.client import Client

                stack = AsyncExitStack()
                client = await stack.enter_async_context(
                    Client(self.url, read_timeout_seconds=self.call_timeout)
                )
                self._stack = stack
                self._client = client

                # 工具发现：服务端有什么以 tools/list 为准（不硬编码）
                result = await client.list_tools()
                all_tools = result.tools
                names = [t.name for t in all_tools]
                logger.info(f"[mcp:{self.name}] 连接成功，服务端提供 {len(all_tools)} 个工具: {names}")

                kept = [t for t in all_tools if t.name in self.whitelist]
                dropped = [n for n in names if n not in self.whitelist]
                logger.info(
                    f"[mcp:{self.name}] 白名单过滤 {self.whitelist}: "
                    f"保留 {len(kept)} 个，忽略 {len(dropped)} 个: {dropped}"
                )
                return kept
            except Exception as e:
                await self.close()
                raise RuntimeError(self._scrub(f"{type(e).__name__}: {e}"))

    async def close(self):
        """关闭连接（幂等，可安全重复调用）"""
        if self._stack:
            try:
                await self._stack.aclose()
            except Exception as e:
                logger.warning(f"[mcp:{self.name}] 关闭连接时出错（忽略）: {type(e).__name__}")
            self._stack = None
            self._client = None

    def make_caller(self, tool_name: str) -> Callable:
        """生成注册进 agent 工具表的 async wrapper（闭包捕获工具名）"""
        async def _call(**kwargs):
            return await self.call(tool_name, kwargs)
        _call.__name__ = tool_name
        return _call

    async def call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """转发 tools/call；失败重连一次再试；任何异常都转友好文案。"""
        for attempt in (1, 2):
            try:
                if self._client is None:
                    await self.connect()  # 冷启动 / 崩溃后按需重连
                t0 = time.monotonic()
                result = await asyncio.wait_for(
                    self._client.call_tool(tool_name, arguments), timeout=self.call_timeout
                )
                elapsed_ms = (time.monotonic() - t0) * 1000
                text = self._extract_text(result)
                if getattr(result, "is_error", False):
                    logger.warning(
                        f"[mcp:{self.name}] {tool_name} 返回错误({elapsed_ms:.0f}ms): {text[:200]}"
                    )
                    return f"{self.service_label}返回错误: {text[:200]}"
                # 日志格式对齐内置工具：压成一行、截断 300 字
                text_flat = " ".join(text.split())
                logger.info(f"[mcp:{self.name}] {tool_name} 执行完成({elapsed_ms:.0f}ms): {text_flat[:300]}")
                return text
            except Exception as e:
                logger.warning(
                    f"[mcp:{self.name}] {tool_name} 调用失败(第{attempt}次): "
                    f"{self._scrub(f'{type(e).__name__}: {e}')}"
                )
                await self.close()  # 丢弃可能失效的连接，下一轮循环重连
        return f"{self.service_label}暂时不可用，请稍后再试～"

    @staticmethod
    def _extract_text(result) -> str:
        """从 CallToolResult 的 content 块里拼出文本"""
        parts = [c.text for c in getattr(result, "content", []) if hasattr(c, "text")]
        return "\n".join(p for p in parts if p) or str(result)[:500]
