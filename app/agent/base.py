"""BaseAgent：封装 LLM 调用的基类

通过 OpenAI 兼容接口调用 Qwen 模型。
子类只需定义 system_prompt 和 output schema。
"""
import asyncio
import json
import logging
import time
from typing import Any, Optional

from openai import AsyncOpenAI

from app.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_MODEL

logger = logging.getLogger(__name__)


class BaseAgent:
    """Agent 基类，封装 LLM 调用能力"""

    name: str = "base"
    system_prompt: str = ""
    # 子类可覆写：指定该 agent 使用的模型（None = 用默认模型）
    model_override = None

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            timeout=30.0,     # SDK 默认 600s 太长：断网时用户会盯 10 分钟转圈
            max_retries=0,    # 关 SDK 内部静默重试，用下面带退避+日志的显式重试
        )
        self.model = self.model_override or DASHSCOPE_MODEL

    async def call_llm(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_retries: int = 2,
    ) -> str:
        """调用 LLM，返回文本响应

        Args:
            user_message: 用户输入
            system_prompt: 覆盖默认 system prompt
            json_mode: 是否要求返回 JSON
            temperature: 生成温度
            max_retries: 解析失败重试次数

        Returns:
            LLM 响应文本
        """
        sys_prompt = system_prompt or self.system_prompt
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_message},
        ]

        # 记录 prompt 日志（system 截断 + user 全量）
        logger.info(
            f"[{self.name}] LLM 调用 | system({len(sys_prompt)}字): {sys_prompt[:2000]}... | "
            f"user: {user_message}"
        )

        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            # sub-agent 固定关闭思考：结构化/摘要类任务不需要思维链，只要快和省
            extra_body={"enable_thinking": False},
        )
        # Qwen 支持 response_format JSON
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries + 1):
            try:
                t0 = time.monotonic()
                response = await self.client.chat.completions.create(**kwargs)
                elapsed_ms = (time.monotonic() - t0) * 1000
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM 返回空内容")
                usage = response.usage
                tok = (f", prompt={usage.prompt_tokens}tok, completion={usage.completion_tokens}tok"
                       if usage else "")
                logger.info(f"[{self.name}] LLM 响应({elapsed_ms:.0f}ms, model={self.model}{tok}, {len(content)}字): {content[:200]}")
                return content.strip()
            except Exception as e:
                logger.warning(f"[{self.name}] LLM 调用失败 (attempt {attempt+1}): {type(e).__name__}: {e}")
                if attempt == max_retries:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))  # 退避：0.5s, 1s，避免打爆已抖动的服务

    async def call_llm_json(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict:
        """调用 LLM 并解析为 JSON，自动重试

        Returns:
            解析后的 dict
        """
        raw = await self.call_llm(
            user_message=user_message,
            system_prompt=system_prompt,
            json_mode=True,
            temperature=temperature,
        )
        # 尝试解析 JSON（LLM 可能返回 ```json ... ``` 包裹的内容）
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取 ```json ... ``` 中的内容
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            raise
