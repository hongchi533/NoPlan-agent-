"""Orchestrator Agent：Agent Loop 实现

核心循环：
  用户消息 → LLM(带 tools) → 有 tool_call?
    ├─ 是 → 执行工具 → 结果回填 messages → 继续循环
    └─ 否 → 返回 LLM 最终文本回复

终止条件：LLM 不再产生 tool_call，直接给出文本回复。
"""
import json
import logging
from typing import Any, Dict, List

from openai import AsyncOpenAI

from app.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_MODEL
from app.agent.tools import TOOL_DEFINITIONS, TOOL_FUNCTION_MAP, ASYNC_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个记忆助手，帮助用户记录生活、管理日程、回忆过去。

你的原则：
- 不制造焦虑，只增加觉察
- 回答温暖简洁，像朋友聊天
- 记录完事件后，简要确认即可
- 搜索到结果后，直接组织成自然的语言回答
- 可以连续调用多个工具（比如记录完事件后顺便查一下今日总览）

今天是 {today}。
"""


class OrchestratorAgent:
    """Agent Loop：LLM 自主决策调用工具，循环直到无 tool call"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
        self.model = DASHSCOPE_MODEL
        self.max_iterations = 5  # 防止无限循环

    async def handle(self, message: str) -> Dict[str, Any]:
        """Agent Loop 主入口

        Returns:
            {
                "reply": "LLM 最终文本回复",
                "iterations": 迭代轮数,
                "tool_calls_log": [{"tool": ..., "args": ..., "result_summary": ...}, ...],
            }
        """
        from datetime import date as date_type
        today = date_type.today().isoformat()
        system_content = SYSTEM_PROMPT.format(today=today)

        # 初始化 messages
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": message},
        ]

        tool_calls_log = []  # 记录所有工具调用，供调试
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[agent-loop] iteration {iteration}, messages: {len(messages)}")

            # 调用 LLM（带 tools 定义）
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.3,
            )

            choice = response.choices[0]
            assistant_msg = choice.message

            # 把 assistant 回复加入 messages（可能包含 tool_calls）
            msg_dict = {"role": "assistant", "content": assistant_msg.content or ""}
            if assistant_msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ]
            messages.append(msg_dict)

            # 终止条件：没有 tool_calls → 返回文本回复
            if not assistant_msg.tool_calls:
                final_reply = assistant_msg.content or ""
                logger.info(f"[agent-loop] 终止于 iteration {iteration}，无 tool_call")
                return {
                    "reply": final_reply,
                    "iterations": iteration,
                    "tool_calls_log": tool_calls_log,
                }

            # 有 tool_calls → 逐个执行，结果回填 messages
            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                tool_args_str = tc.function.arguments
                tool_call_id = tc.id

                logger.info(f"[agent-loop] 调用工具: {tool_name}({tool_args_str})")

                try:
                    # 解析参数
                    tool_args = json.loads(tool_args_str)

                    # 执行工具
                    func = TOOL_FUNCTION_MAP.get(tool_name)
                    if func is None:
                        result = f"错误：未知工具 {tool_name}"
                    elif tool_name in ASYNC_TOOLS:
                        result = await func(**tool_args)
                    else:
                        result = func(**tool_args)

                    logger.info(f"[agent-loop] 工具 {tool_name} 执行成功: {result[:80]}")

                except Exception as e:
                    result = f"工具执行错误: {str(e)}"
                    logger.error(f"[agent-loop] 工具 {tool_name} 执行失败: {e}")

                # 记录日志
                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args_str,
                    "result_summary": result[:100],
                })

                # 工具结果回填 messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                })

        # 超过最大迭代次数，强制结束
        logger.warning(f"[agent-loop] 达到最大迭代 {self.max_iterations}，强制终止")
        return {
            "reply": "抱歉，处理步骤太多了，请简化一下你的请求。",
            "iterations": iteration,
            "tool_calls_log": tool_calls_log,
        }
