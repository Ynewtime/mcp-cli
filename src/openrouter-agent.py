#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ReAct MCP 客户端实现
基于 Model Context Protocol (MCP) 和 OpenRouter API 的交互式工具调用客户端
"""

import argparse
import asyncio
import time
import sys
import threading
import logging
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import datetime
import json
import re

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from dotenv import load_dotenv

from config import MODEL, SERVERS, REACT_PROMPT


# 加载环境变量
load_dotenv()


# 配置日志系统
def setup_logging(debug_mode: bool):
    """配置日志系统

    Args:
        debug_mode: 是否启用调试模式
    """
    level = logging.DEBUG if debug_mode else logging.INFO

    # 创建格式化器
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # 文件处理器 - 处理所有级别的日志
    file_handler = logging.FileHandler("mcp_client.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)  # 文件记录所有级别的日志

    # 终端处理器 - 只处理 WARNING 及以上级别的日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)  # 终端只显示 WARNING 及以上

    # 配置根日志记录器
    logging.basicConfig(level=level, handlers=[file_handler, console_handler])

    # 设置 OpenAI 客户端日志级别
    logging.getLogger("openai").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class TerminalSpinner:
    """终端旋转动画指示器

    Attributes:
        message: 显示的消息
        running: 是否正在运行
        spinner_chars: 动画字符序列
    """

    def __init__(self, message="处理中..."):
        self.message = message
        self.running = False
        self.spinner_thread = None
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0
        self.is_tty = sys.stdout.isatty()

    def spin(self):
        """执行旋转动画"""
        while self.running:
            if self.is_tty:
                sys.stdout.write(
                    f"\r{self.spinner_chars[self.spinner_idx]} {self.message}"
                )
                sys.stdout.flush()
                self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
            time.sleep(0.1)

    def start(self, message=None):
        """启动动画

        Args:
            message: 可选的新消息
        """
        if message:
            self.message = message
        if self.running:
            self.stop()
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin, daemon=True)
        self.spinner_thread.start()

    def stop(self):
        """停止动画并清理终端行"""
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()
        if self.is_tty:
            sys.stdout.write(f"\r{' ' * (len(self.message) + 2)}\r")
            sys.stdout.flush()


class MCPClient:
    """MCP 客户端主类

    Attributes:
        session: MCP 客户端会话
        openai: OpenAI 客户端实例
        tools: 可用工具列表
        messages: 对话消息历史
        conversation_history: 完整对话历史记录
    """

    def __init__(self, debug_mode: bool = False):
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(base_url="https://openrouter.ai/api/v1")
        self.tools = []
        self.messages = []
        self.conversation_history = []
        self.spinner = TerminalSpinner()
        self.debug_mode = debug_mode
        self.tool_to_session = {}  # 工具名称到会话的映射
        logger.info("MCPClient 初始化完成")

    def _sanitize_text(self, text: Any) -> str:
        """确保文本是有效的 UTF-8 编码

        Args:
            text: 输入文本

        Returns:
            清理后的文本
        """
        if not isinstance(text, str):
            text = str(text)
        return text.encode("utf-8", errors="replace").decode("utf-8")

    async def connect_to_server(self, server_config: Dict[str, Any]):
        """连接到单个 MCP 服务器

        Args:
            server_config: 单个服务器配置字典
        """
        try:
            logger.info("正在连接到 MCP 服务器...")
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            await session.initialize()

            # 获取可用工具列表并合并，同时建立工具到会话的映射
            response = await session.list_tools()
            self.tools.extend(response.tools)
            self.sessions.append(session)
            for tool in response.tools:
                self.tool_to_session[tool.name] = session
            logger.info(
                f"成功连接到服务器，可用工具: {[tool.name for tool in response.tools]}"
            )

            # 工具列表已合并到self.tools中
            # 系统消息将在process_query中统一初始化
        except Exception as e:
            logger.error(f"连接服务器失败: {e}")
            raise

    def _format_tools_description(self, tools: List[Any]) -> str:
        """格式化工具描述

        Args:
            tools: 工具列表

        Returns:
            格式化的工具描述字符串
        """
        return "\n".join(
            f"工具: {tool.name}\n描述: {tool.description}\n参数:\n"
            + "\n".join(
                f"- {name}: {info.get('description', '无描述')}{' (必填)' if name in tool.inputSchema.get('required', []) else ''}"
                for name, info in tool.inputSchema["properties"].items()
            )
            + "\n"
            for tool in tools
        )

    def _convert_tool_format(self, tool: Any) -> Dict[str, Any]:
        """转换工具格式为 OpenAI 兼容格式

        Args:
            tool: 工具对象

        Returns:
            转换后的工具字典
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.inputSchema["properties"],
                    "required": tool.inputSchema.get("required", []),
                },
            },
        }

    def _sanitize_json_input(self, input_str: str) -> Optional[Dict[str, Any]]:
        """增强 JSON 解析鲁棒性，处理各种 JSON 格式变体

        Args:
            input_str: 输入 JSON 字符串
        Returns:
            解析后的字典或 None
        """
        if not input_str or not isinstance(input_str, str):
            logger.debug("输入为空或非字符串")
            return {}

        input_str = input_str.strip()
        if not input_str:
            logger.debug("输入为空字符串")
            return {}

        original_input = input_str  # 保存原始输入用于错误报告

        # 尝试1: 标准JSON解析
        try:
            return json.loads(input_str)
        except json.JSONDecodeError as e:
            logger.debug(f"标准JSON解析失败，尝试其他方法。错误: {e}")

        # 尝试2: 处理单引号字符串
        try:
            # 替换单引号为双引号，但排除转义的单引号
            single_quote_fixed = re.sub(r"(?<!\\)'", '"', input_str)
            return json.loads(single_quote_fixed)
        except json.JSONDecodeError as e:
            logger.debug(f"单引号处理失败，错误: {e}")

        # 尝试3: 处理未加引号的键名
        try:
            # 为未加引号的键名添加引号
            unquoted_keys_fixed = re.sub(
                r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)", r'\1"\2"\3', input_str
            )
            return json.loads(unquoted_keys_fixed)
        except json.JSONDecodeError as e:
            logger.debug(f"未加引号键名处理失败，错误: {e}")

        # 尝试4: 综合修复常见问题
        try:
            # 1. 处理单引号
            fixed = re.sub(r"(?<!\\)'", '"', input_str)
            # 2. 处理未加引号的键名
            fixed = re.sub(
                r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)", r'\1"\2"\3', fixed
            )
            # 3. 处理尾随逗号
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            # 4. 处理未加引号的值
            fixed = re.sub(r':\s*([^"{}\[\],\s]+)(?=\s*[,}])', r':"\1"', fixed)

            return json.loads(fixed)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析最终失败。原始输入: {original_input}，错误: {e}")
            return None

    def _extract_paths_from_text(self, text: str) -> List[str]:
        """从文本中提取路径

        Args:
            text: 输入文本

        Returns:
            提取的路径列表
        """
        if not text or not isinstance(text, str):
            return []
        paths = []
        try:
            paths = re.findall(r'(/[^\s",}]+)', text)
        except re.error as e:
            logger.warning(f"提取路径时正则表达式错误: {e}")
        if not paths:
            try:
                source_match = re.search(
                    r'["\']?source["\']?\s*:\s*["\']?([^",}]+)["\']?', text
                )
                if source_match:
                    paths.append(source_match.group(1).strip())
            except re.error as e:
                logger.warning(f"提取源路径时正则表达式错误: {e}")
            try:
                dest_match = re.search(
                    r'["\']?destination["\']?\s*:\s*["\']?([^",}]+)["\']?', text
                )
                if dest_match:
                    paths.append(dest_match.group(1).strip())
            except re.error as e:
                logger.warning(f"提取目标路径时正则表达式错误: {e}")
        return paths

    def _validate_tool_parameters(
        self, tool_name: str, tool_args: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """验证和修复工具参数

        Args:
            tool_name: 工具名称
            tool_args: 工具参数

        Returns:
            验证后的工具参数
        """
        if tool_args is None:
            return {}

        tool_def = next((t for t in self.tools if t.name == tool_name), None)
        if not tool_def:
            logger.warning(f"未找到工具: {tool_name}")
            return tool_args

        required_params = tool_def.inputSchema.get("required", [])
        fixed_args = tool_args.copy()

        # 特殊处理 input 参数
        if "input" in fixed_args and isinstance(fixed_args["input"], str):
            input_value = fixed_args["input"].strip()
            if input_value.startswith("{"):
                parsed_input = self._sanitize_json_input(input_value)
                if parsed_input is not None:
                    fixed_args.update(parsed_input)
                    del fixed_args["input"]
                elif tool_name == "move_file":
                    paths = self._extract_paths_from_text(input_value)
                    if len(paths) >= 2:
                        fixed_args["source"] = paths[0]
                        fixed_args["destination"] = paths[1]
                        del fixed_args["input"]

        # 特定工具的参数修复
        if (
            tool_name == "list_directory"
            and "path" not in fixed_args
            and "input" in fixed_args
        ):
            fixed_args["path"] = fixed_args.pop("input")
        elif (
            tool_name == "search_files"
            and "pattern" not in fixed_args
            and "input" in fixed_args
        ):
            fixed_args["pattern"] = fixed_args.pop("input")
        elif tool_name == "move_file" and (
            "source" not in fixed_args or "destination" not in fixed_args
        ):
            paths = self._extract_paths_from_text(json.dumps(fixed_args))
            if len(paths) >= 2:
                fixed_args = {"source": paths[0], "destination": paths[1]}

        # 检查必填参数
        missing_params = [p for p in required_params if p not in fixed_args]
        if missing_params:
            logger.warning(f"工具 {tool_name} 缺少必填参数: {missing_params}")

        return fixed_args

    def _print_step(self, text: str, with_spinner: bool = False, level: str = "info"):
        """打印步骤信息并记录日志

        Args:
            text: 要打印的文本
            with_spinner: 是否显示旋转动画
            level: 日志级别
        """
        self.spinner.stop()

        # 记录日志
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(text)

        # 打印到控制台
        if with_spinner:
            self.spinner.start(text)
            time.sleep(0.5)
            self.spinner.stop()

        # 添加颜色和格式
        prefixes = {
            "info": "ℹ️",
            "error": "❌",
            "warning": "⚠️",
            "success": "✅",
            "action": "🔧",
            "thought": "🤔",
            "observation": "👁️",
        }

        prefix = next((p for p in prefixes if p in level.lower()), "")
        if prefix:
            text = f"{prefixes[prefix]} {text}"

        print(f"{text}\n")

    async def process_query(self, query: str) -> str:
        """处理用户查询

        Args:
            query: 用户查询文本

        Returns:
            处理结果
        """
        query = self._sanitize_text(query)
        logger.info(f"处理查询: {query}")

        # 初始化消息历史
        if not self.messages:
            tools_desc = self._format_tools_description(self.tools)
            self.messages = [
                {
                    "role": "system",
                    "content": REACT_PROMPT.format(tools_description=tools_desc),
                }
            ]

        self.messages.append({"role": "user", "content": query})

        # 简单意图识别：检查是否为问候
        greetings = ["hi", "hello", "hey", "greetings", "你好"]
        if query.lower().strip() in greetings:
            response = "你好！我是你的AI助手，有什么可以帮你的吗？"
            self._print_step(response, level="success")

            self.conversation_history.append(
                {
                    "query": query,
                    "response": response,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "tokens": {"input": 0, "output": 0, "cost": 0.0},
                }
            )
            return response

        # 进入 ReAct 循环
        available_tools = [self._convert_tool_format(tool) for tool in self.tools]
        conversation_log = []
        total_input_tokens, total_output_tokens = 0, 0
        max_iterations, iterations = 100, 0
        start_time = time.time()

        while iterations < max_iterations:
            iterations += 1
            self.spinner.start(f"思考中... (第 {iterations} 轮)")

            try:
                logger.debug(
                    f"API请求原始数据: available_tools:{available_tools}; self.messages:{self.messages}"
                )
                response = self.openai.chat.completions.create(
                    model=MODEL, tools=available_tools, messages=self.messages
                )
                logger.debug(f"API响应原始数据: {response}")

                # 记录 token 使用情况
                if response.usage:
                    total_input_tokens += response.usage.prompt_tokens or 0
                    total_output_tokens += response.usage.completion_tokens or 0
            except Exception as e:
                self._print_step(f"调用模型失败: {e}", level="error")
                break

            self.spinner.stop()

            # 记录 token 使用情况
            if response.usage:
                total_input_tokens += response.usage.prompt_tokens or 0
                total_output_tokens += response.usage.completion_tokens or 0

            if not response.choices:
                self._print_step("错误: 没有可用的响应选项", level="error")
                break

            content = response.choices[0].message
            self.messages.append(content.model_dump())
            content_text = content.content or ""
            content_text = self._sanitize_text(content_text)
            logger.debug(f"AI响应: {content_text}")

            # 检查是否为直接响应
            if (
                "Action:" not in content_text
                and "Thought:" not in content_text
                and not content.tool_calls
            ):
                self._print_step(f"响应: {content_text}", level="success")
                break

            # 提取思考过程
            reasoning = re.search(
                r"Thought: (.*?)(?=Action:|$)", content_text, re.DOTALL
            )
            reasoning = reasoning.group(1).strip() if reasoning else ""
            if reasoning:
                self._print_step(f"思考: {reasoning}", level="thought")

            # 处理工具调用
            if content.tool_calls:
                # 处理所有工具调用，而不仅是第一个
                for tool_call in content.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args_str = tool_call.function.arguments or "{}"
                    tool_call_id = tool_call.id

                    try:
                        # 解析工具参数
                        tool_args = (
                            self._sanitize_json_input(tool_args_str)
                            if tool_args_str and tool_args_str.startswith("{")
                            else {"input": tool_args_str}
                        )
                        if tool_args is None:
                            raise ValueError("无效的JSON参数")

                        self._print_step(
                            f"执行操作: {tool_name}", level="action", with_spinner=True
                        )
                        self._print_step(
                            f"操作输入: {json.dumps(tool_args, indent=2, ensure_ascii=False)}",
                            level="info",
                        )

                        # 验证和修复工具参数
                        tool_args = self._validate_tool_parameters(tool_name, tool_args)

                        # 执行工具调用
                        self.spinner.start(f"正在执行 {tool_name}...")
                        logger.debug(f"调用工具: {tool_name}，参数: {tool_args}")
                        # 直接通过工具名称找到对应会话调用
                        session = self.tool_to_session.get(tool_name)
                        if not session:
                            raise RuntimeError(f"找不到工具 {tool_name} 对应的会话")
                        result = await session.call_tool(tool_name, tool_args)
                        observation = (
                            result.content
                            if hasattr(result, "content")
                            else str(result)
                        )
                        observation = self._sanitize_text(observation)

                        self._print_step(
                            f"观察结果: {observation}", level="observation"
                        )

                        # 为每个工具调用添加响应消息
                        tool_message = {
                            "role": "tool",
                            "name": tool_name,
                            "content": observation,
                            "tool_call_id": tool_call_id,
                        }
                        self.messages.append(tool_message)
                    except Exception as e:
                        error_msg = f"调用 {tool_name} 失败: {e}"
                        self._print_step(error_msg, level="error")
                        logger.error(f"工具调用异常: {error_msg}", exc_info=True)
                        self.messages.append(
                            {
                                "role": "tool",
                                "name": tool_name,
                                "content": error_msg,
                                "tool_call_id": tool_call_id,
                            }
                        )
                    finally:
                        self.spinner.stop()
            elif "Action:" in content_text:
                action = re.search(
                    r"Action:\s*(.*?)(?=\n|$)", content_text, re.IGNORECASE
                )
                action_input = re.search(
                    r"Action Input:\s*(.*?)(?=\n|Action:|$)",
                    content_text,
                    re.DOTALL | re.IGNORECASE,
                )
                if action and action_input:
                    tool_name = action.group(1).strip()
                    tool_args_str = action_input.group(1).strip()
                    if tool_name.lower() == "final answer":
                        self._print_step(f"最终答案: {tool_args_str}", level="success")
                        break
            else:
                # 如果没有工具调用，视为直接响应
                self._print_step(f"响应: {content_text}", level="success")
                break

        # 处理达到最大迭代次数的情况
        if iterations >= max_iterations:
            self._print_step("⚠️ 达到最大迭代次数，获取最终答案...", level="warning")
            try:
                response = self.openai.chat.completions.create(
                    model=MODEL,
                    messages=self.messages
                    + [{"role": "user", "content": "请提供最终答案。"}],
                )
                logger.debug(f"API响应原始数据: {response}")
                total_input_tokens += response.usage.prompt_tokens or 0
                total_output_tokens += response.usage.completion_tokens or 0

                # 记录 token 使用情况
                if response.usage:
                    total_input_tokens += response.usage.prompt_tokens or 0
                    total_output_tokens += response.usage.completion_tokens or 0

                final_answer = self._sanitize_text(
                    response.choices[0].message.content
                    if response.choices
                    else "无可用答案"
                )
                self._print_step(f"最终答案: {final_answer}", level="success")
            except Exception as e:
                self._print_step(f"获取最终答案失败: {e}", level="error")

        # 生成执行摘要
        execution_time = time.time() - start_time

        summary = f"""
任务执行摘要:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ 执行时间: {execution_time:.2f} 秒
🔄 迭代次数: {iterations}
💬 输入Tokens: {total_input_tokens:,}
💬 输出Tokens: {total_output_tokens:,}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self._print_step(summary, level="info")
        conversation_log.append(summary)

        # 记录对话历史
        self.conversation_history.append(
            {
                "query": query,
                "response": "\n".join(conversation_log),
                "timestamp": datetime.datetime.now().isoformat(),
                "tokens": {"input": total_input_tokens, "output": total_output_tokens},
                "iterations": iterations,
            }
        )

        # 限制消息历史长度
        if len(self.messages) > 10:
            self.messages = [self.messages[0]] + self.messages[-8:]

        return "\n".join(conversation_log)

    async def chat_loop(self):
        """交互式聊天循环"""
        print("\n🚀 ReAct MCP 客户端已启动!")
        print("输入查询或 'quit'/'exit'/'bye' 退出\n")

        session_start_time = datetime.datetime.now()
        total_queries = 0
        total_tokens = {"input": 0, "output": 0}

        while True:
            try:
                query = input("> 请输入查询: ").strip()
                query = self._sanitize_text(query)

                if query.lower() in ["quit", "exit", "bye"]:
                    duration = (
                        datetime.datetime.now() - session_start_time
                    ).total_seconds()
                    summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 会话统计:
⏱️ 持续时间: {duration:.2f} 秒
🔢 查询次数: {total_queries}
💬 总输入Tokens: {total_tokens["input"]:,}
💬 总输出Tokens: {total_tokens["output"]:,}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
感谢使用，再见!
"""
                    print(summary)
                    break

                if not query:
                    continue

                # 处理查询
                await self.process_query(query)
                total_queries += 1

                # 更新统计信息
                latest = self.conversation_history[-1]["tokens"]
                total_tokens["input"] += latest["input"]
                total_tokens["output"] += latest["output"]

            except KeyboardInterrupt:
                print("\n检测到中断信号，退出中...")
                break
            except Exception as e:
                logger.error(f"处理查询时出错: {e}", exc_info=True)
                print(f"❌ 发生错误: {e}\n")

    async def cleanup(self):
        """清理资源"""
        try:
            await self.exit_stack.aclose()
            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")


async def main(debug_mode: bool):
    """主入口函数

    Args:
        debug_mode: 是否启用调试模式
    """
    logger.info("启动 MCP 客户端...")
    client = MCPClient(debug_mode=debug_mode)

    try:
        # Connect to all available servers
        for server_name, server_config in SERVERS.items():
            try:
                await client.connect_to_server(server_config)
                logger.info(f"成功连接到服务器: {server_name}")
            except Exception as e:
                logger.error(f"连接服务器 {server_name} 失败: {e}")
                if server_name == "filesystem":
                    raise  # Re-raise if filesystem server fails as it's critical

        await client.chat_loop()
    except Exception as e:
        logger.error(f"客户端运行出错: {e}", exc_info=True)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ReAct MCP 客户端")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    # 配置日志
    setup_logging(args.debug)

    # 运行主程序
    try:
        asyncio.run(main(args.debug))
    except KeyboardInterrupt:
        print("\n程序已终止")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"程序崩溃: {e}", exc_info=True)
        sys.exit(1)
