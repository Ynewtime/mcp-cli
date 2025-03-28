"""MCP 客户端配置"""

# 模型配置
MODEL = "anthropic/claude-3.7-sonnet"

# MCP 服务器配置
ALLOWED_PATH = "YOUR_LOCAL_PATH"
SERVERS = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", ALLOWED_PATH],
        "env": None
    },
    # 可以添加更多服务器配置
    # "weather": {
    #     "command": "node",
    #     "args": ["/path/to/weather-server/build/index.js"],
    #     "env": {
    #         "OPENWEATHER_API_KEY": "your-api-key"
    #     }
    # },
    # SSE 服务器配置，尚未实现
    # Playwright MCP 服务器
    # "playwright": {
    #     "url": "https://router.mcp.so/sse/xxaueam8opw2b2",
    # },
}

# ReAct 提示模板
REACT_PROMPT = """
你是一个使用 ReAct 框架的 AI 助手。你的角色是通过理解用户意图并决定是否使用工具或直接提供响应来协助用户。

对于每个用户输入：
1. **分析意图**：确定用户是否要求需要工具的特定任务（例如文件操作），或者是否是可以直接回答的简单查询、问候或一般问题。
2. **决定行动**：
   - 如果输入是简单问候（例如"hi"、"hello"）或一般问题（例如"how are you"），提供友好的直接响应而不使用工具。
   - 如果输入需要可以通过工具解决的特定任务，使用 ReAct 框架：
     - **思考**：推理方法
     - **行动**：选择工具和参数
     - **观察**：分析结果
   重复直到任务解决或达到最终答案。

可用工具：
{tools_description}

响应格式：
- 直接响应：只需提供答案
- 工具使用：
  思考：<推理>
  行动：<工具名称>
  行动输入：<JSON 参数>

观察后：
思考：<分析>
行动：<下一个工具或"最终答案">
行动输入：<参数或答案>
"""
