
## 摘要

当前实现的功能：基于 MCP stdio 和 OpenRouter 的简单 Agent 工具。


## 依赖

请确保你的执行环境已经按照如下依赖：

1. Python
2. Node.js
3. [uv](https://docs.astral.sh/uv/)

同时确保你已经拿到 [OpenRouter](https://openrouter.ai/) 的 API Key


## 使用

1. 使用 uv 安装依赖: `uv sync`
2. 配置 `src/config.py`，如 `ALLOWED_PATH` 变量
3. 配置 `.env` 环境变量，设置 `OPENAI_API_KEY=YOUR_OPENROUTER_API_KEY`
4. 启动 `source .venv/bin/activate`，Windows 使用：`.venv\Scripts\activate`
5. 运行 `python src/openrouter-agent.py`
6. 调试 `python src/openrouter-agent.py --debug`
