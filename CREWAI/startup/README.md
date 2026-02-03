# CrewAI OBS控制系统

基于 [CrewAI](https://crewai.com) 框架的智能Agent系统，用于控制OBS直播软件。

## 项目简介

这个项目让你可以用**自然语言**控制OBS，比如说"把麦克风音量调到50%"，AI Agent会自动完成操作。

## 安装

确保已安装 Python >=3.10 <3.14。本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

```bash
# 安装uv
pip install uv

# 安装依赖
uv sync
```
## 配置

在 `.env` 文件中配置：

```env
OPENAI_API_KEY="你的API密钥"
OPENAI_API_BASE="https://你的API地址/v1"
MODEL="模型名称"
```

## 运行

```bash
# 1. 启动后端服务（需要先打开OBS）
uv run python src/main.py backend

# 2. 运行Agent（新终端）
uv run python src/main.py agent "获取所有场景"
```

## 项目结构

```
src/
├── config.py          # 配置管理
├── main.py            # 入口文件
├── tools/             # 工具定义
├── agents/            # Agent定义
├── mcp_servers/       # MCP服务器
└── video_backend/     # Flask后端
```

## 学习指南

详细的学习指南请查看 [LEARNING_GUIDE.md](LEARNING_GUIDE.md)

## 相关文档

- [CrewAI 文档](https://docs.crewai.com)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewai)
