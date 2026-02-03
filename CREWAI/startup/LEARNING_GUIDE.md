# CrewAI OBS控制系统 - 学习指南

> 这是一个使用CrewAI框架构建的智能Agent系统，用于控制OBS直播软件。

## 目录

1. [项目是什么？](#项目是什么)
2. [核心概念](#核心概念)
3. [项目结构](#项目结构)
4. [学习路线](#学习路线)
5. [如何运行](#如何运行)

---

## 项目是什么？

这个项目让你可以用**自然语言**控制OBS直播软件。

比如你说："把麦克风音量调到50%"，AI Agent就会自动调用相应的工具完成操作。

**技术栈：**
- CrewAI - AI Agent框架
- MCP (Model Context Protocol) - Agent与工具通信的协议
- Flask - 后端服务
- OBS WebSocket - 与OBS通信

---

## 核心概念

### 1. 什么是Agent？

Agent = LLM（大语言模型）+ 工具（Tools）

```
用户输入 → Agent思考 → 调用工具 → 返回结果
   ↓           ↓           ↓
"调音量"    "需要用     执行API
           set_volume"   调用
```

### 2. 什么是CrewAI？

CrewAI是一个Python框架，帮你快速创建AI Agent。

核心组件：
- **Agent** - 有角色、目标、工具的AI助手
- **Task** - 要完成的任务
- **Crew** - 管理多个Agent协作
- **Tool** - Agent可以调用的工具

### 3. 什么是MCP？

MCP是一种协议，让Agent和工具之间通信更标准化。

```
传统方式: Agent → 直接调用Python函数
MCP方式:  Agent → MCP协议 → MCP Server → 执行操作
```

---

## 项目结构

```
startup/
├── .env                    # 配置文件（API密钥等）
├── pyproject.toml          # 项目依赖
├── src/
│   ├── config.py           # ⭐ 第1步：配置管理
│   ├── main.py             # ⭐ 入口文件
│   │
│   ├── tools/              # ⭐ 第2步：工具定义
│   │   └── obs_tools.py    # OBS控制工具
│   │
│   ├── agents/             # ⭐ 第3步：Agent定义
│   │   ├── obs_agent.py    # 传统方式Agent
│   │   └── mcp_obs_agent.py # MCP方式Agent
│   │
│   ├── mcp_servers/        # ⭐ 第4步：MCP服务器
│   │   └── obs_control_server.py
│   │
│   └── video_backend/      # ⭐ 第5步：Flask后端
│       └── app.py          # 连接OBS的HTTP服务
```

---

## 学习路线

### 第1步：理解配置 (config.py)

打开 `src/config.py`，这是最简单的文件：

```python
# 从.env读取配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")

# 创建LLM实例
def get_llm():
    return LLM(model=f"openai/{MODEL}", ...)
```

**学习要点：**
- 如何使用python-dotenv读取环境变量
- 如何配置第三方API

---

### 第2步：理解工具 (obs_tools.py)

打开 `src/tools/obs_tools.py`：

```python
class GetScenesTool(BaseTool):
    name = "get_scenes"
    description = "获取OBS所有场景列表"

    def _run(self):
        return call_api("scenes")  # 调用后端API
```

**学习要点：**
- CrewAI的Tool继承自BaseTool
- 必须定义name、description和_run方法
- Agent根据description决定何时使用这个工具

---

### 第3步：理解Agent (obs_agent.py)

打开 `src/agents/obs_agent.py`：

```python
# 创建Agent
obs_agent = Agent(
    role="OBS视频控制专家",      # 角色
    goal="控制OBS的场景和音量",   # 目标
    backstory="你是专业的...",   # 背景故事
    tools=[GetScenesTool(), ...], # 可用工具
    llm=llm                      # 使用的LLM
)

# 创建任务并执行
task = Task(description="用户指令", agent=obs_agent)
crew = Crew(agents=[obs_agent], tasks=[task])
crew.kickoff()  # 开始执行
```

**学习要点：**
- Agent需要role、goal、backstory
- tools列表定义Agent能用什么工具
- Crew.kickoff()启动执行

---

### 第4步：理解MCP Server (obs_control_server.py)

打开 `src/mcp_servers/obs_control_server.py`：

```python
server = Server("obs-control")

@server.list_tools()
async def list_tools():
    return [Tool(name="get_scenes", ...)]

@server.call_tool()
async def call_tool(name, arguments):
    if name == "get_scenes":
        return call_api("scenes")
```

**学习要点：**
- MCP Server通过装饰器定义工具
- list_tools告诉Agent有哪些工具
- call_tool处理工具调用

---

### 第5步：理解后端 (video_backend/app.py)

打开 `src/video_backend/app.py`：

```python
@app.route('/api/scenes')
def get_scenes():
    ws = get_obs()  # 连接OBS
    result = ws.call(obs_requests.GetSceneList())
    return jsonify({"scenes": ...})
```

**学习要点：**
- Flask提供HTTP API
- 使用obs-websocket-py连接OBS
- Agent的工具最终调用这些API

---

## 如何运行

### 准备工作

1. 安装OBS并启用WebSocket（工具→WebSocket服务器设置）
2. 确保.env配置正确

### 运行步骤

```bash
# 1. 安装依赖
uv sync

# 2. 启动Flask后端（新终端）
uv run python src/main.py backend

# 3. 运行Agent（另一个终端）
uv run python src/main.py agent "获取所有场景"
```

---

## 数据流图

```
┌─────────────────────────────────────────────────────────┐
│                      用户输入                            │
│                  "把音量调到50%"                         │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│                    CrewAI Agent                         │
│  思考: 需要用set_volume工具，参数是source和volume        │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│                   Tool / MCP Server                     │
│              调用 POST /api/audio/volume                │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│                    Flask 后端                           │
│           ws.call(SetInputVolume(...))                  │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│                       OBS                               │
│                   音量已调整                             │
└─────────────────────────────────────────────────────────┘
```

---

## 下一步学习

1. **修改Agent的角色和目标**，看看行为有什么变化
2. **添加新的Tool**，比如控制录制开始/停止
3. **尝试多Agent协作**，让多个Agent配合完成任务

有问题随时问！
