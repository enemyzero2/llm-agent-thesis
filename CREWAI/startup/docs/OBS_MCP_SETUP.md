# OBS MCP Server 使用指南

## 📋 概述

本项目实现了一个基于 MCP (Model Context Protocol) 的 OBS Studio 控制系统，允许 AI Agent 通过 MCP 协议实时控制 OBS 的音频、视频参数。

## 🏗️ 架构说明

```
┌─────────────────┐
│  CrewAI Agent   │  用户指令
└────────┬────────┘
         │ 调用MCP工具
         ↓
┌─────────────────┐
│   MCP Client    │  管理MCP连接
└────────┬────────┘
         │ stdio通信
         ↓
┌─────────────────┐
│   MCP Server    │  封装OBS控制
│ (obs_volume_    │
│    server.py)   │
└────────┬────────┘
         │ WebSocket
         ↓
┌─────────────────┐
│  OBS WebSocket  │  OBS插件
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   OBS Studio    │  视频处理
└─────────────────┘
```

## 🚀 快速开始

### 1. 安装 OBS Studio

1. 下载 OBS Studio: https://obsproject.com/
2. 安装并启动 OBS

### 2. 配置 OBS WebSocket

**OBS Studio 28+ 已内置 WebSocket 服务器**

1. 打开 OBS Studio
2. 点击 `工具` → `WebSocket 服务器设置`
3. 勾选 `启用 WebSocket 服务器`
4. 设置端口（默认 4455）
5. 可选：设置密码（建议留空用于测试）
6. 点击 `确定`

### 3. 安装 Python 依赖

```bash
# 进入项目目录
cd CREWAI/startup

# 安装 obs-websocket-py
pip install obs-websocket-py

# 或者添加到 pyproject.toml 后运行
pip install -e .
```

### 4. 测试连接

```bash
# 运行测试脚本
python src/mcp_servers/test_obs_server.py
```

## 📁 文件说明

### MCP Server 文件

**基础功能:**
- `src/mcp_servers/obs_control_server.py` - OBS场景和音频控制
- `src/mcp_servers/obs_volume_server.py` - OBS音量控制MCP服务器

**高级功能:**
- `src/mcp_servers/obs_subtitle_server.py` - OBS字幕显示控制
- `src/mcp_servers/obs_filter_server.py` - OBS真实滤镜控制
- `src/mcp_servers/obs_translation_server.py` - 同声传译(OpenAI Whisper + GPT)
- `src/mcp_servers/obs_recording_server.py` - 录制/直播/虚拟摄像头控制

**测试脚本:**
- `src/mcp_servers/test_obs_server.py` - 基础功能测试
- `src/mcp_servers/test_obs_advanced.py` - 高级功能测试

### 前端文件

- `src/video_frontend/player.html` - 视频播放器页面
- `src/video_frontend/player.js` - 播放器控制脚本

## 🔧 可用的 MCP 工具

### 1. list_audio_sources
列出 OBS 中所有可用的音频源

**参数**: 无

**返回示例**:
```json
{
  "sources": ["麦克风", "桌面音频"],
  "count": 2,
  "message": "找到 2 个音频源"
}
```

### 2. get_source_volume
获取指定音频源的当前音量

**参数**:
- `source_name` (string): 音频源名称

**返回示例**:
```json
{
  "source": "麦克风",
  "volume": 75,
  "message": "麦克风 的音量为 75%"
}
```

### 3. set_source_volume
设置指定音频源的音量

**参数**:
- `source_name` (string): 音频源名称
- `level` (integer): 音量值 (0-100)

**返回示例**:
```json
{
  "source": "麦克风",
  "volume": 80,
  "message": "麦克风 的音量已设置为 80%"
}
```

### 4. mute_source
静音指定的音频源

**参数**:
- `source_name` (string): 音频源名称

**返回示例**:
```json
{
  "source": "麦克风",
  "volume": 0,
  "message": "麦克风 已静音"
}
```

## 💡 使用示例

### 在 CrewAI Agent 中使用

```python
from mcp_bridge.mcp_client import MCPClient

# 创建客户端
client = MCPClient()

# 连接到OBS服务器
await client.connect(
    server_name="obs-volume",
    command="python",
    args=["src/mcp_servers/obs_volume_server.py"]
)

# 调用工具
result = await client.call_tool(
    server_name="obs-volume",
    tool_name="set_source_volume",
    arguments={"source_name": "麦克风", "level": 80}
)
```

## 🎯 已实现的高级功能

### 1. 字幕控制 (obs-subtitle)
```python
# 创建字幕
await client.call_tool("obs-subtitle", "create_subtitle", {
    "source_name": "字幕", 
    "text": "Hello World!",
    "font_size": 48,
    "color": "#FFFFFF"
})

# 更新字幕内容
await client.call_tool("obs-subtitle", "update_subtitle", {
    "source_name": "字幕",
    "text": "新的字幕内容"
})
```

### 2. 滤镜控制 (obs-filter)
```python
# 添加色彩校正滤镜
await client.call_tool("obs-filter", "add_filter", {
    "source_name": "摄像头",
    "filter_name": "暖色调",
    "filter_type": "color_correction",
    "settings": {"saturation": 0.2, "brightness": 0.1}
})

# 快速调整亮度/对比度
await client.call_tool("obs-filter", "apply_color_correction", {
    "source_name": "摄像头",
    "brightness": 0.1,
    "contrast": 0.2
})
```

### 3. 同声传译 (obs-translation)
```python
# 配置翻译
await client.call_tool("obs-translation", "set_translation_config", {
    "source_language": "en",
    "target_language": "zh",
    "subtitle_source": "翻译字幕"
})

# 翻译音频文件
await client.call_tool("obs-translation", "transcribe_and_translate", {
    "audio_path": "audio.wav"
})
```

### 4. 录制/直播控制 (obs-recording)
```python
# 开始录制
await client.call_tool("obs-recording", "start_recording", {})

# 获取录制状态
status = await client.call_tool("obs-recording", "get_record_status", {})

# 开始直播
await client.call_tool("obs-recording", "start_streaming", {})

# 启动虚拟摄像头
await client.call_tool("obs-recording", "start_virtual_cam", {})
```

### 环境变量配置

翻译功能需要设置 OpenAI API Key:
```bash
set OPENAI_API_KEY=your-api-key-here
```

## ❓ 常见问题

### Q: 连接失败怎么办？
A: 检查以下几点：
1. OBS Studio 是否正在运行
2. WebSocket 服务器是否已启用
3. 端口号是否正确（默认 4455）
4. 防火墙是否阻止了连接

### Q: 找不到音频源？
A: 确保在 OBS 中已添加音频源（如麦克风、桌面音频等）

### Q: 音量控制不生效？
A: 检查音频源名称是否正确，名称必须与 OBS 中显示的完全一致

## 📚 参考资料

- [OBS Studio 官网](https://obsproject.com/)
- [obs-websocket 文档](https://github.com/obsproject/obs-websocket)
- [obs-websocket-py](https://github.com/Elektordi/obs-websocket-py)
- [MCP 协议规范](https://modelcontextprotocol.io/)
