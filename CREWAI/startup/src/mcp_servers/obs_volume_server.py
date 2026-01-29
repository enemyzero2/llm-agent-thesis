"""
=============================================================================
OBS音量控制 MCP Server
文件: src/mcp_servers/obs_volume_server.py
说明: 通过OBS WebSocket真实控制OBS音频源音量
=============================================================================
"""

import asyncio
import json
from typing import Optional
from mcp.server import Server
from mcp.types import Tool, TextContent

# 尝试导入obs-websocket-py
try:
    from obswebsocket import obsws, requests as obs_requests
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    print("[WARNING] obs-websocket-py未安装，将使用模拟模式")
    print("安装命令: pip install obs-websocket-py")


# 创建Server实例
server = Server("obs-volume-control")

# OBS连接配置
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "123456"  # OBS WebSocket密码

# OBS WebSocket客户端
obs_client: Optional[obsws] = None


def connect_obs():
    """连接到OBS WebSocket"""
    global obs_client

    if not OBS_AVAILABLE:
        print("[INFO] 使用模拟模式")
        return False

    try:
        obs_client = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
        obs_client.connect()
        print(f"[OK] 已连接到OBS ({OBS_HOST}:{OBS_PORT})")
        return True
    except Exception as e:
        print(f"[ERROR] 连接OBS失败: {e}")
        print("[INFO] 切换到模拟模式")
        obs_client = None
        return False


def get_audio_sources():
    """获取所有音频源列表"""
    if obs_client is None:
        return []

    try:
        response = obs_client.call(obs_requests.GetInputList())
        audio_sources = []
        for input_item in response.getInputs():
            # 检查是否是音频源
            input_kind = input_item.get('inputKind', '')
            if 'audio' in input_kind.lower() or input_item.get('unversionedInputKind') in ['wasapi_input_capture', 'wasapi_output_capture']:
                audio_sources.append(input_item['inputName'])
        return audio_sources
    except Exception as e:
        print(f"[ERROR] 获取音频源失败: {e}")
        return []


def get_volume(source_name: str) -> Optional[float]:
    """获取指定音频源的音量 (0.0-1.0)"""
    if obs_client is None:
        return None

    try:
        response = obs_client.call(obs_requests.GetInputVolume(source_name))
        # OBS返回的是dB值，需要转换为0-1的线性值
        volume_db = response.getInputVolumeDb()
        # 简化：使用mul值（0-1）
        volume_mul = response.getInputVolumeMul()
        return volume_mul
    except Exception as e:
        print(f"[ERROR] 获取音量失败: {e}")
        return None


def set_volume(source_name: str, volume: float) -> bool:
    """设置指定音频源的音量 (0.0-1.0)"""
    if obs_client is None:
        return False

    try:
        obs_client.call(obs_requests.SetInputVolume(source_name, volumeMul=volume))
        return True
    except Exception as e:
        print(f"[ERROR] 设置音量失败: {e}")
        return False


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的OBS音量控制工具"""
    return [
        Tool(
            name="list_audio_sources",
            description="列出OBS中所有可用的音频源",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_source_volume",
            description="获取指定音频源的当前音量(0-100)",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "音频源名称"
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="set_source_volume",
            description="设置指定音频源的音量(0-100)",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "音频源名称"
                    },
                    "level": {
                        "type": "integer",
                        "description": "音量值(0-100)",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["source_name", "level"]
            }
        ),
        Tool(
            name="mute_source",
            description="静音指定的音频源",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "音频源名称"
                    }
                },
                "required": ["source_name"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""

    if name == "list_audio_sources":
        sources = get_audio_sources()
        if sources:
            result = {
                "sources": sources,
                "count": len(sources),
                "message": f"找到 {len(sources)} 个音频源"
            }
        else:
            result = {
                "sources": [],
                "message": "未找到音频源或未连接到OBS"
            }

    elif name == "get_source_volume":
        source_name = arguments.get("source_name")
        volume = get_volume(source_name)

        if volume is not None:
            volume_percent = int(volume * 100)
            result = {
                "source": source_name,
                "volume": volume_percent,
                "message": f"{source_name} 的音量为 {volume_percent}%"
            }
        else:
            result = {
                "error": f"无法获取 {source_name} 的音量"
            }

    elif name == "set_source_volume":
        source_name = arguments.get("source_name")
        level = arguments.get("level", 50)
        level = max(0, min(100, level))

        # 转换为0-1范围
        volume = level / 100.0
        success = set_volume(source_name, volume)

        if success:
            result = {
                "source": source_name,
                "volume": level,
                "message": f"{source_name} 的音量已设置为 {level}%"
            }
        else:
            result = {
                "error": f"无法设置 {source_name} 的音量"
            }

    elif name == "mute_source":
        source_name = arguments.get("source_name")
        success = set_volume(source_name, 0.0)

        if success:
            result = {
                "source": source_name,
                "volume": 0,
                "message": f"{source_name} 已静音"
            }
        else:
            result = {
                "error": f"无法静音 {source_name}"
            }

    else:
        result = {"error": f"未知工具: {name}"}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def main():
    """运行Server"""
    from mcp.server.stdio import stdio_server

    # 尝试连接OBS
    print("=" * 60)
    print("OBS音量控制 MCP Server 启动中...")
    print("=" * 60)
    connect_obs()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
