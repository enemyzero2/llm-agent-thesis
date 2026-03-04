# -*- coding: utf-8 -*-
"""
=============================================================================
OBS控制 MCP Server
文件: src/mcp_servers/obs_control_server.py
说明: 直接通过OBS WebSocket控制场景和音量（不依赖Flask后端）
=============================================================================
"""

import asyncio
import json
import os
import sys
from typing import Optional

from mcp.server import Server
from mcp.types import Tool, TextContent

# 尝试导入obs-websocket-py
try:
    from obswebsocket import obsws, requests as obs_requests
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    print("[WARNING] obs-websocket-py未安装", file=sys.stderr)

# MCP Server实例
SERVER_NAME = "obs-control"
server = Server(SERVER_NAME)

# OBS连接配置（从环境变量读取，与config.py保持一致）
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "123456")

# OBS WebSocket客户端
obs_client: Optional[obsws] = None


def connect_obs() -> bool:
    """连接到OBS WebSocket"""
    global obs_client

    if not OBS_AVAILABLE:
        print(f"[{SERVER_NAME}] obs-websocket-py未安装", file=sys.stderr)
        return False

    try:
        obs_client = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
        obs_client.connect()
        print(f"[{SERVER_NAME}] 已连接到OBS ({OBS_HOST}:{OBS_PORT})", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[{SERVER_NAME}] 连接OBS失败: {e}", file=sys.stderr)
        obs_client = None
        return False


# ============= 场景控制 =============

def get_scenes() -> dict:
    """获取所有场景列表"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    try:
        result = obs_client.call(obs_requests.GetSceneList())
        scenes = [s['sceneName'] for s in result.getScenes()]
        current = result.getCurrentProgramSceneName()
        return {"scenes": scenes, "current": current}
    except Exception as e:
        return {"error": f"获取场景失败: {str(e)}"}


def switch_scene(scene_name: str) -> dict:
    """切换到指定场景"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    try:
        obs_client.call(obs_requests.SetCurrentProgramScene(sceneName=scene_name))
        return {"success": True, "scene": scene_name, "message": f"已切换到场景: {scene_name}"}
    except Exception as e:
        return {"error": f"切换场景失败: {str(e)}"}


# ============= 音量控制 =============

def get_audio_sources() -> dict:
    """获取所有音频源及其音量"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    try:
        inputs = obs_client.call(obs_requests.GetInputList())
        audio_sources = []
        for inp in inputs.getInputs():
            try:
                vol = obs_client.call(obs_requests.GetInputVolume(inputName=inp['inputName']))
                audio_sources.append({
                    "name": inp['inputName'],
                    "volume": int(vol.getInputVolumeMul() * 100)
                })
            except:
                pass
        return {"sources": audio_sources}
    except Exception as e:
        return {"error": f"获取音频源失败: {str(e)}"}


def set_volume(source: str, volume: int) -> dict:
    """设置音频源音量（0-100）"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    try:
        obs_client.call(obs_requests.SetInputVolume(
            inputName=source,
            inputVolumeMul=volume / 100.0
        ))
        return {"success": True, "source": source, "volume": volume, "message": f"已设置 {source} 音量为 {volume}"}
    except Exception as e:
        return {"error": f"设置音量失败: {str(e)}"}


# ============= MCP 接口 =============

@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="get_scenes",
            description="获取OBS所有场景列表。【重要】如果返回结果包含error字段，说明OBS未连接，你必须如实告知用户连接失败，不得编造场景列表。",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="switch_scene",
            description="切换到指定场景。【重要】如果返回error，必须如实告知用户操作失败。",
            inputSchema={
                "type": "object",
                "properties": {
                    "scene": {"type": "string", "description": "场景名称"}
                },
                "required": ["scene"]
            }
        ),
        Tool(
            name="get_audio_sources",
            description="获取所有音频源及其音量。【重要】如果返回error，说明OBS未连接，必须如实告知用户。",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="set_volume",
            description="设置音频源音量(0-100)。【重要】如果返回error，必须如实告知用户操作失败。",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "音频源名称"},
                    "volume": {"type": "integer", "description": "音量(0-100)"}
                },
                "required": ["source", "volume"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    tool_handlers = {
        "get_scenes":       lambda args: get_scenes(),
        "switch_scene":     lambda args: switch_scene(args["scene"]),
        "get_audio_sources": lambda args: get_audio_sources(),
        "set_volume":       lambda args: set_volume(args["source"], args["volume"]),
    }

    handler = tool_handlers.get(name)
    if handler:
        result = handler(arguments)
    else:
        result = {"error": f"未知工具: {name}"}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def main():
    """启动MCP Server"""
    from mcp.server.stdio import stdio_server

    print(f"[{SERVER_NAME}] MCP Server 启动中...", file=sys.stderr)
    connect_obs()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
