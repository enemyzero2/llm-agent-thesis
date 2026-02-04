# -*- coding: utf-8 -*-
"""
OBS控制 MCP Server
通过HTTP API与Flask后端通信，控制OBS

使用方式:
    python -m src.mcp_servers.obs_control_server
"""

import asyncio
import json
import urllib.request
import urllib.error
import os
import sys

from mcp.server import Server
from mcp.types import Tool, TextContent

# 服务器名称
SERVER_NAME = "obs-control"
server = Server(SERVER_NAME)

# Flask后端地址（可通过环境变量配置）
BACKEND_URL = os.getenv("OBS_BACKEND_URL", "http://localhost:5000/api")


def call_api(endpoint, method="GET", data=None):
    """调用Flask后端API

    Args:
        endpoint: API端点
        method: HTTP方法
        data: 请求数据

    Returns:
        API响应结果，失败时返回包含error和message的字典
    """
    url = f"{BACKEND_URL}/{endpoint}"
    try:
        if data:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method=method
            )
        else:
            req = urllib.request.Request(url, method=method)

        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            # 检查响应中是否包含error字段
            if isinstance(result, dict) and "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "message": f"【操作失败】OBS返回错误: {result['error']}。请确保OBS正在运行且WebSocket服务已启用。"
                }
            return result
    except urllib.error.URLError as e:
        return {
            "success": False,
            "error": f"无法连接后端服务: {e.reason}",
            "message": f"【连接失败】无法连接到后端服务。请确保: 1) Flask后端已启动 2) OBS正在运行 3) WebSocket服务已启用"
        }
    except urllib.error.HTTPError as e:
        try:
            error_body = json.loads(e.read().decode('utf-8'))
            error_msg = error_body.get('error', str(e.reason))
        except:
            error_msg = str(e.reason)
        return {
            "success": False,
            "error": f"HTTP错误 {e.code}: {error_msg}",
            "message": f"【请求失败】HTTP {e.code}错误: {error_msg}。请检查OBS连接状态。"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"请求失败: {str(e)}",
            "message": f"【未知错误】{str(e)}"
        }


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
    if name == "get_scenes":
        result = call_api("scenes")
    elif name == "switch_scene":
        result = call_api("scenes/switch", "POST", {"scene": arguments["scene"]})
    elif name == "get_audio_sources":
        result = call_api("audio/sources")
    elif name == "set_volume":
        result = call_api("audio/volume", "POST", {
            "source": arguments["source"],
            "volume": arguments["volume"]
        })
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def main():
    """启动MCP Server"""
    from mcp.server.stdio import stdio_server

    print(f"[{SERVER_NAME}] MCP Server 启动中...", file=sys.stderr)
    print(f"[{SERVER_NAME}] 后端地址: {BACKEND_URL}", file=sys.stderr)

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
