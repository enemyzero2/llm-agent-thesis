# -*- coding: utf-8 -*-
"""
OBS控制 MCP Server
通过HTTP API与Flask后端通信，控制OBS
"""

import asyncio
import json
import urllib.request
import urllib.error
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("obs-control")

# Flask后端地址
BACKEND_URL = "http://localhost:5000/api"


def call_api(endpoint, method="GET", data=None):
    """调用Flask后端API"""
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

        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="get_scenes",
            description="获取OBS所有场景列表",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="switch_scene",
            description="切换到指定场景",
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
            description="获取所有音频源及其音量",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="set_volume",
            description="设置音频源音量(0-100)",
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
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
