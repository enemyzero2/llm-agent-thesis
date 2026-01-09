"""
=============================================================================
音量控制 MCP Server
文件: src/mcp_servers/volume_server.py
说明: 提供音频流的音量控制功能
=============================================================================
"""

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent


# 创建Server实例
server = Server("volume-control")

# 模拟当前音量状态
current_volume = 50  # 0-100


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的音量控制工具"""
    return [
        Tool(
            name="get_volume",
            description="获取当前音量值",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="set_volume",
            description="设置音量到指定值(0-100)",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "音量值(0-100)",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["level"]
            }
        ),
        Tool(
            name="adjust_volume",
            description="调整音量(增加或减少)",
            inputSchema={
                "type": "object",
                "properties": {
                    "delta": {
                        "type": "integer",
                        "description": "调整量,正数增加,负数减少"
                    }
                },
                "required": ["delta"]
            }
        ),
        Tool(
            name="mute",
            description="静音",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    global current_volume

    if name == "get_volume":
        result = {"volume": current_volume}

    elif name == "set_volume":
        level = arguments.get("level", 50)
        level = max(0, min(100, level))  # 限制范围
        current_volume = level
        result = {"volume": current_volume, "message": f"音量已设置为 {level}"}

    elif name == "adjust_volume":
        delta = arguments.get("delta", 0)
        new_volume = max(0, min(100, current_volume + delta))
        current_volume = new_volume
        action = "增加" if delta > 0 else "减少"
        result = {"volume": current_volume, "message": f"音量已{action}到 {current_volume}"}

    elif name == "mute":
        current_volume = 0
        result = {"volume": 0, "message": "已静音"}

    else:
        result = {"error": f"未知工具: {name}"}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def main():
    """运行Server"""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
