"""
=============================================================================
亮度调节 MCP Server
文件: src/mcp_servers/brightness_server.py
说明: 提供视频流的亮度、对比度调节功能
=============================================================================
"""

import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent


# 创建Server实例
server = Server("brightness-control")

# 当前状态
current_brightness = 50  # 0-100
current_contrast = 50    # 0-100


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的亮度控制工具"""
    return [
        Tool(
            name="get_brightness",
            description="获取当前亮度值",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="set_brightness",
            description="设置亮度到指定值(0-100)",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "亮度值(0-100)",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["level"]
            }
        ),
        Tool(
            name="get_contrast",
            description="获取当前对比度值",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="set_contrast",
            description="设置对比度到指定值(0-100)",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "对比度值(0-100)",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["level"]
            }
        ),
        Tool(
            name="reset_display",
            description="重置亮度和对比度到默认值",
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
    global current_brightness, current_contrast

    if name == "get_brightness":
        result = {"brightness": current_brightness}

    elif name == "set_brightness":
        level = arguments.get("level", 50)
        level = max(0, min(100, level))
        current_brightness = level
        result = {"brightness": current_brightness, "message": f"亮度已设置为 {level}"}

    elif name == "get_contrast":
        result = {"contrast": current_contrast}

    elif name == "set_contrast":
        level = arguments.get("level", 50)
        level = max(0, min(100, level))
        current_contrast = level
        result = {"contrast": current_contrast, "message": f"对比度已设置为 {level}"}

    elif name == "reset_display":
        current_brightness = 50
        current_contrast = 50
        result = {
            "brightness": 50,
            "contrast": 50,
            "message": "已重置为默认值"
        }

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
