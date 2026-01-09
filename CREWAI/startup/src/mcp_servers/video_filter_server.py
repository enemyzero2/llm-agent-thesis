"""
=============================================================================
视频滤镜 MCP Server
文件: src/mcp_servers/video_filter_server.py
说明: 提供视频滤镜效果（马赛克、虚化、颜色反转等）
=============================================================================
"""

import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent


# 创建Server实例
server = Server("video-filter")

# 当前滤镜状态
active_filters = {
    "mosaic": {"enabled": False, "intensity": 10},
    "blur": {"enabled": False, "radius": 5},
    "invert": {"enabled": False},
    "grayscale": {"enabled": False}
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的视频滤镜工具"""
    return [
        Tool(
            name="get_filters",
            description="获取当前所有滤镜状态",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="apply_mosaic",
            description="应用马赛克效果",
            inputSchema={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "是否启用"
                    },
                    "intensity": {
                        "type": "integer",
                        "description": "马赛克强度(1-50)",
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": ["enabled"]
            }
        ),
        Tool(
            name="apply_blur",
            description="应用虚化/模糊效果",
            inputSchema={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "是否启用"
                    },
                    "radius": {
                        "type": "integer",
                        "description": "模糊半径(1-30)",
                        "minimum": 1,
                        "maximum": 30
                    }
                },
                "required": ["enabled"]
            }
        ),
        Tool(
            name="apply_invert",
            description="应用颜色反转效果",
            inputSchema={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "是否启用"
                    }
                },
                "required": ["enabled"]
            }
        ),
        Tool(
            name="apply_grayscale",
            description="应用灰度效果",
            inputSchema={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "是否启用"
                    }
                },
                "required": ["enabled"]
            }
        ),
        Tool(
            name="clear_all_filters",
            description="清除所有滤镜效果",
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
    global active_filters

    if name == "get_filters":
        result = {"filters": active_filters}

    elif name == "apply_mosaic":
        enabled = arguments.get("enabled", False)
        intensity = arguments.get("intensity", 10)
        active_filters["mosaic"] = {"enabled": enabled, "intensity": intensity}
        status = "已启用" if enabled else "已关闭"
        result = {"mosaic": active_filters["mosaic"], "message": f"马赛克{status}"}

    elif name == "apply_blur":
        enabled = arguments.get("enabled", False)
        radius = arguments.get("radius", 5)
        active_filters["blur"] = {"enabled": enabled, "radius": radius}
        status = "已启用" if enabled else "已关闭"
        result = {"blur": active_filters["blur"], "message": f"虚化效果{status}"}

    elif name == "apply_invert":
        enabled = arguments.get("enabled", False)
        active_filters["invert"] = {"enabled": enabled}
        status = "已启用" if enabled else "已关闭"
        result = {"invert": active_filters["invert"], "message": f"颜色反转{status}"}

    elif name == "apply_grayscale":
        enabled = arguments.get("enabled", False)
        active_filters["grayscale"] = {"enabled": enabled}
        status = "已启用" if enabled else "已关闭"
        result = {"grayscale": active_filters["grayscale"], "message": f"灰度效果{status}"}

    elif name == "clear_all_filters":
        active_filters = {
            "mosaic": {"enabled": False, "intensity": 10},
            "blur": {"enabled": False, "radius": 5},
            "invert": {"enabled": False},
            "grayscale": {"enabled": False}
        }
        result = {"filters": active_filters, "message": "已清除所有滤镜"}

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
