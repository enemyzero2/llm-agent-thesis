# -*- coding: utf-8 -*-
"""
=============================================================================
OBS同声传译 MCP Server
文件: src/mcp_servers/obs_translation_server.py
说明: 通过 OBS WebSocket 控制 obs-localvocal 插件实现同声传译
     前提: 在 OBS 中安装 obs-localvocal 并在麦克风源上添加该滤镜
     插件地址: https://github.com/locaal-ai/obs-localvocal
=============================================================================
"""

import asyncio
import json
import os
import sys
from typing import Optional

from mcp.server import Server
from mcp.types import Tool, TextContent

try:
    from obswebsocket import obsws, requests as obs_requests
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    print("[WARNING] obs-websocket-py未安装", file=sys.stderr)

SERVER_NAME = "obs-translation"
server = Server(SERVER_NAME)

OBS_HOST     = os.getenv("OBS_HOST", "localhost")
OBS_PORT     = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "123456")

obs_client: Optional[obsws] = None

# localvocal 默认滤镜名（用户在 OBS 里添加滤镜时的名称）
DEFAULT_FILTER_NAME = "LocalVocal"

SUPPORTED_LANGUAGES = {
    "auto": "自动检测", "zh": "中文", "en": "英语",
    "ja": "日语", "ko": "韩语", "es": "西班牙语",
    "fr": "法语", "de": "德语", "ru": "俄语",
}


def connect_obs() -> bool:
    global obs_client
    if not OBS_AVAILABLE:
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


def _set_filter_enabled(source_name: str, filter_name: str, enabled: bool) -> dict:
    """启用或禁用 obs-localvocal 滤镜"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    try:
        obs_client.call(obs_requests.SetSourceFilterEnabled(
            sourceName=source_name,
            filterName=filter_name,
            filterEnabled=enabled
        ))
        status = "已开始" if enabled else "已停止"
        return {
            "success": True,
            "source":  source_name,
            "filter":  filter_name,
            "enabled": enabled,
            "message": f"同声传译{status}（滤镜: {filter_name}）"
        }
    except Exception as e:
        return {"error": f"操作失败: {str(e)}，请确认OBS中已安装obs-localvocal并添加了名为'{filter_name}'的滤镜"}


def _get_filter_status(source_name: str, filter_name: str) -> dict:
    """查询滤镜状态"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    try:
        resp = obs_client.call(obs_requests.GetSourceFilter(
            sourceName=source_name,
            filterName=filter_name
        ))
        return {
            "source":       source_name,
            "filter":       filter_name,
            "enabled":      resp.getFilterEnabled(),
            "obs_connected": True,
        }
    except Exception as e:
        return {"error": f"获取状态失败: {str(e)}"}


# ============= MCP 接口 =============

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="start_translation",
            description=(
                "开始同声传译。启用 OBS 中的 obs-localvocal 滤镜，"
                "该插件会自动实时录音→识别→翻译→显示为OBS字幕。"
                "【前提】需在OBS中安装obs-localvocal并在麦克风源上添加滤镜。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "麦克风/音频源名称（OBS中的源名）"
                    },
                    "filter_name": {
                        "type": "string",
                        "description": f"obs-localvocal滤镜名称（默认'{DEFAULT_FILTER_NAME}'）",
                        "default": DEFAULT_FILTER_NAME
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="stop_translation",
            description="停止同声传译。禁用 obs-localvocal 滤镜。",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "麦克风/音频源名称"
                    },
                    "filter_name": {
                        "type": "string",
                        "description": f"滤镜名称（默认'{DEFAULT_FILTER_NAME}'）",
                        "default": DEFAULT_FILTER_NAME
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="get_translation_status",
            description="查询同声传译（obs-localvocal滤镜）当前是否运行中",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "麦克风/音频源名称"
                    },
                    "filter_name": {
                        "type": "string",
                        "description": f"滤镜名称（默认'{DEFAULT_FILTER_NAME}'）",
                        "default": DEFAULT_FILTER_NAME
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="get_supported_languages",
            description="获取常用语言代码列表（供参考，实际语言在obs-localvocal内设置）",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    source_name = arguments.get("source_name", "")
    filter_name = arguments.get("filter_name", DEFAULT_FILTER_NAME)

    if name == "start_translation":
        result = _set_filter_enabled(source_name, filter_name, enabled=True)

    elif name == "stop_translation":
        result = _set_filter_enabled(source_name, filter_name, enabled=False)

    elif name == "get_translation_status":
        result = _get_filter_status(source_name, filter_name)

    elif name == "get_supported_languages":
        result = {
            "languages": SUPPORTED_LANGUAGES,
            "note": "语言在 OBS 的 obs-localvocal 滤镜属性页面内配置，此列表仅供参考"
        }

    else:
        result = {"error": f"未知工具: {name}"}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def main():
    from mcp.server.stdio import stdio_server

    print(f"[{SERVER_NAME}] MCP Server 启动中...", file=sys.stderr)
    print(f"[{SERVER_NAME}] 依赖: obs-localvocal 插件 (https://github.com/locaal-ai/obs-localvocal)", file=sys.stderr)
    connect_obs()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
