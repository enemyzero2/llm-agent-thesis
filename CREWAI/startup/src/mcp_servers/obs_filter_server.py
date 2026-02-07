# -*- coding: utf-8 -*-
"""
=============================================================================
OBS滤镜控制 MCP Server
文件: src/mcp_servers/obs_filter_server.py
说明: 通过OBS WebSocket控制真实OBS滤镜
=============================================================================
"""

import asyncio
import json
import os
import sys
from typing import Optional, Dict, Any

from mcp.server import Server
from mcp.types import Tool, TextContent

# 尝试导入obs-websocket-py
try:
    from obswebsocket import obsws, requests as obs_requests
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    print("[WARNING] obs-websocket-py未安装，将使用模拟模式", file=sys.stderr)

# 创建Server实例
SERVER_NAME = "obs-filter"
server = Server(SERVER_NAME)

# OBS连接配置
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "123456")

# OBS WebSocket客户端
obs_client: Optional[obsws] = None

# 支持的滤镜类型及其常用参数
FILTER_TYPES = {
    "color_correction": {
        "kind": "color_filter_v2",
        "description": "色彩校正滤镜",
        "default_settings": {
            "brightness": 0.0,    # -1.0 到 1.0
            "contrast": 0.0,      # -4.0 到 4.0
            "saturation": 0.0,    # -1.0 到 5.0
            "gamma": 0.0,         # -3.0 到 3.0
            "hue_shift": 0.0      # -180 到 180
        }
    },
    "blur": {
        "kind": "streamfx-filter-blur",  # StreamFX插件的模糊滤镜
        "description": "模糊/虚化效果",
        "default_settings": {
            "Filter.Blur.Size": 10.0,
            "Filter.Blur.Type": 0  # 0=Box, 1=Gaussian, 2=Dual Filtering
        }
    },
    "sharpen": {
        "kind": "sharpness_filter",
        "description": "锐化滤镜",
        "default_settings": {
            "sharpness": 0.08  # 0.0 到 1.0
        }
    },
    "chroma_key": {
        "kind": "chroma_key_filter_v2",
        "description": "色度键(绿幕)滤镜",
        "default_settings": {
            "key_color_type": "green",
            "similarity": 400,
            "smoothness": 80,
            "spill": 100
        }
    },
    "lut": {
        "kind": "clut_filter",
        "description": "3D LUT颜色分级",
        "default_settings": {
            "clut_amount": 1.0,
            "clut": ""  # LUT文件路径
        }
    },
    "mask": {
        "kind": "mask_filter_v2",
        "description": "图像遮罩滤镜",
        "default_settings": {
            "type": "mask_alpha_filter.effect",
            "image_path": ""
        }
    },
    "scroll": {
        "kind": "scroll_filter",
        "description": "滚动效果滤镜",
        "default_settings": {
            "speed_x": 0.0,
            "speed_y": 50.0,
            "loop": True
        }
    },
    "render_delay": {
        "kind": "gpu_delay",
        "description": "渲染延迟滤镜",
        "default_settings": {
            "delay_ms": 0
        }
    }
}


def connect_obs() -> bool:
    """连接到OBS WebSocket"""
    global obs_client

    if not OBS_AVAILABLE:
        print(f"[{SERVER_NAME}] obs-websocket-py未安装，使用模拟模式", file=sys.stderr)
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


def list_source_filters(source_name: str) -> dict:
    """获取源的所有滤镜"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    try:
        response = obs_client.call(obs_requests.GetSourceFilterList(
            sourceName=source_name
        ))
        filters = response.getFilters()
        
        return {
            "source": source_name,
            "filters": filters,
            "count": len(filters),
            "message": f"源'{source_name}'有{len(filters)}个滤镜"
        }
    except Exception as e:
        return {"error": f"获取滤镜列表失败: {str(e)}"}


def add_filter(source_name: str, filter_name: str, filter_type: str, 
               settings: Dict[str, Any] = None) -> dict:
    """添加滤镜到源"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    # 获取滤镜类型配置
    if filter_type not in FILTER_TYPES:
        return {
            "error": f"不支持的滤镜类型: {filter_type}",
            "supported_types": list(FILTER_TYPES.keys())
        }
    
    filter_config = FILTER_TYPES[filter_type]
    filter_kind = filter_config["kind"]
    
    # 合并默认设置和用户设置
    final_settings = filter_config["default_settings"].copy()
    if settings:
        final_settings.update(settings)

    try:
        obs_client.call(obs_requests.CreateSourceFilter(
            sourceName=source_name,
            filterName=filter_name,
            filterKind=filter_kind,
            filterSettings=final_settings
        ))
        
        return {
            "success": True,
            "source": source_name,
            "filter_name": filter_name,
            "filter_type": filter_type,
            "settings": final_settings,
            "message": f"已添加滤镜'{filter_name}'到源'{source_name}'"
        }
    except Exception as e:
        return {"error": f"添加滤镜失败: {str(e)}"}


def remove_filter(source_name: str, filter_name: str) -> dict:
    """删除源的滤镜"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    try:
        obs_client.call(obs_requests.RemoveSourceFilter(
            sourceName=source_name,
            filterName=filter_name
        ))
        
        return {
            "success": True,
            "source": source_name,
            "filter_name": filter_name,
            "message": f"已删除滤镜'{filter_name}'"
        }
    except Exception as e:
        return {"error": f"删除滤镜失败: {str(e)}"}


def get_filter_settings(source_name: str, filter_name: str) -> dict:
    """获取滤镜当前设置"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    try:
        response = obs_client.call(obs_requests.GetSourceFilter(
            sourceName=source_name,
            filterName=filter_name
        ))
        
        return {
            "source": source_name,
            "filter_name": filter_name,
            "filter_kind": response.getFilterKind(),
            "filter_enabled": response.getFilterEnabled(),
            "settings": response.getFilterSettings(),
            "message": f"滤镜'{filter_name}'的设置"
        }
    except Exception as e:
        return {"error": f"获取滤镜设置失败: {str(e)}"}


def set_filter_settings(source_name: str, filter_name: str, 
                        settings: Dict[str, Any]) -> dict:
    """修改滤镜参数"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    try:
        obs_client.call(obs_requests.SetSourceFilterSettings(
            sourceName=source_name,
            filterName=filter_name,
            filterSettings=settings
        ))
        
        return {
            "success": True,
            "source": source_name,
            "filter_name": filter_name,
            "settings": settings,
            "message": f"滤镜'{filter_name}'设置已更新"
        }
    except Exception as e:
        return {"error": f"设置滤镜参数失败: {str(e)}"}


def toggle_filter(source_name: str, filter_name: str, enabled: bool) -> dict:
    """启用/禁用滤镜"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    try:
        obs_client.call(obs_requests.SetSourceFilterEnabled(
            sourceName=source_name,
            filterName=filter_name,
            filterEnabled=enabled
        ))
        
        status = "启用" if enabled else "禁用"
        return {
            "success": True,
            "source": source_name,
            "filter_name": filter_name,
            "enabled": enabled,
            "message": f"滤镜'{filter_name}'已{status}"
        }
    except Exception as e:
        return {"error": f"切换滤镜状态失败: {str(e)}"}


def set_filter_index(source_name: str, filter_name: str, index: int) -> dict:
    """调整滤镜顺序"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    try:
        obs_client.call(obs_requests.SetSourceFilterIndex(
            sourceName=source_name,
            filterName=filter_name,
            filterIndex=index
        ))
        
        return {
            "success": True,
            "source": source_name,
            "filter_name": filter_name,
            "index": index,
            "message": f"滤镜'{filter_name}'位置已调整到{index}"
        }
    except Exception as e:
        return {"error": f"调整滤镜顺序失败: {str(e)}"}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的滤镜控制工具"""
    return [
        Tool(
            name="list_filters",
            description="获取指定源的所有滤镜列表",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "源名称"
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="add_filter",
            description=f"添加滤镜到源。支持的滤镜类型: {', '.join(FILTER_TYPES.keys())}",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "源名称"
                    },
                    "filter_name": {
                        "type": "string",
                        "description": "滤镜名称(自定义)"
                    },
                    "filter_type": {
                        "type": "string",
                        "description": "滤镜类型",
                        "enum": list(FILTER_TYPES.keys())
                    },
                    "settings": {
                        "type": "object",
                        "description": "滤镜参数(可选)"
                    }
                },
                "required": ["source_name", "filter_name", "filter_type"]
            }
        ),
        Tool(
            name="remove_filter",
            description="删除指定源的滤镜",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "源名称"
                    },
                    "filter_name": {
                        "type": "string",
                        "description": "滤镜名称"
                    }
                },
                "required": ["source_name", "filter_name"]
            }
        ),
        Tool(
            name="get_filter_settings",
            description="获取滤镜的当前设置",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "源名称"
                    },
                    "filter_name": {
                        "type": "string",
                        "description": "滤镜名称"
                    }
                },
                "required": ["source_name", "filter_name"]
            }
        ),
        Tool(
            name="set_filter_settings",
            description="修改滤镜参数",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "源名称"
                    },
                    "filter_name": {
                        "type": "string",
                        "description": "滤镜名称"
                    },
                    "settings": {
                        "type": "object",
                        "description": "新的滤镜参数"
                    }
                },
                "required": ["source_name", "filter_name", "settings"]
            }
        ),
        Tool(
            name="toggle_filter",
            description="启用或禁用滤镜",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "源名称"
                    },
                    "filter_name": {
                        "type": "string",
                        "description": "滤镜名称"
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "是否启用"
                    }
                },
                "required": ["source_name", "filter_name", "enabled"]
            }
        ),
        Tool(
            name="get_supported_filters",
            description="获取支持的滤镜类型列表及其说明",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="apply_color_correction",
            description="快速应用色彩校正滤镜(亮度/对比度/饱和度)",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "源名称"
                    },
                    "brightness": {
                        "type": "number",
                        "description": "亮度(-1.0到1.0，0为默认)",
                        "minimum": -1.0,
                        "maximum": 1.0
                    },
                    "contrast": {
                        "type": "number",
                        "description": "对比度(-4.0到4.0，0为默认)",
                        "minimum": -4.0,
                        "maximum": 4.0
                    },
                    "saturation": {
                        "type": "number",
                        "description": "饱和度(-1.0到5.0，0为默认)",
                        "minimum": -1.0,
                        "maximum": 5.0
                    },
                    "gamma": {
                        "type": "number",
                        "description": "伽马(-3.0到3.0，0为默认)",
                        "minimum": -3.0,
                        "maximum": 3.0
                    },
                    "hue_shift": {
                        "type": "number",
                        "description": "色相偏移(-180到180度，0为默认)",
                        "minimum": -180,
                        "maximum": 180
                    }
                },
                "required": ["source_name"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    
    if name == "list_filters":
        result = list_source_filters(arguments.get("source_name"))
    
    elif name == "add_filter":
        result = add_filter(
            arguments.get("source_name"),
            arguments.get("filter_name"),
            arguments.get("filter_type"),
            arguments.get("settings")
        )
    
    elif name == "remove_filter":
        result = remove_filter(
            arguments.get("source_name"),
            arguments.get("filter_name")
        )
    
    elif name == "get_filter_settings":
        result = get_filter_settings(
            arguments.get("source_name"),
            arguments.get("filter_name")
        )
    
    elif name == "set_filter_settings":
        result = set_filter_settings(
            arguments.get("source_name"),
            arguments.get("filter_name"),
            arguments.get("settings", {})
        )
    
    elif name == "toggle_filter":
        result = toggle_filter(
            arguments.get("source_name"),
            arguments.get("filter_name"),
            arguments.get("enabled", True)
        )
    
    elif name == "get_supported_filters":
        result = {
            "supported_filters": {
                k: {"kind": v["kind"], "description": v["description"]}
                for k, v in FILTER_TYPES.items()
            },
            "message": f"支持{len(FILTER_TYPES)}种滤镜类型"
        }
    
    elif name == "apply_color_correction":
        source_name = arguments.get("source_name")
        filter_name = f"{source_name}_color_correction"
        
        settings = {}
        if "brightness" in arguments:
            settings["brightness"] = arguments["brightness"]
        if "contrast" in arguments:
            settings["contrast"] = arguments["contrast"]
        if "saturation" in arguments:
            settings["saturation"] = arguments["saturation"]
        if "gamma" in arguments:
            settings["gamma"] = arguments["gamma"]
        if "hue_shift" in arguments:
            settings["hue_shift"] = arguments["hue_shift"]
        
        # 尝试更新现有滤镜，如果不存在则创建
        existing = get_filter_settings(source_name, filter_name)
        if "error" in existing:
            result = add_filter(source_name, filter_name, "color_correction", settings)
        else:
            result = set_filter_settings(source_name, filter_name, settings)
    
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
