# -*- coding: utf-8 -*-
"""
=============================================================================
OBS背景虚化/虚拟背景 MCP Server
文件: src/mcp_servers/obs_background_server.py
说明: 提供背景虚化和虚拟背景功能，作为独立的能力引擎
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
    print("[WARNING] obs-websocket-py未安装", file=sys.stderr)

# 创建Server实例
SERVER_NAME = "obs-background"
server = Server(SERVER_NAME)

# OBS连接配置
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "123456")

# OBS WebSocket客户端
obs_client: Optional[obsws] = None

# 背景处理状态
background_state = {
    "blur_enabled": False,
    "blur_intensity": 10,
    "virtual_bg_enabled": False,
    "virtual_bg_path": None,
    "target_source": None,  # 要处理的视频源
}

# 预设的虚拟背景
PRESET_BACKGROUNDS = {
    "office": "办公室背景",
    "library": "图书馆背景",
    "blur": "模糊背景",
    "gradient": "渐变色背景",
    "custom": "自定义图片"
}

# 滤镜类型
BLUR_FILTER_KINDS = [
    "streamfx-filter-blur",       # StreamFX (推荐)
    "obs-shaderfilter",           # 着色器滤镜
]


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


def get_video_sources() -> dict:
    """获取所有视频源"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    try:
        response = obs_client.call(obs_requests.GetInputList())
        video_sources = []
        
        for input_item in response.getInputs():
            input_kind = input_item.get('inputKind', '')
            # 视频捕获源
            if any(x in input_kind.lower() for x in ['video', 'capture', 'camera', 'dshow', 'v4l2']):
                video_sources.append({
                    "name": input_item['inputName'],
                    "kind": input_kind
                })
        
        return {
            "sources": video_sources,
            "count": len(video_sources),
            "message": f"找到 {len(video_sources)} 个视频源"
        }
    except Exception as e:
        return {"error": f"获取视频源失败: {str(e)}"}


def apply_blur_filter(source_name: str, intensity: int = 10) -> dict:
    """应用模糊滤镜实现背景虚化
    
    注意：完整的背景虚化需要配合人像分割，这里提供基础的模糊效果
    """
    if obs_client is None:
        return {"error": "未连接到OBS"}

    filter_name = f"{source_name}_背景虚化"
    
    try:
        # 尝试使用不同的模糊滤镜
        # 方法1: 使用OBS内置的色彩校正滤镜模拟效果
        # 方法2: 使用StreamFX的模糊滤镜 (如果安装了)
        
        # 首先尝试创建滤镜
        filter_settings = {
            "Filter.Blur.Size": float(intensity),
            "Filter.Blur.Type": 1  # Gaussian
        }
        
        # 尝试StreamFX滤镜
        try:
            obs_client.call(obs_requests.CreateSourceFilter(
                sourceName=source_name,
                filterName=filter_name,
                filterKind="streamfx-filter-blur",
                filterSettings=filter_settings
            ))
            background_state["blur_enabled"] = True
            background_state["blur_intensity"] = intensity
            background_state["target_source"] = source_name
            
            return {
                "success": True,
                "source": source_name,
                "filter": filter_name,
                "intensity": intensity,
                "message": f"已为'{source_name}'应用背景虚化效果(强度:{intensity})"
            }
        except Exception:
            # StreamFX不可用，使用替代方案提示
            return {
                "warning": "StreamFX插件未安装",
                "suggestion": "请安装StreamFX插件以获得更好的模糊效果",
                "download": "https://github.com/Xaymar/obs-StreamFX/releases",
                "alternative": "可以使用obs_filter服务的color_correction滤镜调整画面"
            }
            
    except Exception as e:
        return {"error": f"应用模糊滤镜失败: {str(e)}"}


def remove_blur_filter(source_name: str) -> dict:
    """移除背景虚化滤镜"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    filter_name = f"{source_name}_背景虚化"
    
    try:
        obs_client.call(obs_requests.RemoveSourceFilter(
            sourceName=source_name,
            filterName=filter_name
        ))
        background_state["blur_enabled"] = False
        
        return {
            "success": True,
            "source": source_name,
            "message": f"已移除'{source_name}'的背景虚化效果"
        }
    except Exception as e:
        return {"error": f"移除滤镜失败: {str(e)}"}


def set_virtual_background(source_name: str, scene_name: str, 
                           background_path: str = None,
                           background_color: str = None) -> dict:
    """设置虚拟背景
    
    实现方式：
    1. 在源下方添加图片/颜色源作为背景
    2. 对视频源应用色度键滤镜(绿幕)
    """
    if obs_client is None:
        return {"error": "未连接到OBS"}

    bg_source_name = f"{source_name}_虚拟背景"
    
    try:
        # 创建背景源
        if background_path:
            # 图片背景
            obs_client.call(obs_requests.CreateInput(
                sceneName=scene_name,
                inputName=bg_source_name,
                inputKind="image_source",
                inputSettings={"file": background_path}
            ))
        elif background_color:
            # 颜色背景
            if background_color.startswith("#"):
                background_color = background_color[1:]
            color_int = int(background_color, 16) | 0xFF000000
            
            obs_client.call(obs_requests.CreateInput(
                sceneName=scene_name,
                inputName=bg_source_name,
                inputKind="color_source_v3",
                inputSettings={"color": color_int}
            ))
        
        # 调整层级，确保背景在视频源下方
        # 获取场景项ID
        try:
            bg_item = obs_client.call(obs_requests.GetSceneItemId(
                sceneName=scene_name,
                sourceName=bg_source_name
            ))
            video_item = obs_client.call(obs_requests.GetSceneItemId(
                sceneName=scene_name,
                sourceName=source_name
            ))
            
            # 将背景移到最底层
            obs_client.call(obs_requests.SetSceneItemIndex(
                sceneName=scene_name,
                sceneItemId=bg_item.getSceneItemId(),
                sceneItemIndex=0
            ))
        except:
            pass  # 层级调整失败也继续
        
        background_state["virtual_bg_enabled"] = True
        background_state["virtual_bg_path"] = background_path
        background_state["target_source"] = source_name
        
        return {
            "success": True,
            "source": source_name,
            "background_source": bg_source_name,
            "message": f"已为'{source_name}'设置虚拟背景"
        }
    except Exception as e:
        return {"error": f"设置虚拟背景失败: {str(e)}"}


def apply_chroma_key(source_name: str, key_color: str = "green",
                     similarity: int = 400, smoothness: int = 80) -> dict:
    """应用色度键滤镜(绿幕抠像)"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    filter_name = f"{source_name}_色度键"
    
    try:
        obs_client.call(obs_requests.CreateSourceFilter(
            sourceName=source_name,
            filterName=filter_name,
            filterKind="chroma_key_filter_v2",
            filterSettings={
                "key_color_type": key_color,
                "similarity": similarity,
                "smoothness": smoothness,
                "spill": 100
            }
        ))
        
        return {
            "success": True,
            "source": source_name,
            "filter": filter_name,
            "key_color": key_color,
            "message": f"已为'{source_name}'应用色度键滤镜"
        }
    except Exception as e:
        return {"error": f"应用色度键失败: {str(e)}"}


def remove_virtual_background(source_name: str, scene_name: str) -> dict:
    """移除虚拟背景"""
    if obs_client is None:
        return {"error": "未连接到OBS"}

    bg_source_name = f"{source_name}_虚拟背景"
    filter_name = f"{source_name}_色度键"
    
    errors = []
    
    # 移除背景源
    try:
        # 获取场景项ID并删除
        item = obs_client.call(obs_requests.GetSceneItemId(
            sceneName=scene_name,
            sourceName=bg_source_name
        ))
        obs_client.call(obs_requests.RemoveSceneItem(
            sceneName=scene_name,
            sceneItemId=item.getSceneItemId()
        ))
    except Exception as e:
        errors.append(f"移除背景源失败: {e}")
    
    # 移除色度键滤镜
    try:
        obs_client.call(obs_requests.RemoveSourceFilter(
            sourceName=source_name,
            filterName=filter_name
        ))
    except Exception as e:
        errors.append(f"移除滤镜失败: {e}")
    
    background_state["virtual_bg_enabled"] = False
    background_state["virtual_bg_path"] = None
    
    if errors:
        return {
            "partial_success": True,
            "errors": errors,
            "message": "虚拟背景部分移除"
        }
    
    return {
        "success": True,
        "source": source_name,
        "message": f"已移除'{source_name}'的虚拟背景"
    }


def get_background_status() -> dict:
    """获取当前背景处理状态"""
    return {
        "blur_enabled": background_state["blur_enabled"],
        "blur_intensity": background_state["blur_intensity"],
        "virtual_bg_enabled": background_state["virtual_bg_enabled"],
        "virtual_bg_path": background_state["virtual_bg_path"],
        "target_source": background_state["target_source"],
        "obs_connected": obs_client is not None
    }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的背景处理工具"""
    return [
        Tool(
            name="get_video_sources",
            description="获取OBS中所有视频源（摄像头等）",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="apply_background_blur",
            description="对视频源应用背景虚化效果",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "视频源名称"
                    },
                    "intensity": {
                        "type": "integer",
                        "description": "虚化强度(1-50，默认10)",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="remove_background_blur",
            description="移除视频源的背景虚化效果",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "视频源名称"
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="set_virtual_background",
            description="设置虚拟背景（图片或纯色）",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "视频源名称"
                    },
                    "scene_name": {
                        "type": "string",
                        "description": "场景名称"
                    },
                    "background_path": {
                        "type": "string",
                        "description": "背景图片路径(可选)"
                    },
                    "background_color": {
                        "type": "string",
                        "description": "背景颜色，十六进制格式如#336699(可选)"
                    }
                },
                "required": ["source_name", "scene_name"]
            }
        ),
        Tool(
            name="apply_chroma_key",
            description="应用色度键滤镜(绿幕抠像)",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "视频源名称"
                    },
                    "key_color": {
                        "type": "string",
                        "description": "抠像颜色",
                        "enum": ["green", "blue", "magenta", "custom"],
                        "default": "green"
                    },
                    "similarity": {
                        "type": "integer",
                        "description": "相似度(1-1000，默认400)",
                        "default": 400
                    },
                    "smoothness": {
                        "type": "integer",
                        "description": "平滑度(1-1000，默认80)",
                        "default": 80
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="remove_virtual_background",
            description="移除虚拟背景和色度键滤镜",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "视频源名称"
                    },
                    "scene_name": {
                        "type": "string",
                        "description": "场景名称"
                    }
                },
                "required": ["source_name", "scene_name"]
            }
        ),
        Tool(
            name="get_background_status",
            description="获取当前背景处理状态",
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
    
    if name == "get_video_sources":
        result = get_video_sources()
    
    elif name == "apply_background_blur":
        result = apply_blur_filter(
            arguments.get("source_name"),
            arguments.get("intensity", 10)
        )
    
    elif name == "remove_background_blur":
        result = remove_blur_filter(arguments.get("source_name"))
    
    elif name == "set_virtual_background":
        result = set_virtual_background(
            arguments.get("source_name"),
            arguments.get("scene_name"),
            background_path=arguments.get("background_path"),
            background_color=arguments.get("background_color")
        )
    
    elif name == "apply_chroma_key":
        result = apply_chroma_key(
            arguments.get("source_name"),
            key_color=arguments.get("key_color", "green"),
            similarity=arguments.get("similarity", 400),
            smoothness=arguments.get("smoothness", 80)
        )
    
    elif name == "remove_virtual_background":
        result = remove_virtual_background(
            arguments.get("source_name"),
            arguments.get("scene_name")
        )
    
    elif name == "get_background_status":
        result = get_background_status()
    
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
