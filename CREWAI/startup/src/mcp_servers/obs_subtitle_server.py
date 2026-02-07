# -*- coding: utf-8 -*-
"""
=============================================================================
OBS字幕控制 MCP Server
文件: src/mcp_servers/obs_subtitle_server.py
说明: 通过OBS WebSocket控制字幕文本源
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
    print("[WARNING] obs-websocket-py未安装，将使用模拟模式", file=sys.stderr)

# 创建Server实例
SERVER_NAME = "obs-subtitle"
server = Server(SERVER_NAME)

# OBS连接配置
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "123456")

# OBS WebSocket客户端
obs_client: Optional[obsws] = None


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


def create_text_source(scene_name: str, source_name: str, text: str = "", 
                       font_size: int = 48, color: str = "#FFFFFF") -> dict:
    """在场景中创建文本源作为字幕"""
    if obs_client is None:
        return {"error": "未连接到OBS", "simulated": True}

    try:
        # 将颜色从十六进制转换为ARGB整数 (OBS使用ABGR格式)
        if color.startswith("#"):
            color = color[1:]
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        # OBS使用ABGR格式，Alpha为255
        color_int = (255 << 24) | (b << 16) | (g << 8) | r

        # 创建文本源设置
        input_settings = {
            "text": text,
            "font": {
                "face": "Microsoft YaHei",
                "size": font_size,
                "style": "Regular"
            },
            "color": color_int,
            "outline": True,
            "outline_size": 2,
            "outline_color": 0xFF000000  # 黑色描边
        }

        # 创建输入源
        obs_client.call(obs_requests.CreateInput(
            sceneName=scene_name,
            inputName=source_name,
            inputKind="text_gdiplus_v3",
            inputSettings=input_settings
        ))

        return {
            "success": True,
            "source_name": source_name,
            "scene": scene_name,
            "message": f"已在场景'{scene_name}'中创建字幕源'{source_name}'"
        }
    except Exception as e:
        return {"error": f"创建字幕源失败: {str(e)}"}


def update_subtitle_text(source_name: str, text: str) -> dict:
    """更新字幕文本内容"""
    if obs_client is None:
        return {"error": "未连接到OBS", "simulated": True}

    try:
        obs_client.call(obs_requests.SetInputSettings(
            inputName=source_name,
            inputSettings={"text": text}
        ))
        return {
            "success": True,
            "source_name": source_name,
            "text": text,
            "message": f"已更新字幕内容"
        }
    except Exception as e:
        return {"error": f"更新字幕失败: {str(e)}"}


def set_subtitle_style(source_name: str, font: str = None, size: int = None,
                       color: str = None, outline: bool = None, 
                       outline_size: int = None) -> dict:
    """设置字幕样式"""
    if obs_client is None:
        return {"error": "未连接到OBS", "simulated": True}

    try:
        settings = {}
        
        if font or size:
            font_settings = {"face": font or "Microsoft YaHei"}
            if size:
                font_settings["size"] = size
            settings["font"] = font_settings
        
        if color:
            if color.startswith("#"):
                color = color[1:]
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            settings["color"] = (255 << 24) | (b << 16) | (g << 8) | r
        
        if outline is not None:
            settings["outline"] = outline
        
        if outline_size is not None:
            settings["outline_size"] = outline_size

        obs_client.call(obs_requests.SetInputSettings(
            inputName=source_name,
            inputSettings=settings
        ))
        
        return {
            "success": True,
            "source_name": source_name,
            "message": "字幕样式已更新"
        }
    except Exception as e:
        return {"error": f"设置样式失败: {str(e)}"}


def set_subtitle_visibility(source_name: str, scene_name: str, visible: bool) -> dict:
    """设置字幕可见性"""
    if obs_client is None:
        return {"error": "未连接到OBS", "simulated": True}

    try:
        # 获取场景项ID
        response = obs_client.call(obs_requests.GetSceneItemId(
            sceneName=scene_name,
            sourceName=source_name
        ))
        item_id = response.getSceneItemId()
        
        # 设置可见性
        obs_client.call(obs_requests.SetSceneItemEnabled(
            sceneName=scene_name,
            sceneItemId=item_id,
            sceneItemEnabled=visible
        ))
        
        status = "显示" if visible else "隐藏"
        return {
            "success": True,
            "source_name": source_name,
            "visible": visible,
            "message": f"字幕已{status}"
        }
    except Exception as e:
        return {"error": f"设置可见性失败: {str(e)}"}


def set_subtitle_position(source_name: str, scene_name: str, 
                          x: int = None, y: int = None, 
                          alignment: int = 5) -> dict:
    """设置字幕位置
    
    alignment: 对齐方式 (数字键盘布局)
        7=左上, 8=中上, 9=右上
        4=左中, 5=居中, 6=右中  
        1=左下, 2=中下, 3=右下
    """
    if obs_client is None:
        return {"error": "未连接到OBS", "simulated": True}

    try:
        # 获取场景项ID
        response = obs_client.call(obs_requests.GetSceneItemId(
            sceneName=scene_name,
            sourceName=source_name
        ))
        item_id = response.getSceneItemId()
        
        # 获取当前变换设置
        transform_response = obs_client.call(obs_requests.GetSceneItemTransform(
            sceneName=scene_name,
            sceneItemId=item_id
        ))
        current = transform_response.getSceneItemTransform()
        
        # 更新位置
        new_transform = {}
        if x is not None:
            new_transform["positionX"] = x
        if y is not None:
            new_transform["positionY"] = y
        new_transform["alignment"] = alignment
        
        obs_client.call(obs_requests.SetSceneItemTransform(
            sceneName=scene_name,
            sceneItemId=item_id,
            sceneItemTransform=new_transform
        ))
        
        return {
            "success": True,
            "source_name": source_name,
            "position": {"x": x, "y": y, "alignment": alignment},
            "message": "字幕位置已更新"
        }
    except Exception as e:
        return {"error": f"设置位置失败: {str(e)}"}


def get_current_scene() -> str:
    """获取当前场景名称"""
    if obs_client is None:
        return None
    try:
        response = obs_client.call(obs_requests.GetCurrentProgramScene())
        return response.getCurrentProgramSceneName()
    except:
        return None


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的字幕控制工具"""
    return [
        Tool(
            name="create_subtitle",
            description="在OBS场景中创建字幕文本源。如果不指定场景，则使用当前活动场景。",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "字幕源名称"
                    },
                    "text": {
                        "type": "string",
                        "description": "初始字幕文本",
                        "default": ""
                    },
                    "scene_name": {
                        "type": "string",
                        "description": "场景名称(可选，默认当前场景)"
                    },
                    "font_size": {
                        "type": "integer",
                        "description": "字体大小(默认48)",
                        "default": 48
                    },
                    "color": {
                        "type": "string",
                        "description": "文字颜色，十六进制格式如#FFFFFF",
                        "default": "#FFFFFF"
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="update_subtitle",
            description="更新字幕文本内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "字幕源名称"
                    },
                    "text": {
                        "type": "string",
                        "description": "新的字幕文本"
                    }
                },
                "required": ["source_name", "text"]
            }
        ),
        Tool(
            name="set_subtitle_style",
            description="设置字幕样式（字体、大小、颜色、描边等）",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "字幕源名称"
                    },
                    "font": {
                        "type": "string",
                        "description": "字体名称"
                    },
                    "size": {
                        "type": "integer",
                        "description": "字体大小"
                    },
                    "color": {
                        "type": "string",
                        "description": "文字颜色(十六进制)"
                    },
                    "outline": {
                        "type": "boolean",
                        "description": "是否启用描边"
                    },
                    "outline_size": {
                        "type": "integer",
                        "description": "描边宽度"
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="show_subtitle",
            description="显示字幕",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "字幕源名称"
                    },
                    "scene_name": {
                        "type": "string",
                        "description": "场景名称(可选)"
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="hide_subtitle",
            description="隐藏字幕",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "字幕源名称"
                    },
                    "scene_name": {
                        "type": "string",
                        "description": "场景名称(可选)"
                    }
                },
                "required": ["source_name"]
            }
        ),
        Tool(
            name="set_subtitle_position",
            description="设置字幕位置",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "字幕源名称"
                    },
                    "scene_name": {
                        "type": "string",
                        "description": "场景名称(可选)"
                    },
                    "x": {
                        "type": "integer",
                        "description": "X坐标"
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y坐标"
                    },
                    "alignment": {
                        "type": "integer",
                        "description": "对齐方式(1-9，数字键盘布局：5=居中，2=底部居中)",
                        "default": 5
                    }
                },
                "required": ["source_name"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    
    if name == "create_subtitle":
        source_name = arguments.get("source_name")
        text = arguments.get("text", "")
        scene_name = arguments.get("scene_name") or get_current_scene()
        font_size = arguments.get("font_size", 48)
        color = arguments.get("color", "#FFFFFF")
        
        if not scene_name:
            result = {"error": "无法获取当前场景，请指定scene_name"}
        else:
            result = create_text_source(scene_name, source_name, text, font_size, color)
    
    elif name == "update_subtitle":
        source_name = arguments.get("source_name")
        text = arguments.get("text", "")
        result = update_subtitle_text(source_name, text)
    
    elif name == "set_subtitle_style":
        source_name = arguments.get("source_name")
        result = set_subtitle_style(
            source_name,
            font=arguments.get("font"),
            size=arguments.get("size"),
            color=arguments.get("color"),
            outline=arguments.get("outline"),
            outline_size=arguments.get("outline_size")
        )
    
    elif name == "show_subtitle":
        source_name = arguments.get("source_name")
        scene_name = arguments.get("scene_name") or get_current_scene()
        if not scene_name:
            result = {"error": "无法获取当前场景，请指定scene_name"}
        else:
            result = set_subtitle_visibility(source_name, scene_name, True)
    
    elif name == "hide_subtitle":
        source_name = arguments.get("source_name")
        scene_name = arguments.get("scene_name") or get_current_scene()
        if not scene_name:
            result = {"error": "无法获取当前场景，请指定scene_name"}
        else:
            result = set_subtitle_visibility(source_name, scene_name, False)
    
    elif name == "set_subtitle_position":
        source_name = arguments.get("source_name")
        scene_name = arguments.get("scene_name") or get_current_scene()
        if not scene_name:
            result = {"error": "无法获取当前场景，请指定scene_name"}
        else:
            result = set_subtitle_position(
                source_name, scene_name,
                x=arguments.get("x"),
                y=arguments.get("y"),
                alignment=arguments.get("alignment", 5)
            )
    
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
