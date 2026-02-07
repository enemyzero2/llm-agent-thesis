# -*- coding: utf-8 -*-
"""
=============================================================================
OBS录制/直播控制 MCP Server
文件: src/mcp_servers/obs_recording_server.py
说明: 通过OBS WebSocket控制录制和直播
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
SERVER_NAME = "obs-recording"
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


# ============= 录制控制 =============

def start_recording() -> dict:
    """开始录制"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        obs_client.call(obs_requests.StartRecord())
        return {
            "success": True,
            "action": "start_recording",
            "message": "录制已开始"
        }
    except Exception as e:
        return {"error": f"开始录制失败: {str(e)}"}


def stop_recording() -> dict:
    """停止录制"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        response = obs_client.call(obs_requests.StopRecord())
        output_path = response.getOutputPath() if hasattr(response, 'getOutputPath') else None
        
        return {
            "success": True,
            "action": "stop_recording",
            "output_path": output_path,
            "message": "录制已停止"
        }
    except Exception as e:
        return {"error": f"停止录制失败: {str(e)}"}


def pause_recording() -> dict:
    """暂停录制"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        obs_client.call(obs_requests.PauseRecord())
        return {
            "success": True,
            "action": "pause_recording",
            "message": "录制已暂停"
        }
    except Exception as e:
        return {"error": f"暂停录制失败: {str(e)}"}


def resume_recording() -> dict:
    """恢复录制"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        obs_client.call(obs_requests.ResumeRecord())
        return {
            "success": True,
            "action": "resume_recording",
            "message": "录制已恢复"
        }
    except Exception as e:
        return {"error": f"恢复录制失败: {str(e)}"}


def toggle_recording_pause() -> dict:
    """切换录制暂停状态"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        obs_client.call(obs_requests.ToggleRecordPause())
        return {
            "success": True,
            "action": "toggle_pause",
            "message": "已切换录制暂停状态"
        }
    except Exception as e:
        return {"error": f"切换暂停状态失败: {str(e)}"}


def get_record_status() -> dict:
    """获取录制状态"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        response = obs_client.call(obs_requests.GetRecordStatus())
        
        return {
            "recording": response.getOutputActive(),
            "paused": response.getOutputPaused() if hasattr(response, 'getOutputPaused') else False,
            "duration": response.getOutputDuration() if hasattr(response, 'getOutputDuration') else None,
            "bytes": response.getOutputBytes() if hasattr(response, 'getOutputBytes') else None,
            "timecode": response.getOutputTimecode() if hasattr(response, 'getOutputTimecode') else None,
            "message": "录制中" if response.getOutputActive() else "未录制"
        }
    except Exception as e:
        return {"error": f"获取录制状态失败: {str(e)}"}


# ============= 直播控制 =============

def start_streaming() -> dict:
    """开始直播"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        obs_client.call(obs_requests.StartStream())
        return {
            "success": True,
            "action": "start_streaming",
            "message": "直播已开始"
        }
    except Exception as e:
        return {"error": f"开始直播失败: {str(e)}"}


def stop_streaming() -> dict:
    """停止直播"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        obs_client.call(obs_requests.StopStream())
        return {
            "success": True,
            "action": "stop_streaming",
            "message": "直播已停止"
        }
    except Exception as e:
        return {"error": f"停止直播失败: {str(e)}"}


def get_stream_status() -> dict:
    """获取直播状态"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        response = obs_client.call(obs_requests.GetStreamStatus())
        
        return {
            "streaming": response.getOutputActive(),
            "reconnecting": response.getOutputReconnecting() if hasattr(response, 'getOutputReconnecting') else False,
            "duration": response.getOutputDuration() if hasattr(response, 'getOutputDuration') else None,
            "bytes": response.getOutputBytes() if hasattr(response, 'getOutputBytes') else None,
            "timecode": response.getOutputTimecode() if hasattr(response, 'getOutputTimecode') else None,
            "congestion": response.getOutputCongestion() if hasattr(response, 'getOutputCongestion') else None,
            "skipped_frames": response.getOutputSkippedFrames() if hasattr(response, 'getOutputSkippedFrames') else None,
            "total_frames": response.getOutputTotalFrames() if hasattr(response, 'getOutputTotalFrames') else None,
            "message": "直播中" if response.getOutputActive() else "未直播"
        }
    except Exception as e:
        return {"error": f"获取直播状态失败: {str(e)}"}


def toggle_stream() -> dict:
    """切换直播状态"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        response = obs_client.call(obs_requests.ToggleStream())
        active = response.getOutputActive() if hasattr(response, 'getOutputActive') else None
        
        return {
            "success": True,
            "action": "toggle_stream",
            "streaming": active,
            "message": "已切换直播状态"
        }
    except Exception as e:
        return {"error": f"切换直播状态失败: {str(e)}"}


def toggle_record() -> dict:
    """切换录制状态"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        response = obs_client.call(obs_requests.ToggleRecord())
        active = response.getOutputActive() if hasattr(response, 'getOutputActive') else None
        
        return {
            "success": True,
            "action": "toggle_record",
            "recording": active,
            "message": "已切换录制状态"
        }
    except Exception as e:
        return {"error": f"切换录制状态失败: {str(e)}"}


# ============= 虚拟摄像头 =============

def start_virtual_cam() -> dict:
    """启动虚拟摄像头"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        obs_client.call(obs_requests.StartVirtualCam())
        return {
            "success": True,
            "action": "start_virtual_cam",
            "message": "虚拟摄像头已启动"
        }
    except Exception as e:
        return {"error": f"启动虚拟摄像头失败: {str(e)}"}


def stop_virtual_cam() -> dict:
    """停止虚拟摄像头"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        obs_client.call(obs_requests.StopVirtualCam())
        return {
            "success": True,
            "action": "stop_virtual_cam",
            "message": "虚拟摄像头已停止"
        }
    except Exception as e:
        return {"error": f"停止虚拟摄像头失败: {str(e)}"}


def get_virtual_cam_status() -> dict:
    """获取虚拟摄像头状态"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    try:
        response = obs_client.call(obs_requests.GetVirtualCamStatus())
        active = response.getOutputActive()
        
        return {
            "active": active,
            "message": "虚拟摄像头运行中" if active else "虚拟摄像头未启动"
        }
    except Exception as e:
        return {"error": f"获取虚拟摄像头状态失败: {str(e)}"}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的录制/直播控制工具"""
    return [
        # 录制控制
        Tool(
            name="start_recording",
            description="开始OBS录制",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="stop_recording",
            description="停止OBS录制，返回录制文件路径",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="pause_recording",
            description="暂停OBS录制",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="resume_recording",
            description="恢复OBS录制",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="toggle_recording",
            description="切换OBS录制状态（如果正在录制则停止，否则开始）",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="get_record_status",
            description="获取OBS录制状态（是否正在录制、时长等）",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        # 直播控制
        Tool(
            name="start_streaming",
            description="开始OBS直播推流",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="stop_streaming",
            description="停止OBS直播推流",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="toggle_streaming",
            description="切换OBS直播状态",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="get_stream_status",
            description="获取OBS直播状态（是否正在推流、时长、丢帧等）",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        # 虚拟摄像头
        Tool(
            name="start_virtual_cam",
            description="启动OBS虚拟摄像头",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="stop_virtual_cam",
            description="停止OBS虚拟摄像头",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="get_virtual_cam_status",
            description="获取OBS虚拟摄像头状态",
            inputSchema={"type": "object", "properties": {}, "required": []}
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    
    tool_handlers = {
        "start_recording": start_recording,
        "stop_recording": stop_recording,
        "pause_recording": pause_recording,
        "resume_recording": resume_recording,
        "toggle_recording": toggle_record,
        "get_record_status": get_record_status,
        "start_streaming": start_streaming,
        "stop_streaming": stop_streaming,
        "toggle_streaming": toggle_stream,
        "get_stream_status": get_stream_status,
        "start_virtual_cam": start_virtual_cam,
        "stop_virtual_cam": stop_virtual_cam,
        "get_virtual_cam_status": get_virtual_cam_status,
    }
    
    handler = tool_handlers.get(name)
    if handler:
        result = handler()
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
