"""
=============================================================================
MCP Servers 模块
说明: 视频处理相关的MCP Server集合
=============================================================================

包含以下Server:
- volume_server: 音量控制
- brightness_server: 亮度调节
- video_filter_server: 视频滤镜（马赛克、虚化、颜色反转等）
"""

from .volume_server import VolumeControlServer
from .brightness_server import BrightnessControlServer

__all__ = [
    "VolumeControlServer",
    "BrightnessControlServer",
]
