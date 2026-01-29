# -*- coding: utf-8 -*-
"""
OBS控制工具
"""
import json
import urllib.request
from crewai.tools import BaseTool
from pydantic import Field

from src.config import BACKEND_URL


def call_api(endpoint, method="GET", data=None):
    """调用后端API"""
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


class GetScenesTool(BaseTool):
    name: str = "get_scenes"
    description: str = "获取OBS所有场景列表"

    def _run(self) -> str:
        return json.dumps(call_api("scenes"), ensure_ascii=False)


class SwitchSceneTool(BaseTool):
    name: str = "switch_scene"
    description: str = "切换到指定场景"

    def _run(self, scene: str) -> str:
        return json.dumps(call_api("scenes/switch", "POST", {"scene": scene}), ensure_ascii=False)


class GetAudioTool(BaseTool):
    name: str = "get_audio_sources"
    description: str = "获取所有音频源及其音量"

    def _run(self) -> str:
        return json.dumps(call_api("audio/sources"), ensure_ascii=False)


class SetVolumeTool(BaseTool):
    name: str = "set_volume"
    description: str = "设置音频源音量(0-100)"

    def _run(self, source: str, volume: int) -> str:
        return json.dumps(call_api("audio/volume", "POST", {"source": source, "volume": volume}), ensure_ascii=False)
