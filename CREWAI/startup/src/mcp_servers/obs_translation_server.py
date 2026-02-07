# -*- coding: utf-8 -*-
"""
=============================================================================
OBS同声传译 MCP Server
文件: src/mcp_servers/obs_translation_server.py
说明: 提供实时语音识别和翻译功能，使用OpenAI Whisper和GPT-4
=============================================================================
"""

import asyncio
import json
import os
import sys
import tempfile
import wave
from typing import Optional, Dict, Any
from datetime import datetime

from mcp.server import Server
from mcp.types import Tool, TextContent

# 检查OpenAI可用性
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARNING] openai未安装，翻译功能将不可用", file=sys.stderr)
    print("安装命令: pip install openai", file=sys.stderr)

# 尝试导入obs-websocket-py用于更新字幕
try:
    from obswebsocket import obsws, requests as obs_requests
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    print("[WARNING] obs-websocket-py未安装", file=sys.stderr)

# 尝试导入音频处理库
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARNING] sounddevice未安装，实时音频捕获将不可用", file=sys.stderr)
    print("安装命令: pip install sounddevice numpy", file=sys.stderr)

# 创建Server实例
SERVER_NAME = "obs-translation"
server = Server(SERVER_NAME)

# OpenAI配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # 支持第三方中转站
openai_client: Optional[OpenAI] = None

# OBS连接配置
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "123456")
obs_client: Optional[obsws] = None

# 翻译状态
translation_state = {
    "active": False,
    "source_language": "auto",  # 源语言 (auto表示自动检测)
    "target_language": "zh",    # 目标语言
    "subtitle_source": None,    # OBS字幕源名称
    "scene_name": None,         # OBS场景名称
    "history": [],              # 翻译历史
    "recording_stream": None,   # 音频录制流
}

# 支持的语言
SUPPORTED_LANGUAGES = {
    "auto": "自动检测",
    "zh": "中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "es": "西班牙语",
    "fr": "法语",
    "de": "德语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "pt": "葡萄牙语",
    "it": "意大利语"
}


def init_openai() -> bool:
    """初始化OpenAI客户端"""
    global openai_client
    
    if not OPENAI_AVAILABLE:
        return False
    
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"[{SERVER_NAME}] 未设置OPENAI_API_KEY环境变量", file=sys.stderr)
        return False
    
    try:
        # 支持第三方中转站
        base_url = OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL")
        if base_url:
            openai_client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"[{SERVER_NAME}] 使用自定义API地址: {base_url}", file=sys.stderr)
        else:
            openai_client = OpenAI(api_key=api_key)
        return True
    except Exception as e:
        print(f"[{SERVER_NAME}] 初始化OpenAI失败: {e}", file=sys.stderr)
        return False


def connect_obs() -> bool:
    """连接到OBS WebSocket"""
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


def transcribe_audio(audio_file_path: str, language: str = "auto") -> dict:
    """使用OpenAI Whisper进行语音识别"""
    if openai_client is None:
        return {"error": "OpenAI客户端未初始化"}
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            params = {"model": "whisper-1", "file": audio_file}
            
            # 如果指定了语言且不是auto,则传递给API
            if language != "auto" and language in SUPPORTED_LANGUAGES:
                params["language"] = language
            
            response = openai_client.audio.transcriptions.create(**params)
            
        return {
            "success": True,
            "text": response.text,
            "language": language
        }
    except Exception as e:
        return {"error": f"语音识别失败: {str(e)}"}


def translate_text(text: str, source_lang: str, target_lang: str) -> dict:
    """使用OpenAI GPT进行文本翻译"""
    if openai_client is None:
        return {"error": "OpenAI客户端未初始化"}
    
    if not text.strip():
        return {"success": True, "translated_text": "", "original_text": text}
    
    try:
        source_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
        
        prompt = f"""请将以下{source_name}文本翻译成{target_name}。只返回翻译结果，不要添加任何解释或额外内容。

原文：{text}

翻译："""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个专业的翻译助手，只返回翻译结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        translated = response.choices[0].message.content.strip()
        
        return {
            "success": True,
            "original_text": text,
            "translated_text": translated,
            "source_language": source_lang,
            "target_language": target_lang
        }
    except Exception as e:
        return {"error": f"翻译失败: {str(e)}"}


def update_obs_subtitle(text: str) -> dict:
    """更新OBS字幕源"""
    if obs_client is None:
        return {"error": "未连接到OBS"}
    
    source_name = translation_state.get("subtitle_source")
    if not source_name:
        return {"error": "未设置字幕源"}
    
    try:
        obs_client.call(obs_requests.SetInputSettings(
            inputName=source_name,
            inputSettings={"text": text}
        ))
        return {"success": True, "text": text}
    except Exception as e:
        return {"error": f"更新字幕失败: {str(e)}"}


def transcribe_and_translate(audio_file_path: str) -> dict:
    """完整的语音识别+翻译流程"""
    source_lang = translation_state.get("source_language", "auto")
    target_lang = translation_state.get("target_language", "zh")
    
    # 1. 语音识别
    transcription = transcribe_audio(audio_file_path, source_lang)
    if "error" in transcription:
        return transcription
    
    original_text = transcription.get("text", "")
    
    # 2. 翻译（如果源语言和目标语言不同）
    if source_lang == target_lang and source_lang != "auto":
        translated_text = original_text
    else:
        translation = translate_text(original_text, source_lang, target_lang)
        if "error" in translation:
            return translation
        translated_text = translation.get("translated_text", "")
    
    # 3. 更新OBS字幕
    if translation_state.get("subtitle_source"):
        update_obs_subtitle(translated_text)
    
    # 4. 记录历史
    record = {
        "timestamp": datetime.now().isoformat(),
        "original": original_text,
        "translated": translated_text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    translation_state["history"].append(record)
    
    # 保留最近50条记录
    if len(translation_state["history"]) > 50:
        translation_state["history"] = translation_state["history"][-50:]
    
    return {
        "success": True,
        "original_text": original_text,
        "translated_text": translated_text,
        "source_language": source_lang,
        "target_language": target_lang
    }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的翻译工具"""
    return [
        Tool(
            name="transcribe_audio_file",
            description="对音频文件进行语音识别（使用OpenAI Whisper）",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "音频文件路径"
                    },
                    "language": {
                        "type": "string",
                        "description": "源语言代码(可选，默认自动检测)",
                        "enum": list(SUPPORTED_LANGUAGES.keys())
                    }
                },
                "required": ["audio_path"]
            }
        ),
        Tool(
            name="translate_text",
            description="翻译文本（使用OpenAI GPT-4）",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要翻译的文本"
                    },
                    "source_language": {
                        "type": "string",
                        "description": "源语言代码",
                        "enum": list(SUPPORTED_LANGUAGES.keys()),
                        "default": "auto"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "目标语言代码",
                        "enum": list(SUPPORTED_LANGUAGES.keys()),
                        "default": "zh"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="transcribe_and_translate",
            description="语音识别并翻译音频文件，结果可更新到OBS字幕",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "音频文件路径"
                    }
                },
                "required": ["audio_path"]
            }
        ),
        Tool(
            name="set_translation_config",
            description="配置翻译参数（源语言、目标语言、OBS字幕源）",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_language": {
                        "type": "string",
                        "description": "源语言代码",
                        "enum": list(SUPPORTED_LANGUAGES.keys())
                    },
                    "target_language": {
                        "type": "string",
                        "description": "目标语言代码",
                        "enum": list(SUPPORTED_LANGUAGES.keys())
                    },
                    "subtitle_source": {
                        "type": "string",
                        "description": "OBS字幕源名称(用于显示翻译结果)"
                    },
                    "scene_name": {
                        "type": "string",
                        "description": "OBS场景名称(可选)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_translation_config",
            description="获取当前翻译配置",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_translation_history",
            description="获取翻译历史记录",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "返回的记录数量(默认10)",
                        "default": 10
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="update_subtitle",
            description="直接更新OBS字幕内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "字幕文本"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="get_supported_languages",
            description="获取支持的语言列表",
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
    
    if name == "transcribe_audio_file":
        audio_path = arguments.get("audio_path")
        language = arguments.get("language", "auto")
        result = transcribe_audio(audio_path, language)
    
    elif name == "translate_text":
        text = arguments.get("text", "")
        source_lang = arguments.get("source_language", "auto")
        target_lang = arguments.get("target_language", "zh")
        result = translate_text(text, source_lang, target_lang)
    
    elif name == "transcribe_and_translate":
        audio_path = arguments.get("audio_path")
        result = transcribe_and_translate(audio_path)
    
    elif name == "set_translation_config":
        if "source_language" in arguments:
            translation_state["source_language"] = arguments["source_language"]
        if "target_language" in arguments:
            translation_state["target_language"] = arguments["target_language"]
        if "subtitle_source" in arguments:
            translation_state["subtitle_source"] = arguments["subtitle_source"]
        if "scene_name" in arguments:
            translation_state["scene_name"] = arguments["scene_name"]
        
        result = {
            "success": True,
            "config": {
                "source_language": translation_state["source_language"],
                "target_language": translation_state["target_language"],
                "subtitle_source": translation_state["subtitle_source"],
                "scene_name": translation_state["scene_name"]
            },
            "message": "翻译配置已更新"
        }
    
    elif name == "get_translation_config":
        result = {
            "config": {
                "source_language": translation_state["source_language"],
                "target_language": translation_state["target_language"],
                "subtitle_source": translation_state["subtitle_source"],
                "scene_name": translation_state["scene_name"],
                "active": translation_state["active"]
            },
            "openai_available": openai_client is not None,
            "obs_connected": obs_client is not None
        }
    
    elif name == "get_translation_history":
        limit = arguments.get("limit", 10)
        history = translation_state["history"][-limit:]
        result = {
            "history": history,
            "count": len(history),
            "total": len(translation_state["history"])
        }
    
    elif name == "update_subtitle":
        text = arguments.get("text", "")
        result = update_obs_subtitle(text)
    
    elif name == "get_supported_languages":
        result = {
            "languages": SUPPORTED_LANGUAGES,
            "default_source": "auto",
            "default_target": "zh"
        }
    
    else:
        result = {"error": f"未知工具: {name}"}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def main():
    """启动MCP Server"""
    from mcp.server.stdio import stdio_server

    print(f"[{SERVER_NAME}] MCP Server 启动中...", file=sys.stderr)
    
    # 初始化OpenAI
    if init_openai():
        print(f"[{SERVER_NAME}] OpenAI客户端已初始化", file=sys.stderr)
    else:
        print(f"[{SERVER_NAME}] OpenAI不可用，请设置OPENAI_API_KEY环境变量", file=sys.stderr)
    
    # 连接OBS
    connect_obs()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
