# -*- coding: utf-8 -*-
"""
配置管理
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 加载.env文件
load_dotenv(PROJECT_ROOT / ".env")

# LLM配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
MODEL = os.getenv("MODEL", "gpt-4o")

# OBS配置
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "123456")

# 后端配置
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/api")


def get_llm():
    """获取配置好的LLM实例"""
    from crewai import LLM
    # 使用openai/前缀强制使用OpenAI兼容模式
    return LLM(
        model=f"openai/{MODEL}",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )
