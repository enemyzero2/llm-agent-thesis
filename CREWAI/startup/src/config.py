# -*- coding: utf-8 -*-
"""
配置管理
"""
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# LLM配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
MODEL = os.getenv("MODEL", "gpt-4o")

# OBS配置
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "123456")

# 后端配置
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/api")
