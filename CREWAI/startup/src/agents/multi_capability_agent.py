# -*- coding: utf-8 -*-
"""
OBS多能力引擎智能体

支持所有OBS能力引擎的CrewAI Agent:
- 字幕控制 (obs-subtitle)
- 翻译转写 (obs-translation)
- 背景虚化 (obs-background)
- 滤镜效果 (obs-filter)
- 录制直播 (obs-recording)
- 场景控制 (obs-control)
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import get_llm, BACKEND_URL, PROJECT_ROOT, OPENAI_API_KEY, OPENAI_API_BASE
from crewai import Agent, Task, Crew, Process
from crewai.mcp import MCPServerStdio

# 获取LLM实例
llm = get_llm()

# MCP Server目录
MCP_SERVERS_DIR = PROJECT_ROOT / "src" / "mcp_servers"

# 环境变量配置
MCP_ENV = {
    "OBS_BACKEND_URL": BACKEND_URL,
    "OBS_HOST": os.getenv("OBS_HOST", "localhost"),
    "OBS_PORT": os.getenv("OBS_PORT", "4455"),
    "OBS_PASSWORD": os.getenv("OBS_PASSWORD", "123456"),
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "OPENAI_BASE_URL": OPENAI_API_BASE,
}

# =============================================================================
# 定义所有MCP能力引擎
# =============================================================================

# 1. 场景和音频控制
obs_control_server = MCPServerStdio(
    command="uv",
    args=["run", "python", str(MCP_SERVERS_DIR / "obs_control_server.py")],
    env=MCP_ENV
)

# 2. 字幕控制
obs_subtitle_server = MCPServerStdio(
    command="uv",
    args=["run", "python", str(MCP_SERVERS_DIR / "obs_subtitle_server.py")],
    env=MCP_ENV
)

# 3. 翻译和语音识别
obs_translation_server = MCPServerStdio(
    command="uv",
    args=["run", "python", str(MCP_SERVERS_DIR / "obs_translation_server.py")],
    env=MCP_ENV
)

# 4. 背景虚化
obs_background_server = MCPServerStdio(
    command="uv",
    args=["run", "python", str(MCP_SERVERS_DIR / "obs_background_server.py")],
    env=MCP_ENV
)

# 5. 滤镜控制
obs_filter_server = MCPServerStdio(
    command="uv",
    args=["run", "python", str(MCP_SERVERS_DIR / "obs_filter_server.py")],
    env=MCP_ENV
)

# 6. 录制和直播控制
obs_recording_server = MCPServerStdio(
    command="uv",
    args=["run", "python", str(MCP_SERVERS_DIR / "obs_recording_server.py")],
    env=MCP_ENV
)

# 所有MCP服务器列表
ALL_MCP_SERVERS = [
    obs_control_server,
    obs_subtitle_server,
    obs_translation_server,
    obs_background_server,
    obs_filter_server,
    obs_recording_server,
]


def create_obs_agent(include_all_servers=True):
    """创建OBS多能力智能体
    
    Args:
        include_all_servers: 是否包含所有MCP服务器，False则只包含基础控制
    """
    servers = ALL_MCP_SERVERS if include_all_servers else [obs_control_server]
    
    return Agent(
        role="OBS视频处理专家",
        goal="""使用提供的MCP工具来执行各种OBS视频处理操作。
你可以：
- 控制场景和音频 (get_scenes, switch_scene, set_volume)
- 管理字幕 (create_subtitle, update_subtitle, show_subtitle, hide_subtitle)
- 进行语音识别和翻译 (transcribe_audio_file, translate_text)
- 应用背景虚化 (apply_background_blur, set_virtual_background)
- 添加滤镜效果 (add_filter, apply_color_correction)
- 控制录制和直播 (start_recording, stop_recording, start_streaming)

你必须调用工具来获取真实数据，绝不能编造结果。""",
        backstory="""你是一个专业的OBS视频处理专家，拥有以下能力：

【可用能力引擎】
1. 场景控制 - 切换场景、调整音量
2. 字幕系统 - 创建、更新、显示/隐藏字幕
3. 翻译引擎 - 语音识别(Whisper)和实时翻译(GPT)
4. 背景处理 - 背景虚化、虚拟背景、绿幕抠像
5. 滤镜效果 - 色彩校正、模糊、锐化等
6. 录制控制 - 开始/停止录制、直播推流

【重要规则】
1. 必须使用工具执行操作，不能编造数据
2. 如果工具返回error，如实告知用户并说明原因
3. 根据用户意图选择合适的能力引擎
4. 可以组合多个能力来完成复杂任务""",
        llm=llm,
        mcp_servers=servers,
        verbose=True
    )


def run_task(user_input: str, include_all_servers=True):
    """运行任务

    Args:
        user_input: 用户输入的指令
        include_all_servers: 是否使用所有MCP服务器

    Returns:
        执行结果
    """
    agent = create_obs_agent(include_all_servers)

    task = Task(
        description=user_input,
        expected_output="执行结果的详细描述",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()


def interactive_mode():
    """交互式模式"""
    from src.config import MODEL
    
    print("=" * 60)
    print("OBS多能力智能体 - 交互模式")
    print(f"使用模型: {MODEL}")
    print("=" * 60)
    print()
    print("可用指令示例:")
    print("  - 获取所有场景")
    print("  - 创建一个字幕显示'Hello World'")
    print("  - 把摄像头的背景虚化")
    print("  - 开始录制")
    print("  - 翻译这段文字: Hello")
    print()
    print("输入 'quit' 退出")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n请输入指令: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            if not user_input:
                continue
                
            result = run_task(user_input)
            print(f"\n结果: {result}")
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OBS多能力智能体")
    parser.add_argument("command", nargs="?", default="interactive",
                        help="命令: interactive(交互模式) 或直接输入任务")
    parser.add_argument("--basic", action="store_true",
                        help="只使用基础控制服务器")
    args = parser.parse_args()
    
    if args.command == "interactive":
        interactive_mode()
    else:
        # 直接执行任务
        result = run_task(args.command, include_all_servers=not args.basic)
        print(f"\n结果: {result}")
