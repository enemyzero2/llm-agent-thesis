# -*- coding: utf-8 -*-
"""
使用MCP协议的OBS控制Agent

这是CrewAI原生MCP集成的实现方式
Agent通过MCP协议与MCP Server通信，而不是直接调用HTTP API
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import get_llm, BACKEND_URL, PROJECT_ROOT
from crewai import Agent, Task, Crew, Process
from crewai.mcp import MCPServerStdio

# 获取LLM实例
llm = get_llm()

# 获取MCP Server路径
MCP_SERVER_PATH = str(PROJECT_ROOT / "src" / "mcp_servers" / "obs_control_server.py")

# 配置MCP Server (stdio方式)
# 使用 uv run 来确保使用项目虚拟环境
obs_mcp_server = MCPServerStdio(
    command="uv",
    args=["run", "python", MCP_SERVER_PATH],
    env={"OBS_BACKEND_URL": BACKEND_URL}
)


def create_obs_agent():
    """创建OBS控制Agent"""
    return Agent(
        role="OBS视频控制专家",
        goal="使用提供的MCP工具来执行OBS控制操作。你必须调用工具来获取真实数据，绝不能编造或猜测结果。",
        backstory="""你是一个专业的OBS控制执行者。

【重要规则】
1. 你必须使用提供的工具(get_scenes, switch_scene, get_audio_sources, set_volume)来执行操作
2. 绝对不能编造场景列表、音量数值等任何数据
3. 如果工具返回error字段，你必须如实告知用户操作失败，并说明错误原因
4. 不要给用户提供"如何手动操作OBS"的教程，直接执行工具获取结果
5. 如果无法连接OBS，直接报告连接失败，不要编造任何数据""",
        llm=llm,
        mcp_servers=[obs_mcp_server],
        verbose=True
    )


def run_task(user_input: str):
    """运行任务

    Args:
        user_input: 用户输入的指令

    Returns:
        执行结果
    """
    agent = create_obs_agent()

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


if __name__ == "__main__":
    from src.config import MODEL
    print("=" * 50)
    print("OBS控制Agent (MCP版)")
    print(f"使用模型: {MODEL}")
    print(f"MCP Server: {MCP_SERVER_PATH}")
    print("=" * 50)

    # 测试任务
    result = run_task("请获取当前所有场景列表")
    print(f"\n结果: {result}")
