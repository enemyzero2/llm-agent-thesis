# -*- coding: utf-8 -*-
"""
OBS控制Agent
"""
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
load_dotenv()

# 设置环境变量（CrewAI使用这些）
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("MODEL", "gpt-4o")

from crewai import Agent, Task, Crew
from src.tools.obs_tools import GetScenesTool, SwitchSceneTool, GetAudioTool, SetVolumeTool

# 创建Agent
obs_agent = Agent(
    role="OBS视频控制专家",
    goal="根据用户指令控制OBS的场景切换和音量调节",
    backstory="你是一个专业的视频直播控制专家。",
    tools=[GetScenesTool(), SwitchSceneTool(), GetAudioTool(), SetVolumeTool()],
    verbose=True
)


def run_task(user_input: str):
    """运行任务"""
    task = Task(
        description=user_input,
        expected_output="执行结果",
        agent=obs_agent
    )
    crew = Crew(agents=[obs_agent], tasks=[task], verbose=True)
    return crew.kickoff()


if __name__ == "__main__":
    print("OBS控制Agent")
    print(f"使用模型: {os.getenv('MODEL', 'gpt-4o')}")
    result = run_task("请把桌面音频的音量设置为50%")
    print(f"结果: {result}")
