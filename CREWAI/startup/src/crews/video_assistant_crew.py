"""
=============================================================================
三层Agent Crew实现
文件: src/crews/video_assistant_crew.py
说明: 实现规划-实施-调用三层Agent架构
=============================================================================
"""

import json
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import Dict, Any
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_bridge.mcp_manager import MCPManager


@CrewBase
class VideoAssistantCrew:
    """
    视频助手Crew

    三层架构:
    1. 规划Agent: 分析用户意图
    2. 实施Agent: 生成TODO列表
    3. 调用Agent: 执行MCP调用
    """

    def __init__(self, mcp_manager: MCPManager):
        """
        初始化视频助手Crew

        Args:
            mcp_manager: MCP管理器实例
        """
        self.mcp_manager = mcp_manager

    # =========================================================================
    # Agent 定义
    # =========================================================================

    @agent
    def planner_agent(self) -> Agent:
        """
        规划Agent

        职责: 分析用户语音指令，理解意图，规划处理方案
        """
        return Agent(
            config=self.agents_config['planner_agent'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def executor_agent(self) -> Agent:
        """
        实施Agent

        职责: 将规划转化为详细的TODO列表
        """
        return Agent(
            config=self.agents_config['executor_agent'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def caller_agent(self) -> Agent:
        """
        调用Agent

        职责: 按TODO列表执行MCP Server调用
        """
        return Agent(
            config=self.agents_config['caller_agent'],
            verbose=True,
            allow_delegation=False,
        )

    # =========================================================================
    # Task 定义
    # =========================================================================

    @task
    def planning_task(self) -> Task:
        """规划任务 - 分析用户意图"""
        return Task(
            config=self.tasks_config['planning_task'],
        )

    @task
    def execution_planning_task(self) -> Task:
        """实施任务 - 生成TODO列表"""
        return Task(
            config=self.tasks_config['execution_planning_task'],
        )

    @task
    def calling_task(self) -> Task:
        """调用任务 - 执行MCP调用"""
        return Task(
            config=self.tasks_config['calling_task'],
        )

    # =========================================================================
    # Crew 定义
    # =========================================================================

    @crew
    def crew(self) -> Crew:
        """创建视频助手Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,  # 顺序执行
            verbose=True,
        )

    # =========================================================================
    # 运行方法
    # =========================================================================

    async def process_command(self, user_command: str) -> Dict[str, Any]:
        """
        处理用户语音指令

        Args:
            user_command: 用户的语音指令

        Returns:
            Dict: 处理结果
        """
        # 获取所有可用能力
        capabilities = self.mcp_manager.get_all_capabilities()
        capabilities_text = self._format_capabilities(capabilities)

        # 准备输入
        inputs = {
            'user_command': user_command,
            'available_capabilities': capabilities_text
        }

        # 执行Crew
        result = self.crew().kickoff(inputs=inputs)

        return {
            'user_command': user_command,
            'result': result.raw,
            'success': True
        }

    def _format_capabilities(self, capabilities: list) -> str:
        """格式化能力列表为文本"""
        text = ""
        for cap in capabilities:
            text += f"- [{cap['server_name']}] {cap['tool_name']}: {cap['tool_description']}\n"
            text += f"  关键词: {', '.join(cap['keywords'])}\n"
        return text


