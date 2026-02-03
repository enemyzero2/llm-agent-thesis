# -*- coding: utf-8 -*-
"""
CrewAI + MCP 完整演示

启动顺序:
1. 启动Flask后端 (连接OBS)
2. 运行此脚本 (Agent通过MCP调用工具)
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agents.mcp_obs_agent import run_task
from src.config import MODEL


def main():
    """演示主函数"""
    print("=" * 60)
    print("CrewAI + MCP 演示")
    print(f"使用模型: {MODEL}")
    print("=" * 60)
    print()
    print("确保Flask后端已启动: python src/main.py backend")
    print()

    # 演示任务
    tasks = [
        "获取当前所有场景列表",
        "获取所有音频源及其音量",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n--- 任务 {i}: {task} ---")
        try:
            result = run_task(task)
            print(f"结果: {result}")
        except Exception as e:
            print(f"错误: {e}")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

