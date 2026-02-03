# -*- coding: utf-8 -*-
"""
CrewAI OBS控制系统 - 主入口

使用方式:
    python src/main.py [命令]

命令:
    backend  - 启动Flask后端服务
    agent    - 运行OBS控制Agent (传统工具方式)
    mcp      - 运行OBS控制Agent (MCP方式)
    demo     - 运行演示
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def start_backend():
    """启动Flask后端服务"""
    from src.video_backend.app import app
    print("=" * 50)
    print("启动Flask后端服务")
    print("访问地址: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)


def run_agent(user_input: str = None):
    """运行传统工具方式的Agent"""
    from src.agents.obs_agent import run_task
    from src.config import MODEL

    print("=" * 50)
    print("OBS控制Agent (传统工具方式)")
    print(f"使用模型: {MODEL}")
    print("=" * 50)

    if user_input is None:
        user_input = input("请输入指令: ")

    result = run_task(user_input)
    print(f"\n结果: {result}")
    return result


def run_mcp_agent(user_input: str = None):
    """运行MCP方式的Agent"""
    from src.agents.mcp_obs_agent import run_task
    from src.config import MODEL

    print("=" * 50)
    print("OBS控制Agent (MCP方式)")
    print(f"使用模型: {MODEL}")
    print("=" * 50)

    if user_input is None:
        user_input = input("请输入指令: ")

    result = run_task(user_input)
    print(f"\n结果: {result}")
    return result


def run_demo():
    """运行演示"""
    from src.config import MODEL

    print("=" * 60)
    print("CrewAI + MCP 演示")
    print(f"使用模型: {MODEL}")
    print("=" * 60)
    print()
    print("请确保Flask后端已启动: python src/main.py backend")
    print()

    tasks = [
        "获取当前所有场景列表",
        "获取所有音频源及其音量",
    ]

    from src.agents.mcp_obs_agent import run_task

    for i, task in enumerate(tasks, 1):
        print(f"\n--- 任务 {i}: {task} ---")
        try:
            result = run_task(task)
            print(f"结果: {result}")
        except Exception as e:
            print(f"错误: {e}")

    print("\n" + "=" * 60)
    print("演示完成")


def print_help():
    """打印帮助信息"""
    print(__doc__)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    if command == "backend":
        start_backend()
    elif command == "agent":
        user_input = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        run_agent(user_input)
    elif command == "mcp":
        user_input = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        run_mcp_agent(user_input)
    elif command == "demo":
        run_demo()
    elif command in ["help", "-h", "--help"]:
        print_help()
    else:
        print(f"未知命令: {command}")
        print_help()


if __name__ == "__main__":
    main()
