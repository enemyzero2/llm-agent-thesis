"""
=============================================================================
视频助手系统测试脚本
文件: src/test_video_assistant.py
说明: 测试完整的三层Agent + MCP Server架构
=============================================================================
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcp_bridge.mcp_manager import MCPManager
from crews.video_assistant_crew import VideoAssistantCrew


async def test_video_assistant():
    """测试视频助手系统"""

    print("="*60)
    print("视频助手系统测试")
    print("="*60)

    # 1. 初始化MCP Manager
    print("\n[1/4] 初始化MCP Manager...")
    registry_path = project_root / "config" / "mcp_server_registry.yaml"
    mcp_manager = MCPManager(str(registry_path))

    # 2. 连接所有MCP Server
    print("\n[2/4] 连接MCP Servers...")
    await mcp_manager.connect_all_servers()

    # 3. 创建视频助手Crew
    print("\n[3/4] 创建视频助手Crew...")
    assistant = VideoAssistantCrew(mcp_manager)

    # 4. 测试用户指令
    print("\n[4/4] 测试用户指令...")
    print("-"*60)

    test_commands = [
        "把音量调到80",
        "增加亮度",
        "给视频加个马赛克效果",
    ]

    for i, command in enumerate(test_commands, 1):
        print(f"\n测试 {i}/{len(test_commands)}: {command}")
        print("-"*40)

        try:
            result = await assistant.process_command(command)
            print(f"[OK] 执行成功")
            print(f"结果: {result['result'][:200]}...")  # 只显示前200字符
        except Exception as e:
            print(f"[ERROR] 执行失败: {e}")

    # 5. 清理
    print("\n" + "="*60)
    print("清理资源...")
    await mcp_manager.close_all()
    print("✓ 测试完成")


if __name__ == "__main__":
    asyncio.run(test_video_assistant())
