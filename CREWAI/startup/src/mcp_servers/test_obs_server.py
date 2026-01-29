"""
=============================================================================
OBS MCP Server 测试脚本
文件: src/mcp_servers/test_obs_server.py
说明: 测试OBS音量控制MCP Server的功能
=============================================================================
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp_bridge.mcp_client import MCPClient


async def test_obs_volume_control():
    """测试OBS音量控制功能"""
    print("=" * 60)
    print("OBS音量控制 MCP Server 测试")
    print("=" * 60)

    # 创建MCP客户端
    client = MCPClient()

    # 连接到OBS音量控制服务器
    print("\n[1] 连接到OBS音量控制服务器...")
    success = await client.connect(
        server_name="obs-volume",
        command="python",
        args=[
            os.path.join(os.path.dirname(__file__), "obs_volume_server.py")
        ]
    )

    if not success:
        print("[ERROR] 连接失败")
        return

    # 等待服务器初始化
    await asyncio.sleep(2)

    # 列出可用工具
    print("\n[2] 获取可用工具列表...")
    tools = await client.list_tools("obs-volume")
    if tools:
        print(f"[OK] 找到 {len(tools)} 个工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
    else:
        print("[ERROR] 无法获取工具列表")
        return

    # 测试：列出音频源
    print("\n[3] 列出所有音频源...")
    result = await client.call_tool(
        server_name="obs-volume",
        tool_name="list_audio_sources",
        arguments={}
    )
    print(f"[结果] {result}")

    # 如果有音频源，测试音量控制
    if result and "sources" in result:
        import json
        data = json.loads(result)
        sources = data.get("sources", [])

        if sources:
            test_source = sources[0]
            print(f"\n[4] 测试音频源: {test_source}")

            # 获取当前音量
            print(f"\n[4.1] 获取 {test_source} 的当前音量...")
            result = await client.call_tool(
                server_name="obs-volume",
                tool_name="get_source_volume",
                arguments={"source_name": test_source}
            )
            print(f"[结果] {result}")

            # 设置音量为70%
            print(f"\n[4.2] 设置 {test_source} 的音量为 70%...")
            result = await client.call_tool(
                server_name="obs-volume",
                tool_name="set_source_volume",
                arguments={"source_name": test_source, "level": 70}
            )
            print(f"[结果] {result}")

            # 等待一下
            await asyncio.sleep(1)

            # 静音
            print(f"\n[4.3] 静音 {test_source}...")
            result = await client.call_tool(
                server_name="obs-volume",
                tool_name="mute_source",
                arguments={"source_name": test_source}
            )
            print(f"[结果] {result}")

        else:
            print("[INFO] 没有找到音频源，跳过音量控制测试")

    # 关闭连接
    print("\n[5] 关闭连接...")
    await client.close("obs-volume")
    print("[OK] 测试完成")


if __name__ == "__main__":
    asyncio.run(test_obs_volume_control())
