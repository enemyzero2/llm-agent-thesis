"""
=============================================================================
MCP Server 测试脚本 (简化版)
文件: src/mcp_servers/test_servers.py
说明: 直接测试各个MCP Server的工具函数
=============================================================================
"""

import asyncio
import json
import sys
sys.path.insert(0, '.')


async def test_volume_server():
    """测试音量控制Server"""
    print("\n" + "="*50)
    print("测试音量控制 MCP Server")
    print("="*50)

    # 直接导入并测试call_tool函数
    import volume_server as vs

    print("\n--- 测试 get_volume ---")
    result = await vs.call_tool("get_volume", {})
    print(f"结果: {result[0].text}")

    print("\n--- 测试 set_volume (设为80) ---")
    result = await vs.call_tool("set_volume", {"level": 80})
    print(f"结果: {result[0].text}")

    print("\n--- 测试 adjust_volume (减少20) ---")
    result = await vs.call_tool("adjust_volume", {"delta": -20})
    print(f"结果: {result[0].text}")

    print("\n--- 测试 mute ---")
    result = await vs.call_tool("mute", {})
    print(f"结果: {result[0].text}")


async def test_brightness_server():
    """测试亮度控制Server"""
    print("\n" + "="*50)
    print("测试亮度控制 MCP Server")
    print("="*50)

    import brightness_server as bs

    print("\n--- 测试 set_brightness (设为70) ---")
    result = await bs.call_tool("set_brightness", {"level": 70})
    print(f"结果: {result[0].text}")

    print("\n--- 测试 set_contrast (设为60) ---")
    result = await bs.call_tool("set_contrast", {"level": 60})
    print(f"结果: {result[0].text}")

    print("\n--- 测试 get_brightness ---")
    result = await bs.call_tool("get_brightness", {})
    print(f"结果: {result[0].text}")

    print("\n--- 测试 reset_display ---")
    result = await bs.call_tool("reset_display", {})
    print(f"结果: {result[0].text}")


async def test_filter_server():
    """测试视频滤镜Server"""
    print("\n" + "="*50)
    print("测试视频滤镜 MCP Server")
    print("="*50)

    import video_filter_server as vfs

    print("\n--- 测试 apply_mosaic (启用,强度20) ---")
    result = await vfs.call_tool("apply_mosaic", {"enabled": True, "intensity": 20})
    print(f"结果: {result[0].text}")

    print("\n--- 测试 apply_blur (启用,半径10) ---")
    result = await vfs.call_tool("apply_blur", {"enabled": True, "radius": 10})
    print(f"结果: {result[0].text}")

    print("\n--- 测试 apply_invert (启用) ---")
    result = await vfs.call_tool("apply_invert", {"enabled": True})
    print(f"结果: {result[0].text}")

    print("\n--- 测试 get_filters ---")
    result = await vfs.call_tool("get_filters", {})
    print(f"结果: {result[0].text}")

    print("\n--- 测试 clear_all_filters ---")
    result = await vfs.call_tool("clear_all_filters", {})
    print(f"结果: {result[0].text}")


async def main():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("#" + " "*15 + "MCP Server 功能测试" + " "*16 + "#")
    print("#"*60)

    await test_volume_server()
    await test_brightness_server()
    await test_filter_server()

    print("\n" + "="*50)
    print("所有测试完成!")
    print("="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
