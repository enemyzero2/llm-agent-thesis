# -*- coding: utf-8 -*-
"""
=============================================================================
OBS高级功能 MCP Server 测试脚本
文件: src/mcp_servers/test_obs_advanced.py
说明: 测试字幕、滤镜、翻译、录制等高级功能
=============================================================================
"""

import asyncio
import sys
import os
import json

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp_bridge.mcp_client import MCPClient


async def test_subtitle_server():
    """测试字幕服务器"""
    print("\n" + "=" * 60)
    print("【测试1】OBS字幕控制服务器")
    print("=" * 60)

    client = MCPClient()

    print("\n[1.1] 连接到字幕控制服务器...")
    success = await client.connect(
        server_name="obs-subtitle",
        command="python",
        args=[os.path.join(os.path.dirname(__file__), "obs_subtitle_server.py")]
    )

    if not success:
        print("[ERROR] 连接失败")
        return False

    await asyncio.sleep(2)

    # 列出工具
    print("\n[1.2] 获取可用工具列表...")
    tools = await client.list_tools("obs-subtitle")
    if tools:
        print(f"[OK] 找到 {len(tools)} 个工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:40]}...")
    else:
        print("[ERROR] 无法获取工具列表")
        await client.close("obs-subtitle")
        return False

    # 测试创建字幕
    print("\n[1.3] 创建字幕源...")
    result = await client.call_tool(
        server_name="obs-subtitle",
        tool_name="create_subtitle",
        arguments={
            "source_name": "测试字幕",
            "text": "Hello, World! 你好，世界！",
            "font_size": 48,
            "color": "#FFFFFF"
        }
    )
    print(f"[结果] {result}")

    # 测试更新字幕
    print("\n[1.4] 更新字幕内容...")
    result = await client.call_tool(
        server_name="obs-subtitle",
        tool_name="update_subtitle",
        arguments={
            "source_name": "测试字幕",
            "text": "这是更新后的字幕内容"
        }
    )
    print(f"[结果] {result}")

    await client.close("obs-subtitle")
    print("\n[OK] 字幕服务器测试完成")
    return True


async def test_filter_server():
    """测试滤镜服务器"""
    print("\n" + "=" * 60)
    print("【测试2】OBS滤镜控制服务器")
    print("=" * 60)

    client = MCPClient()

    print("\n[2.1] 连接到滤镜控制服务器...")
    success = await client.connect(
        server_name="obs-filter",
        command="python",
        args=[os.path.join(os.path.dirname(__file__), "obs_filter_server.py")]
    )

    if not success:
        print("[ERROR] 连接失败")
        return False

    await asyncio.sleep(2)

    # 列出工具
    print("\n[2.2] 获取可用工具列表...")
    tools = await client.list_tools("obs-filter")
    if tools:
        print(f"[OK] 找到 {len(tools)} 个工具:")
        for tool in tools:
            print(f"  - {tool.name}")

    # 获取支持的滤镜类型
    print("\n[2.3] 获取支持的滤镜类型...")
    result = await client.call_tool(
        server_name="obs-filter",
        tool_name="get_supported_filters",
        arguments={}
    )
    print(f"[结果] {result}")

    await client.close("obs-filter")
    print("\n[OK] 滤镜服务器测试完成")
    return True


async def test_translation_server():
    """测试翻译服务器"""
    print("\n" + "=" * 60)
    print("【测试3】OBS翻译服务器")
    print("=" * 60)

    client = MCPClient()

    print("\n[3.1] 连接到翻译服务器...")
    success = await client.connect(
        server_name="obs-translation",
        command="python",
        args=[os.path.join(os.path.dirname(__file__), "obs_translation_server.py")]
    )

    if not success:
        print("[ERROR] 连接失败")
        return False

    await asyncio.sleep(2)

    # 列出工具
    print("\n[3.2] 获取可用工具列表...")
    tools = await client.list_tools("obs-translation")
    if tools:
        print(f"[OK] 找到 {len(tools)} 个工具:")
        for tool in tools:
            print(f"  - {tool.name}")

    # 获取支持的语言
    print("\n[3.3] 获取支持的语言列表...")
    result = await client.call_tool(
        server_name="obs-translation",
        tool_name="get_supported_languages",
        arguments={}
    )
    print(f"[结果] {result}")

    # 测试翻译（如果配置了OpenAI API Key）
    print("\n[3.4] 测试文本翻译...")
    result = await client.call_tool(
        server_name="obs-translation",
        tool_name="translate_text",
        arguments={
            "text": "Hello, this is a test.",
            "source_language": "en",
            "target_language": "zh"
        }
    )
    print(f"[结果] {result}")

    await client.close("obs-translation")
    print("\n[OK] 翻译服务器测试完成")
    return True


async def test_recording_server():
    """测试录制服务器"""
    print("\n" + "=" * 60)
    print("【测试4】OBS录制控制服务器")
    print("=" * 60)

    client = MCPClient()

    print("\n[4.1] 连接到录制控制服务器...")
    success = await client.connect(
        server_name="obs-recording",
        command="python",
        args=[os.path.join(os.path.dirname(__file__), "obs_recording_server.py")]
    )

    if not success:
        print("[ERROR] 连接失败")
        return False

    await asyncio.sleep(2)

    # 列出工具
    print("\n[4.2] 获取可用工具列表...")
    tools = await client.list_tools("obs-recording")
    if tools:
        print(f"[OK] 找到 {len(tools)} 个工具:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

    # 获取录制状态
    print("\n[4.3] 获取录制状态...")
    result = await client.call_tool(
        server_name="obs-recording",
        tool_name="get_record_status",
        arguments={}
    )
    print(f"[结果] {result}")

    # 获取直播状态
    print("\n[4.4] 获取直播状态...")
    result = await client.call_tool(
        server_name="obs-recording",
        tool_name="get_stream_status",
        arguments={}
    )
    print(f"[结果] {result}")

    # 获取虚拟摄像头状态
    print("\n[4.5] 获取虚拟摄像头状态...")
    result = await client.call_tool(
        server_name="obs-recording",
        tool_name="get_virtual_cam_status",
        arguments={}
    )
    print(f"[结果] {result}")

    await client.close("obs-recording")
    print("\n[OK] 录制服务器测试完成")
    return True


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("OBS高级功能 MCP Server 测试")
    print("=" * 60)
    print("\n请确保:")
    print("1. OBS Studio 正在运行")
    print("2. OBS WebSocket 服务器已启用 (工具 -> WebSocket服务器设置)")
    print("3. 端口: 4455, 密码: 123456 (默认)")
    print("\n开始测试...\n")

    results = []

    # 测试字幕服务器
    try:
        results.append(("字幕服务器", await test_subtitle_server()))
    except Exception as e:
        print(f"[ERROR] 字幕服务器测试失败: {e}")
        results.append(("字幕服务器", False))

    # 测试滤镜服务器
    try:
        results.append(("滤镜服务器", await test_filter_server()))
    except Exception as e:
        print(f"[ERROR] 滤镜服务器测试失败: {e}")
        results.append(("滤镜服务器", False))

    # 测试翻译服务器
    try:
        results.append(("翻译服务器", await test_translation_server()))
    except Exception as e:
        print(f"[ERROR] 翻译服务器测试失败: {e}")
        results.append(("翻译服务器", False))

    # 测试录制服务器
    try:
        results.append(("录制服务器", await test_recording_server()))
    except Exception as e:
        print(f"[ERROR] 录制服务器测试失败: {e}")
        results.append(("录制服务器", False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\n总计: {passed}/{len(results)} 通过")


if __name__ == "__main__":
    asyncio.run(main())
