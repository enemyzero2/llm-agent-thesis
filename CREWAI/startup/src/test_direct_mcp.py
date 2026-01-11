"""
=============================================================================
直接调用MCP Server测试
文件: src/test_direct_mcp.py
说明: 直接导入并调用MCP Server的函数，不通过stdio连接
=============================================================================
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "mcp_servers"))


async def test_direct_calls():
    """直接调用MCP Server函数"""

    print("="*60)
    print("直接调用MCP Server测试")
    print("="*60)

    # 导入各个server
    import volume_server as vs
    import brightness_server as bs
    import video_filter_server as vfs

    # 测试1: 音量控制
    print("\n[测试1] 音量控制")
    print("-"*40)
    result = await vs.call_tool("set_volume", {"level": 80})
    print(f"设置音量到80: {result[0].text}")

    result = await vs.call_tool("get_volume", {})
    print(f"查询当前音量: {result[0].text}")

    # 测试2: 亮度控制
    print("\n[测试2] 亮度控制")
    print("-"*40)
    result = await bs.call_tool("set_brightness", {"level": 70})
    print(f"设置亮度到70: {result[0].text}")

    result = await bs.call_tool("set_contrast", {"level": 60})
    print(f"设置对比度到60: {result[0].text}")

    # 测试3: 视频滤镜
    print("\n[测试3] 视频滤镜")
    print("-"*40)
    result = await vfs.call_tool("apply_mosaic", {"enabled": True, "intensity": 20})
    print(f"应用马赛克: {result[0].text}")

    result = await vfs.call_tool("apply_blur", {"enabled": True, "radius": 10})
    print(f"应用虚化: {result[0].text}")

    result = await vfs.call_tool("get_filters", {})
    print(f"查询滤镜状态: {result[0].text}")

    print("\n" + "="*60)
    print("[OK] 所有测试通过")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_direct_calls())
