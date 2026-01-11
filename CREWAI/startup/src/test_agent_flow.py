"""
=============================================================================
简化测试脚本 - 只测试Agent流程
文件: src/test_agent_flow.py
说明: 先测试Agent的规划-实施-调用流程，不实际连接MCP Server
=============================================================================
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcp_bridge.mcp_manager import MCPManager


def test_mcp_manager():
    """测试MCP Manager的基础功能"""

    print("="*60)
    print("测试 MCP Manager")
    print("="*60)

    # 1. 加载注册表
    print("\n[1/3] 加载MCP Server注册表...")
    registry_path = project_root / "config" / "mcp_server_registry.yaml"
    mcp_manager = MCPManager(str(registry_path))

    # 2. 查看所有能力
    print("\n[2/3] 查看所有可用能力...")
    capabilities = mcp_manager.get_all_capabilities()
    print(f"共有 {len(capabilities)} 个能力:")
    for cap in capabilities[:5]:  # 只显示前5个
        print(f"  - [{cap['server_name']}] {cap['tool_name']}: {cap['tool_description']}")

    # 3. 测试关键词匹配
    print("\n[3/3] 测试关键词匹配...")
    test_keywords = ["音量", "亮度", "马赛克"]
    for keyword in test_keywords:
        result = mcp_manager.find_tool_by_keywords(keyword)
        if result:
            print(f"  '{keyword}' -> {result['server_name']}.{result['tool_name']}")
        else:
            print(f"  '{keyword}' -> 未找到匹配")

    print("\n" + "="*60)
    print("[OK] MCP Manager测试完成")
    print("="*60)


if __name__ == "__main__":
    test_mcp_manager()
