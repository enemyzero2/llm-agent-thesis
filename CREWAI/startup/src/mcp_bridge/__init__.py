"""
=============================================================================
MCP Bridge 模块
说明: CrewAI与MCP Server之间的桥接层
=============================================================================
"""

from .mcp_client import MCPClient
from .mcp_manager import MCPManager, MCPServerInfo

__all__ = ["MCPClient", "MCPManager", "MCPServerInfo"]
