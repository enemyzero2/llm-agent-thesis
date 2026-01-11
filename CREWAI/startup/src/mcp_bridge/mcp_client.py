"""
=============================================================================
MCP Client 实现
文件: src/mcp_bridge/mcp_client.py
说明: MCP协议客户端，负责与MCP Server通信
=============================================================================
"""

import asyncio
from typing import Dict, List, Optional, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """
    MCP客户端

    职责:
    1. 管理与MCP Server的连接
    2. 提供工具调用接口
    3. 处理通信错误和重连
    """

    def __init__(self):
        """初始化MCP客户端"""
        self.sessions: Dict[str, ClientSession] = {}
        self.server_params: Dict[str, StdioServerParameters] = {}

    async def connect(
        self,
        server_name: str,
        command: str,
        args: List[str]
    ) -> bool:
        """
        连接到MCP Server

        Args:
            server_name: Server标识名
            command: 启动命令
            args: 命令参数

        Returns:
            bool: 连接是否成功
        """
        if server_name in self.sessions:
            print(f"[OK] Server '{server_name}' 已连接")
            return True

        try:
            # 创建Server参数
            server_params = StdioServerParameters(
                command=command,
                args=args
            )
            self.server_params[server_name] = server_params

            # 建立连接 - 不使用async with，保持连接
            read_stream, write_stream = await stdio_client(server_params)
            session = ClientSession(read_stream, write_stream)

            # 初始化会话
            await session.initialize()
            self.sessions[server_name] = session

            print(f"[OK] 已连接到 '{server_name}'")
            return True

        except Exception as e:
            print(f"[ERROR] 连接 '{server_name}' 失败: {e}")
            return False

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[str]:
        """
        调用MCP Server的工具

        Args:
            server_name: Server名称
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            Optional[str]: 工具执行结果，失败返回None
        """
        if server_name not in self.sessions:
            print(f"[ERROR] Server '{server_name}' 未连接")
            return None

        try:
            session = self.sessions[server_name]
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text

        except Exception as e:
            print(f"[ERROR] 调用工具失败 [{server_name}.{tool_name}]: {e}")
            return None

    async def list_tools(self, server_name: str) -> Optional[List]:
        """
        获取Server的工具列表

        Args:
            server_name: Server名称

        Returns:
            Optional[List]: 工具列表
        """
        if server_name not in self.sessions:
            print(f"[ERROR] Server '{server_name}' 未连接")
            return None

        try:
            session = self.sessions[server_name]
            tools = await session.list_tools()
            return tools.tools

        except Exception as e:
            print(f"[ERROR] 获取工具列表失败 [{server_name}]: {e}")
            return None

    async def reconnect(self, server_name: str) -> bool:
        """
        重新连接到Server

        Args:
            server_name: Server名称

        Returns:
            bool: 重连是否成功
        """
        # 先关闭现有连接
        await self.close(server_name)

        # 重新连接
        if server_name not in self.server_params:
            print(f"[ERROR] 没有 '{server_name}' 的连接参数")
            return False

        params = self.server_params[server_name]
        return await self.connect(server_name, params.command, params.args)

    async def close(self, server_name: str = None):
        """
        关闭连接

        Args:
            server_name: Server名称，None表示关闭所有连接
        """
        if server_name:
            # 关闭指定Server
            if server_name in self.sessions:
                try:
                    await self.sessions[server_name].close()
                    del self.sessions[server_name]
                    print(f"[OK] 已关闭 '{server_name}'")
                except Exception as e:
                    print(f"[ERROR] 关闭 '{server_name}' 失败: {e}")
        else:
            # 关闭所有Server
            for name in list(self.sessions.keys()):
                await self.close(name)

    def is_connected(self, server_name: str) -> bool:
        """
        检查Server是否已连接

        Args:
            server_name: Server名称

        Returns:
            bool: 是否已连接
        """
        return server_name in self.sessions

    def get_connected_servers(self) -> List[str]:
        """
        获取所有已连接的Server名称

        Returns:
            List[str]: Server名称列表
        """
        return list(self.sessions.keys())

