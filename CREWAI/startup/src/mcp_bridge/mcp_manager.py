"""
=============================================================================
MCP 管理器
文件: src/mcp_bridge/mcp_manager.py
说明: 管理MCP Server的连接、调用和生命周期
=============================================================================
"""

import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .mcp_client import MCPClient


@dataclass
class MCPServerInfo:
    """MCP Server信息"""
    name: str
    description: str
    command: str
    args: List[str]
    capabilities: List[Dict[str, Any]]


class MCPManager:
    """
    MCP管理器

    职责:
    1. 加载MCP Server注册表
    2. 使用MCPClient管理Server连接
    3. 提供统一的调用接口
    4. 根据关键词匹配合适的Server和工具
    """

    def __init__(self, registry_path: str):
        """
        初始化MCP管理器

        Args:
            registry_path: MCP Server注册表配置文件路径
        """
        self.registry_path = registry_path
        self.servers: Dict[str, MCPServerInfo] = {}
        self.client = MCPClient()  # 使用独立的MCPClient

        # 加载注册表
        self._load_registry()

    def _load_registry(self):
        """加载MCP Server注册表"""
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        for server_config in config['servers']:
            server_info = MCPServerInfo(
                name=server_config['name'],
                description=server_config['description'],
                command=server_config['command'],
                args=server_config['args'],
                capabilities=server_config['capabilities']
            )
            self.servers[server_info.name] = server_info

        print(f"[OK] 已加载 {len(self.servers)} 个MCP Server")

    async def connect_server(self, server_name: str) -> bool:
        """
        连接到指定的MCP Server

        Args:
            server_name: Server名称

        Returns:
            bool: 连接是否成功
        """
        if server_name not in self.servers:
            print(f"[ERROR] Server '{server_name}' 不存在")
            return False

        server_info = self.servers[server_name]
        return await self.client.connect(
            server_name,
            server_info.command,
            server_info.args
        )

    async def connect_all_servers(self):
        """连接所有注册的Server"""
        print(f"\n正在连接 {len(self.servers)} 个MCP Server...")
        for server_name in self.servers.keys():
            await self.connect_server(server_name)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[str]:
        """
        调用指定Server的工具

        Args:
            server_name: Server名称
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            Optional[str]: 工具执行结果，失败返回None
        """
        # 确保Server已连接
        if not self.client.is_connected(server_name):
            success = await self.connect_server(server_name)
            if not success:
                return None

        return await self.client.call_tool(server_name, tool_name, arguments)

    def find_tool_by_keywords(self, keywords: str) -> Optional[Dict[str, Any]]:
        """
        根据关键词查找合适的工具

        Args:
            keywords: 关键词字符串

        Returns:
            Optional[Dict]: 包含server_name, tool_name, capability的字典
        """
        keywords_lower = keywords.lower()

        for server_name, server_info in self.servers.items():
            for capability in server_info.capabilities:
                # 检查关键词是否匹配
                for keyword in capability.get('keywords', []):
                    if keyword in keywords_lower:
                        return {
                            'server_name': server_name,
                            'tool_name': capability['name'],
                            'capability': capability,
                            'matched_keyword': keyword
                        }
        return None

    def get_all_capabilities(self) -> List[Dict[str, Any]]:
        """
        获取所有Server的能力列表

        Returns:
            List[Dict]: 所有能力的列表
        """
        capabilities = []
        for server_name, server_info in self.servers.items():
            for capability in server_info.capabilities:
                capabilities.append({
                    'server_name': server_name,
                    'server_description': server_info.description,
                    'tool_name': capability['name'],
                    'tool_description': capability['description'],
                    'keywords': capability.get('keywords', [])
                })
        return capabilities

    async def close_all(self):
        """关闭所有连接"""
        await self.client.close()

