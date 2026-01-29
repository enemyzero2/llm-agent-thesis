# -*- coding: utf-8 -*-
"""简单的OBS连接测试"""
import sys

try:
    from obswebsocket import obsws, requests as obs_requests
    print("[OK] obs-websocket-py 已导入")
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    sys.exit(1)

# 连接配置
HOST = "localhost"
PORT = 4455
PASSWORD = "123456"

print(f"\n尝试连接到 OBS...")
print(f"  主机: {HOST}")
print(f"  端口: {PORT}")
print(f"  密码: {'*' * len(PASSWORD)}")

try:
    ws = obsws(HOST, PORT, PASSWORD)
    ws.connect()
    print("\n[OK] 连接成功!")

    # 获取版本信息
    version = ws.call(obs_requests.GetVersion())
    print(f"\nOBS 版本信息:")
    print(f"  OBS Studio: {version.getObsVersion()}")
    print(f"  WebSocket: {version.getObsWebSocketVersion()}")

    ws.disconnect()
    print("\n[OK] 测试完成")

except Exception as e:
    print(f"\n[ERROR] 连接失败: {e}")
    sys.exit(1)
