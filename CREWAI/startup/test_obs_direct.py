# -*- coding: utf-8 -*-
"""
直接测试OBS音量控制（不使用MCP）
"""
from obswebsocket import obsws, requests as obs_requests

# 连接配置
HOST = "localhost"
PORT = 4455
PASSWORD = "123456"

print("=" * 50)
print("OBS 音量控制 - 直接测试")
print("=" * 50)

# 连接OBS
ws = obsws(HOST, PORT, PASSWORD)
ws.connect()
print("[OK] 已连接到OBS")

# 获取所有输入源
print("\n[1] 获取所有输入源...")
inputs = ws.call(obs_requests.GetInputList())
print(f"找到 {len(inputs.getInputs())} 个输入源:")
for inp in inputs.getInputs():
    print(f"  - {inp['inputName']} ({inp['inputKind']})")

# 获取音频源
print("\n[2] 查找音频源...")
audio_sources = []
for inp in inputs.getInputs():
    try:
        vol = ws.call(obs_requests.GetInputVolume(inputName=inp['inputName']))
        audio_sources.append(inp['inputName'])
        print(f"  [音频] {inp['inputName']}: {int(vol.getInputVolumeMul() * 100)}%")
    except:
        pass

if audio_sources:
    # 测试设置音量
    test_source = audio_sources[0]
    print(f"\n[3] 测试设置 '{test_source}' 音量为 80%...")
    ws.call(obs_requests.SetInputVolume(inputName=test_source, inputVolumeMul=0.8))
    print("[OK] 音量已设置")

    # 验证
    vol = ws.call(obs_requests.GetInputVolume(inputName=test_source))
    print(f"[OK] 当前音量: {int(vol.getInputVolumeMul() * 100)}%")

ws.disconnect()
print("\n[OK] 测试完成")
