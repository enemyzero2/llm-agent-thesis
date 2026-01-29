"""
=============================================================================
视频流控制演示 - 启动脚本
文件: start_demo.py
说明: 跨平台启动脚本，用于演示OBS MCP控制系统
=============================================================================
"""

import os
import sys
import subprocess
import webbrowser
import time
import platform


def check_obs_running():
    """检查OBS是否正在运行"""
    system = platform.system()

    if system == "Windows":
        result = subprocess.run(
            ["tasklist"],
            capture_output=True,
            text=True
        )
        return "obs64.exe" in result.stdout or "obs32.exe" in result.stdout

    elif system == "Darwin":  # macOS
        result = subprocess.run(
            ["pgrep", "-x", "obs"],
            capture_output=True
        )
        return result.returncode == 0

    elif system == "Linux":
        result = subprocess.run(
            ["pgrep", "-x", "obs"],
            capture_output=True
        )
        return result.returncode == 0

    return False


def main():
    print("=" * 60)
    print("视频流控制演示 - 启动脚本")
    print("=" * 60)
    print()

    # 检查OBS
    print("[1] 检查 OBS Studio 是否运行...")
    if check_obs_running():
        print("✓ OBS Studio 正在运行")
    else:
        print("✗ OBS Studio 未运行")
        print()
        print("请先启动 OBS Studio 并启用 WebSocket 服务器:")
        print("  工具 → WebSocket 服务器设置 → 启用")
        print()
        input("按回车键退出...")
        return

    # 打开前端
    print()
    print("[2] 打开视频播放器前端...")
    html_path = os.path.join(
        os.path.dirname(__file__),
        "src", "video_frontend", "player.html"
    )

    if os.path.exists(html_path):
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
        print("✓ 已在浏览器中打开")
    else:
        print(f"✗ 找不到文件: {html_path}")

    # 等待浏览器打开
    time.sleep(2)

    # 运行测试
    print()
    print("[3] 运行 MCP Server 测试...")
    test_script = os.path.join(
        os.path.dirname(__file__),
        "src", "mcp_servers", "test_obs_server.py"
    )

    if os.path.exists(test_script):
        subprocess.run([sys.executable, test_script])
    else:
        print(f"✗ 找不到测试脚本: {test_script}")

    print()
    print("=" * 60)
    print("演示完成")
    print("=" * 60)
    input("按回车键退出...")


if __name__ == "__main__":
    main()
