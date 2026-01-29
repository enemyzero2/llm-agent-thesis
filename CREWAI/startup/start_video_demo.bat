@echo off
chcp 65001 >nul
echo ========================================
echo 视频流控制演示 - 启动脚本
echo ========================================
echo.

echo [1] 检查 OBS Studio 是否运行...
tasklist /FI "IMAGENAME eq obs64.exe" 2>NUL | find /I /N "obs64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo ✓ OBS Studio 正在运行
) else (
    echo ✗ OBS Studio 未运行
    echo.
    echo 请先启动 OBS Studio 并启用 WebSocket 服务器
    echo 工具 → WebSocket 服务器设置 → 启用
    pause
    exit
)

echo.
echo [2] 打开视频播放器前端...
start "" "src\video_frontend\player.html"

echo.
echo [3] 运行 MCP Server 测试...
python src\mcp_servers\test_obs_server.py

echo.
echo ========================================
echo 演示完成
echo ========================================
pause
