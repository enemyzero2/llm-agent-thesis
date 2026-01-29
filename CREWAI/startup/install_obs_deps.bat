@echo off
chcp 65001 >nul
echo ========================================
echo 安装 OBS MCP Server 依赖
echo ========================================
echo.

echo [1] 安装 obs-websocket-py...
pip install obs-websocket-py

echo.
echo [2] 安装 Flask (可选，用于Web后端)...
pip install flask flask-cors

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 下一步：
echo 1. 启动 OBS Studio
echo 2. 启用 WebSocket 服务器（工具 → WebSocket 服务器设置）
echo 3. 运行 start_demo.py 或 start_video_demo.bat
echo.
pause
