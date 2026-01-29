# -*- coding: utf-8 -*-
"""
视频控制后端服务
提供HTTP API连接前端和OBS
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

# OBS连接
from obswebsocket import obsws, requests as obs_requests

app = Flask(__name__, static_folder='../video_frontend')
CORS(app)

# OBS配置
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "123456"

# OBS客户端
obs_client = None


def get_obs():
    """获取OBS连接"""
    global obs_client
    if obs_client is None:
        obs_client = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
        obs_client.connect()
    return obs_client


@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    """静态文件"""
    return send_from_directory(app.static_folder, filename)


# ============ 场景API ============

@app.route('/api/scenes', methods=['GET'])
def get_scenes():
    """获取所有场景"""
    try:
        ws = get_obs()
        result = ws.call(obs_requests.GetSceneList())
        scenes = [s['sceneName'] for s in result.getScenes()]
        current = result.getCurrentProgramSceneName()
        return jsonify({
            "scenes": scenes,
            "current": current
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/scenes/switch', methods=['POST'])
def switch_scene():
    """切换场景"""
    try:
        data = request.json
        scene_name = data.get('scene')
        ws = get_obs()
        ws.call(obs_requests.SetCurrentProgramScene(sceneName=scene_name))
        return jsonify({"success": True, "scene": scene_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ 音量API ============

@app.route('/api/audio/sources', methods=['GET'])
def get_audio_sources():
    """获取所有音频源"""
    try:
        ws = get_obs()
        inputs = ws.call(obs_requests.GetInputList())
        audio_sources = []
        for inp in inputs.getInputs():
            try:
                vol = ws.call(obs_requests.GetInputVolume(inputName=inp['inputName']))
                audio_sources.append({
                    "name": inp['inputName'],
                    "volume": int(vol.getInputVolumeMul() * 100)
                })
            except:
                pass
        return jsonify({"sources": audio_sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audio/volume', methods=['POST'])
def set_volume():
    """设置音量"""
    try:
        data = request.json
        source = data.get('source')
        volume = data.get('volume', 50)
        ws = get_obs()
        ws.call(obs_requests.SetInputVolume(
            inputName=source,
            inputVolumeMul=volume / 100.0
        ))
        return jsonify({"success": True, "source": source, "volume": volume})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ 启动服务器 ============

if __name__ == '__main__':
    print("=" * 50)
    print("视频控制后端服务")
    print("访问地址: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
