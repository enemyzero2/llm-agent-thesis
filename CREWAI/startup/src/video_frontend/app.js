// OBS控制台 - 前端脚本
const API = 'http://localhost:5000/api';

// 启动虚拟摄像头视频
async function startVideo() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const obsCamera = devices.find(d =>
            d.kind === 'videoinput' && d.label.includes('OBS')
        );

        const constraints = {
            video: obsCamera ? {deviceId: obsCamera.deviceId} : true,
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        document.getElementById('videoPreview').srcObject = stream;
        document.getElementById('startVideo').classList.add('hidden');
        updateStatus('视频已启动');
    } catch (e) {
        updateStatus('启动视频失败: ' + e.message);
    }
}

// 绑定按钮事件
document.getElementById('startVideo').onclick = startVideo;

// 加载场景列表
async function loadScenes() {
    try {
        const res = await fetch(`${API}/scenes`);
        const data = await res.json();

        const container = document.getElementById('scenes');
        container.innerHTML = data.scenes.map(scene => `
            <button class="scene-btn ${scene === data.current ? 'active' : ''}"
                    onclick="switchScene('${scene}')">
                ${scene}
            </button>
        `).join('');

        updateStatus('场景加载成功');
    } catch (e) {
        updateStatus('加载场景失败: ' + e.message);
    }
}

// 切换场景
async function switchScene(name) {
    try {
        await fetch(`${API}/scenes/switch`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({scene: name})
        });
        loadScenes();
        updateStatus(`已切换到: ${name}`);
    } catch (e) {
        updateStatus('切换失败: ' + e.message);
    }
}

// 加载音频源
async function loadAudio() {
    try {
        const res = await fetch(`${API}/audio/sources`);
        const data = await res.json();

        const container = document.getElementById('audio');
        if (data.sources.length === 0) {
            container.innerHTML = '<p>没有找到音频源</p>';
            return;
        }

        container.innerHTML = data.sources.map(src => `
            <div class="audio-item">
                <span class="audio-name">${src.name}</span>
                <input type="range" min="0" max="100" value="${src.volume}"
                       onchange="setVolume('${src.name}', this.value)">
                <span class="audio-value">${src.volume}%</span>
            </div>
        `).join('');
    } catch (e) {
        updateStatus('加载音频失败: ' + e.message);
    }
}

// 设置音量
async function setVolume(source, volume) {
    try {
        await fetch(`${API}/audio/volume`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({source, volume: parseInt(volume)})
        });
        updateStatus(`${source} 音量: ${volume}%`);
    } catch (e) {
        updateStatus('设置音量失败: ' + e.message);
    }
}

// 更新状态
function updateStatus(msg) {
    document.getElementById('status').textContent = msg;
}

// 初始化
loadScenes();
loadAudio();
setInterval(() => { loadScenes(); loadAudio(); }, 5000);
