// 视频播放器控制脚本

const videoPlayer = document.getElementById('videoPlayer');
const volumeSlider = document.getElementById('volumeSlider');
const brightnessSlider = document.getElementById('brightnessSlider');
const contrastSlider = document.getElementById('contrastSlider');
const volumeValue = document.getElementById('volumeValue');
const brightnessValue = document.getElementById('brightnessValue');
const contrastValue = document.getElementById('contrastValue');
const status = document.getElementById('status');

// 音量控制
volumeSlider.addEventListener('input', (e) => {
    const volume = e.target.value;
    videoPlayer.volume = volume / 100;
    volumeValue.textContent = `${volume}%`;
    updateStatus(`音量已设置为 ${volume}%`);
});

// 亮度控制
brightnessSlider.addEventListener('input', (e) => {
    const brightness = e.target.value;
    brightnessValue.textContent = `${brightness}%`;
    applyFilters();
    updateStatus(`亮度已设置为 ${brightness}%`);
});

// 对比度控制
contrastSlider.addEventListener('input', (e) => {
    const contrast = e.target.value;
    contrastValue.textContent = `${contrast}%`;
    applyFilters();
    updateStatus(`对比度已设置为 ${contrast}%`);
});

// 应用视频滤镜
function applyFilters() {
    const brightness = brightnessSlider.value;
    const contrast = contrastSlider.value;
    videoPlayer.style.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
}

// 更新状态显示
function updateStatus(message) {
    status.textContent = `✅ ${message}`;
    status.style.background = '#e8f5e9';
    status.style.borderColor = '#4caf50';
}

// 初始化
applyFilters();
console.log('视频播放器已初始化');
