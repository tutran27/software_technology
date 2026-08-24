/**
 * 🚁 UAV Traffic Intelligent Operations Center - Modern Frontend Engine
 */

let appState = {
    isRunning: false,
    isPaused: false,
    historyChart: null,
    donutChart: null,
    analyticsTrendChart: null,
    pollInterval: null,
    cachedProfileText: ""
};

function switchTab(targetId) {
    const tabs = ['tab-live', 'tab-profile', 'tab-analytics'];
    tabs.forEach(id => {
        const el = document.getElementById(id);
        const btn = document.getElementById('tabBtn-' + id.replace('tab-', ''));
        if (el) el.classList.remove('active');
        if (btn) btn.classList.remove('active');
    });

    const activeEl = document.getElementById(targetId);
    const activeBtn = document.getElementById('tabBtn-' + targetId.replace('tab-', ''));
    if (activeEl) activeEl.classList.add('active');
    if (activeBtn) activeBtn.classList.add('active');

    if (targetId === 'tab-analytics') {
        if (appState.donutChart) appState.donutChart.resize();
        if (appState.analyticsTrendChart) appState.analyticsTrendChart.resize();
    } else if (targetId === 'tab-live' && appState.historyChart) {
        appState.historyChart.resize();
    }
}

function initCharts() {
    try {
        const ctxMini = document.getElementById('miniTrendChart');
        if (ctxMini && typeof Chart !== 'undefined') {
            appState.historyChart = new Chart(ctxMini.getContext('2d'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Chỉ số CI',
                            data: [],
                            borderColor: '#f97316',
                            backgroundColor: 'rgba(249, 115, 22, 0.15)',
                            fill: true,
                            tension: 0.3,
                            borderWidth: 2,
                            pointRadius: 2
                        },
                        {
                            label: 'Chiếm dụng OCR (%)',
                            data: [],
                            borderColor: '#38bdf8',
                            borderDash: [3, 3],
                            borderWidth: 1.6,
                            fill: false,
                            tension: 0.3,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top', align: 'end', labels: { color: '#cbd5e1', font: { size: 10 }, boxWidth: 10, padding: 4 } }
                    },
                    scales: {
                        x: { display: false },
                        y: { min: 0, max: 100, grid: { color: 'rgba(51, 65, 85, 0.3)' }, ticks: { color: '#94a3b8', font: { size: 9 } } }
                    }
                }
            });
        }

        const ctxDonut = document.getElementById('donutChart');
        if (ctxDonut && typeof Chart !== 'undefined') {
            appState.donutChart = new Chart(ctxDonut.getContext('2d'), {
                type: 'doughnut',
                data: {
                    labels: ['Ô tô', 'Xe máy', 'Xe buýt', 'Xe tải'],
                    datasets: [{
                        data: [1, 1, 0, 0],
                        backgroundColor: ['#38bdf8', '#fbbf24', '#f87171', '#c084fc'],
                        borderColor: '#0f172a',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom', labels: { color: '#f8fafc', font: { size: 11 }, padding: 8 } } }
                }
            });
        }

        const ctxAnalytics = document.getElementById('analyticsTrendChart');
        if (ctxAnalytics && typeof Chart !== 'undefined') {
            appState.analyticsTrendChart = new Chart(ctxAnalytics.getContext('2d'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Chỉ số Tắc Nghẽn (CI)', data: [], borderColor: '#f97316', backgroundColor: 'rgba(249, 115, 22, 0.15)', fill: true, tension: 0.3, borderWidth: 2 },
                        { label: 'Chiếm Dụng (OCR %)', data: [], borderColor: '#38bdf8', tension: 0.3, borderWidth: 2 },
                        { label: 'Vận Tốc TB (km/h)', data: [], borderColor: '#10b981', tension: 0.3, borderWidth: 1.8 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#f8fafc', font: { size: 11 } } } },
                    scales: {
                        x: { grid: { color: 'rgba(51, 65, 85, 0.3)' }, ticks: { color: '#94a3b8', font: { size: 10 } } },
                        y: { grid: { color: 'rgba(51, 65, 85, 0.3)' }, ticks: { color: '#94a3b8', font: { size: 10 } } }
                    }
                }
            });
        }
    } catch(e) { console.warn('Chart init skipped:', e); }
}

async function sendControl(action) {
    try {
        const res = await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action })
        });
        const data = await res.json();
        updateButtonStates(data.is_running, data.is_paused);
    } catch (err) {
        console.error('Control error:', err);
    }
}

function updateButtonStates(isRunning, isPaused) {
    appState.isRunning = isRunning;
    appState.isPaused = isPaused;

    const btnRun = document.getElementById('btnRun');
    const btnPause = document.getElementById('btnPause');
    const btnResume = document.getElementById('btnResume');
    const statusBeacon = document.getElementById('statusBeacon');
    const statusText = document.getElementById('statusText');

    if (!btnRun || !btnPause || !btnResume || !statusBeacon || !statusText) return;

    if (isRunning) {
        btnRun.style.display = 'none';
        btnPause.style.display = 'inline-flex';
        btnResume.style.display = 'none';
        statusBeacon.className = 'status-beacon active';
        statusText.innerText = 'ĐANG PHÁT LUỒNG';
        statusText.style.color = '#10b981';
    } else if (isPaused) {
        btnRun.style.display = 'none';
        btnPause.style.display = 'none';
        btnResume.style.display = 'inline-flex';
        statusBeacon.className = 'status-beacon paused';
        statusText.innerText = 'TẠM DỪNG';
        statusText.style.color = '#f59e0b';
    } else {
        btnRun.style.display = 'inline-flex';
        btnPause.style.display = 'none';
        btnResume.style.display = 'none';
        statusBeacon.className = 'status-beacon';
        statusText.innerText = 'SẴN SÀNG';
        statusText.style.color = '#94a3b8';
    }
}

async function updateConfig(configObj) {
    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(configObj)
        });
        const data = await res.json();
        if (data && data.video_meta) {
            const srcEl = document.getElementById('hudSource');
            const resEl = document.getElementById('hudRes');
            if (srcEl && data.video_meta.name) srcEl.innerText = data.video_meta.name;
            if (resEl && data.video_meta.resolution) resEl.innerText = data.video_meta.resolution;
        }
        const videoEl = document.getElementById('videoStream');
        if (videoEl) {
            videoEl.src = '/video_feed?t=' + Date.now();
        }
    } catch (err) {
        console.error('Config update error:', err);
    }
}

async function uploadVideo(e) {
    const file = e.target.files[0];
    if (!file) return;

    const statusText = document.getElementById('statusText');
    if (statusText) {
        statusText.innerText = 'ĐANG TẢI LÊN...';
        statusText.style.color = '#38bdf8';
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/upload_video', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        const srcEl = document.getElementById('hudSource');
        const resEl = document.getElementById('hudRes');
        if (srcEl && data.filename) srcEl.innerText = data.filename;
        if (resEl && data.video_meta && data.video_meta.resolution) resEl.innerText = data.video_meta.resolution;

        if (statusText) {
            statusText.innerText = 'ĐÃ TẢI XONG';
            statusText.style.color = '#10b981';
        }

        const videoEl = document.getElementById('videoStream');
        if (videoEl) {
            videoEl.src = '/video_feed?t=' + Date.now();
        }
        updateButtonStates(true, false);
    } catch (err) {
        if (statusText) {
            statusText.innerText = 'LỖI TẢI VIDEO';
            statusText.style.color = '#f43f5e';
        }
    }
}

function startMetricsPolling() {
    appState.pollInterval = setInterval(async () => {
        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();
            renderMetrics(data);
        } catch (err) {}
    }, 350);
}

function renderMetrics(data) {
    const { is_running, is_paused, stats, counts, line_counts, alerts, history, video_meta } = data;

    if (appState.isRunning !== is_running || appState.isPaused !== is_paused) {
        updateButtonStates(is_running, is_paused);
    }

    const setTxt = (id, txt) => {
        const el = document.getElementById(id);
        if (el) el.innerText = txt;
    };

    setTxt('hudSource', video_meta?.name || '');
    setTxt('hudFps', (stats?.fps || 30.0).toFixed(1) + ' FPS');
    setTxt('hudRes', video_meta?.resolution || '1920x1080');

    setTxt('cntCar', (counts?.car || 0).toLocaleString());
    setTxt('lineCar', line_counts?.car ? `(Qua: ${line_counts.car})` : '');

    setTxt('cntMotor', (counts?.motorcycle || 0).toLocaleString());
    setTxt('lineMotor', line_counts?.motorcycle ? `(Qua: ${line_counts.motorcycle})` : '');

    setTxt('cntBus', (counts?.bus || 0).toLocaleString());
    setTxt('lineBus', line_counts?.bus ? `(Qua: ${line_counts.bus})` : '');

    setTxt('cntTruck', (counts?.truck || 0).toLocaleString());
    setTxt('lineTruck', line_counts?.truck ? `(Qua: ${line_counts.truck})` : '');

    setTxt('valOcr', (stats?.occupancy_rate || 0.0).toFixed(1) + '%');
    setTxt('valSpeed', (stats?.avg_speed_kmh || stats?.avg_speed || 0.0).toFixed(1) + ' km/h');
    setTxt('valStopped', (stats?.stopped_ratio || 0.0).toFixed(0) + '%');

    const ci = stats?.congestion_index || 0.0;
    const ciBanner = document.getElementById('ciBanner');
    setTxt('valCi', ci.toFixed(0) + '/100');

    if (ciBanner) {
        if (ci < 30) {
            ciBanner.className = 'ci-pill-banner free';
            setTxt('lblCiState', '🟢 THÔNG THOÁNG');
        } else if (ci < 60) {
            ciBanner.className = 'ci-pill-banner normal';
            setTxt('lblCiState', '🟡 BÌNH THƯỜNG');
        } else if (ci < 80) {
            ciBanner.className = 'ci-pill-banner crowded';
            setTxt('lblCiState', '🟠 ĐÔNG ĐÚC');
        } else {
            ciBanner.className = 'ci-pill-banner severe';
            setTxt('lblCiState', '🔴 ÙN TẮC');
        }
    }

    const alertBox = document.getElementById('alertMessages');
    if (alertBox) {
        if (alerts && alerts.length > 0) {
            alertBox.innerHTML = alerts.map(a => `<div style="color:#fb7185; font-weight:700;">⚠️ ${a.message}</div>`).join('');
        } else {
            alertBox.innerHTML = '🟢 <b>Giao thông ổn định:</b> Không có cảnh báo bất thường.';
        }
    }

    if (history && history.length > 0 && appState.historyChart) {
        appState.historyChart.data.labels = history.map(h => h.time);
        appState.historyChart.data.datasets[0].data = history.map(h => h.congestion_index);
        appState.historyChart.data.datasets[1].data = history.map(h => h.occupancy_rate);
        appState.historyChart.update('none');

        if (appState.donutChart) {
            appState.donutChart.data.datasets[0].data = [counts?.car || 0, counts?.motorcycle || 0, counts?.bus || 0, counts?.truck || 0];
            appState.donutChart.update('none');
        }

        if (appState.analyticsTrendChart) {
            appState.analyticsTrendChart.data.labels = history.map(h => h.time);
            appState.analyticsTrendChart.data.datasets[0].data = history.map(h => h.congestion_index);
            appState.analyticsTrendChart.data.datasets[1].data = history.map(h => h.occupancy_rate);
            appState.analyticsTrendChart.data.datasets[2].data = history.map(h => h.avg_speed);
            appState.analyticsTrendChart.update('none');
        }

        const totalCounted = stats?.total_counted || ((counts?.car||0) + (counts?.motorcycle||0) + (counts?.bus||0) + (counts?.truck||0));
        setTxt('totalVehiclesSeen', totalCounted.toLocaleString());
        setTxt('peakCi', Math.max(...history.map(h => h.congestion_index)).toFixed(0) + '/100');
    }

    const inputV = document.getElementById('llmVehicles');
    if (inputV && !inputV.matches(':focus')) {
        inputV.value = stats?.vehicle_count || 0;
        const ocrIn = document.getElementById('llmOcr');
        const spdIn = document.getElementById('llmSpeed');
        const stpIn = document.getElementById('llmStopped');
        const ciIn = document.getElementById('llmCi');
        if (ocrIn && !ocrIn.matches(':focus')) ocrIn.value = (stats?.occupancy_rate || 0).toFixed(1);
        if (spdIn && !spdIn.matches(':focus')) spdIn.value = (stats?.avg_speed_kmh || stats?.avg_speed || 0).toFixed(1);
        if (stpIn && !stpIn.matches(':focus')) stpIn.value = (stats?.stopped_ratio || 0).toFixed(0);
        if (ciIn && !ciIn.matches(':focus')) ciIn.value = (stats?.congestion_index || 0).toFixed(0);
    }
}

async function requestLLMProfile() {
    const btn = document.getElementById('btnGenerateLLM');
    if (!btn) return;
    btn.innerText = '⚡ ĐANG SINH KHUYẾN NGHỊ (openai/gpt-oss-20b)...';
    btn.disabled = true;

    const payload = {
        vehicles: parseInt(document.getElementById('llmVehicles')?.value) || 0,
        ocr: parseFloat(document.getElementById('llmOcr')?.value) || 0.0,
        avg_speed: parseFloat(document.getElementById('llmSpeed')?.value) || 0.0,
        stopped_ratio: parseFloat(document.getElementById('llmStopped')?.value) || 0.0,
        congestion_index: parseFloat(document.getElementById('llmCi')?.value) || 0.0,
        state_text: document.getElementById('lblCiState')?.innerText || 'Bình thường'
    };

    try {
        const res = await fetch('/api/llm_profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        appState.cachedProfileText = data.profile;
        const resBox = document.getElementById('llmResultBox');
        if (resBox) {
            resBox.innerHTML = `<pre style="white-space:pre-wrap; font-family:'JetBrains Mono', monospace; font-size:12.5px; color:#e2e8f0; line-height:1.6;">${data.profile}</pre>`;
        }
        const actionRow = document.getElementById('profileActionRow');
        if (actionRow) actionRow.style.display = 'flex';
    } catch (err) {
        alert('Lỗi gọi LLM: ' + err);
    } finally {
        btn.innerText = '🚀 SINH KHUYẾN NGHỊ ĐÈN (openai/gpt-oss-20b)';
        btn.disabled = false;
    }
}

function copyProfileText() {
    if (appState.cachedProfileText) {
        navigator.clipboard.writeText(appState.cachedProfileText);
        const btn = document.getElementById('btnCopyProfile');
        if (btn) {
            btn.innerText = '✅ ĐÃ SAO CHÉP!';
            setTimeout(() => { btn.innerText = '📋 SAO CHÉP'; }, 1800);
        }
    }
}

function downloadProfileMd() {
    if (appState.cachedProfileText) {
        const blob = new Blob([appState.cachedProfileText], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'traffic_control_profile.md';
        a.click();
        URL.revokeObjectURL(url);
    }
}
