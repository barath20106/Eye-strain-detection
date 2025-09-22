let streamStarted = false;
let strainAlertShown = false;
let fatigueAlertShown = false;

document.getElementById('startBtn').addEventListener('click', () => {
    const selectedCam = document.getElementById('cameraSelect').value;

    if (!streamStarted) {
        fetch('/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera: selectedCam })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'started' || data.status === 'already_running') {
                document.getElementById('videoFeed').src = '/video_feed';
                streamStarted = true;
                document.getElementById('startBtn').textContent = 'Stop';
            } else {
                alert('Error: ' + data.message);
            }
        });
    } else {
        fetch('/stop', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'stopped') {
                document.getElementById('videoFeed').src = '';
                streamStarted = false;
                document.getElementById('startBtn').textContent = 'Start';
                strainAlertShown = false;
                fatigueAlertShown = false;
            }
        });
    }
});

document.getElementById('clearLogsBtn').addEventListener('click', () => {
    fetch('/clear_logs', { method: 'POST' })
        .then(res => res.json())
        .then(() => {
            document.getElementById('logContent').textContent = '';
        });
});

function fetchLogs() {
    fetch('/get_logs')
        .then(res => res.json())
        .then(data => {
            const logContent = document.getElementById('logContent');
            logContent.textContent = data.join('\n');
        });
}

function checkForStrainAlert() {
    fetch('/should_alert_strain')
        .then(res => res.json())
        .then(data => {
            if (data.alert && !strainAlertShown) {
                strainAlertShown = true;
                playBeep();
                alert("Eye strain detected: Blink count low in the last minute.");
                fetch('/acknowledge_strain', { method: 'POST' });
            } else if (!data.alert) {
                strainAlertShown = false; // Reset flag when no alert
            }
        });
}

function checkForFatigueAlert() {
    fetch('/should_alert_fatigue')
        .then(res => res.json())
        .then(data => {
            if (data.alert && !fatigueAlertShown) {
                fatigueAlertShown = true;
                playBeep();
                alertFatigue(); // Custom function to block until OK
            } else if (!data.alert) {
                fatigueAlertShown = false; // Reset flag when no alert
            }
        });
}

function alertFatigue() {
    // Custom blocking alert that doesn't allow continuation until OK
    alert("Fatigue detected: Eyes have been closed for over 15 seconds.");
    fetch('/acknowledge_fatigue', { method: 'POST' });
}

function playBeep() {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(800, audioCtx.currentTime);
    oscillator.start();

    setTimeout(() => {
        oscillator.stop();
    }, 300);
}

// Poll logs and strain alert every 10 seconds
setInterval(() => {
    fetchLogs();
    checkForStrainAlert();
    checkForFatigueAlert();
}, 10000);

// Fetch logs immediately on load
fetchLogs();

