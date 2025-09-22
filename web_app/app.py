import cv2
import mediapipe as mp
import time
import threading
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
EAR_THRESHOLD = 0.23
CONSEC_FRAMES = 3
MIN_BLINKS_PER_MIN = 12
FATIGUE_TIME = 15  # seconds

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = None
cam_index = 0
running = False

blink_count = 0
counter = 0
fatigue_start = None
fatigue_alerted = False
eye_strain_alerted = False
user_acknowledged_strain = False

user_acknowledged_fatigue = False  # Added for fatigue alert

start_time = None
paused = False
pause_start_time = None
paused_time_accumulator = 0
last_minute_check = 0
minute_blink_start = 0

logs = []

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def calculate_ear(eye_points, landmarks):
    A = euclidean_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    B = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    C = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
    return (A + B) / (2.0 * C)

def get_available_cameras(max_tested=5):
    cams = []
    for i in range(max_tested):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            cams.append(i)
            test_cap.release()
    return cams if cams else [0]

def generate_frames():
    global cap, running
    global blink_count, counter, fatigue_start, fatigue_alerted
    global start_time, paused, pause_start_time, paused_time_accumulator
    global last_minute_check, minute_blink_start, logs
    global eye_strain_alerted, user_acknowledged_strain
    global user_acknowledged_fatigue  # Added here

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = mp_face_mesh.process(rgb_frame)

        face_detected = False
        eye_closed = False
        current_time = time.time()

        if results.multi_face_landmarks:
            face_detected = True
            if paused:
                if pause_start_time:
                    paused_time_accumulator += current_time - pause_start_time
                paused = False

            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                left_ear = calculate_ear(LEFT_EYE_IDX, landmarks)
                right_ear = calculate_ear(RIGHT_EYE_IDX, landmarks)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    counter += 1
                    eye_closed = True
                    if fatigue_start is None:
                        fatigue_start = current_time
                    elif (current_time - fatigue_start >= FATIGUE_TIME) and not fatigue_alerted:
                        fatigue_alerted = True
                        user_acknowledged_fatigue = False  # Reset acknowledgment on new fatigue alert
                else:
                    if counter >= CONSEC_FRAMES:
                        blink_count += 1
                    counter = 0
                    fatigue_start = None
                    fatigue_alerted = False
        else:
            if not paused:
                paused = True
                pause_start_time = current_time

        if paused:
            elapsed_time = pause_start_time - start_time - paused_time_accumulator
        else:
            elapsed_time = current_time - start_time - paused_time_accumulator

        # Log every minute
        if elapsed_time - last_minute_check >= 60:
            start_interval = last_minute_check
            end_interval = last_minute_check + 60
            last_minute_check += 60

            blinks_this_min = blink_count - minute_blink_start
            strain = blinks_this_min < MIN_BLINKS_PER_MIN

            logs.append({
                'start': start_interval,
                'end': end_interval,
                'blink_count': blinks_this_min,
                'strain': strain
            })

            minute_blink_start = blink_count
            eye_strain_alerted = strain
            user_acknowledged_strain = False  # reset acknowledgment each minute

        overlay_texts = [
            f"Time: {int(elapsed_time)//3600:02d}:{(int(elapsed_time)%3600)//60:02d}:{int(elapsed_time)%60:02d}",
            f"Blinks: {blink_count}",
            f"Eye Closed: {'Yes' if eye_closed else 'No'}",
            f"Face Detected: {'Yes' if face_detected else 'No'}"
        ]

        if paused:
            cv2.putText(frame, "Paused - Face Not Detected", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            for i, text in enumerate(overlay_texts):
                cv2.putText(frame, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if cap:
        cap.release()

@app.route('/')
def index():
    cameras = get_available_cameras()
    return render_template('index.html', cameras=cameras)

@app.route('/start', methods=['POST'])
def start():
    global cap, cam_index, running
    global blink_count, counter, fatigue_start, fatigue_alerted
    global start_time, paused, pause_start_time, paused_time_accumulator
    global last_minute_check, minute_blink_start, logs
    global eye_strain_alerted, user_acknowledged_strain
    global user_acknowledged_fatigue  # Added here

    if running:
        return jsonify({'status': 'already_running'})

    cam_index = int(request.json.get('camera', 0))
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return jsonify({'status': 'error', 'message': f'Cannot open camera {cam_index}'})

    running = True
    blink_count = 0
    counter = 0
    fatigue_start = None
    fatigue_alerted = False
    user_acknowledged_fatigue = False  # Reset on start
    start_time = time.time()
    paused = False
    pause_start_time = None
    paused_time_accumulator = 0
    last_minute_check = 0
    minute_blink_start = 0
    logs = []
    eye_strain_alerted = False
    user_acknowledged_strain = False

    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None
    return jsonify({'status': 'stopped'})

@app.route('/video_feed')
def video_feed():
    if not running or cap is None:
        return "No active camera feed", 404
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_logs')
def get_logs():
    global logs
    formatted_logs = []
    for entry in logs:
        start_str = time.strftime("%H:%M:%S", time.gmtime(entry['start']))
        end_str = time.strftime("%H:%M:%S", time.gmtime(entry['end']))
        status = "Eye strain detected" if entry['strain'] else "No eye strain"
        formatted_logs.append(f"{start_str} - {end_str}: {status} (Blinks: {entry['blink_count']})")
    return jsonify(formatted_logs)

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    global logs
    logs = []
    return jsonify({'status': 'cleared'})

@app.route('/should_alert_strain')
def should_alert_strain():
    global eye_strain_alerted, user_acknowledged_strain
    if eye_strain_alerted and not user_acknowledged_strain:
        return jsonify({'alert': True})
    return jsonify({'alert': False})

@app.route('/acknowledge_strain', methods=['POST'])
def acknowledge_strain():
    global user_acknowledged_strain
    user_acknowledged_strain = True
    return jsonify({'status': 'acknowledged'})

# === Added for fatigue alert ===
@app.route('/should_alert_fatigue')
def should_alert_fatigue():
    global fatigue_alerted, user_acknowledged_fatigue
    if fatigue_alerted and not user_acknowledged_fatigue:
        return jsonify({'alert': True})
    return jsonify({'alert': False})

@app.route('/acknowledge_fatigue', methods=['POST'])
def acknowledge_fatigue():
    global user_acknowledged_fatigue
    user_acknowledged_fatigue = True
    return jsonify({'status': 'acknowledged'})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)





