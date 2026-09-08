import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import json
import random
import threading
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template, send_from_directory

app = Flask(__name__, template_folder='templates')

# --- Simulation State Engine ---
sim_state = {
    "ignition_enabled": True,
    "driver_state": "Active",  # Active, Drowsy, Medical Emergency
    "heart_rate": 72,
    "blood_pressure_sys": 120,
    "blood_pressure_dia": 80,
    "bac": 0.00,  # Blood Alcohol Concentration
}

state_lock = threading.Lock()

def update_telemetry_loop():
    """Background thread to continuously fluctuate telemetry data based on current state."""
    while True:
        with state_lock:
            state = sim_state["driver_state"]
            
            if state == "Active":
                sim_state["heart_rate"] = int(np.clip(sim_state["heart_rate"] + random.choice([-1, 0, 1]), 65, 80))
                sim_state["blood_pressure_sys"] = int(np.clip(sim_state["blood_pressure_sys"] + random.choice([-2, 0, 2]), 115, 125))
                sim_state["blood_pressure_dia"] = int(np.clip(sim_state["blood_pressure_dia"] + random.choice([-1, 0, 1]), 75, 85))
                sim_state["bac"] = max(0.0, round(sim_state["bac"] + random.uniform(-0.005, 0.005), 3))
                
            elif state == "Drowsy":
                sim_state["heart_rate"] = int(np.clip(sim_state["heart_rate"] + random.choice([-1, 0, 1]), 52, 62))
                sim_state["blood_pressure_sys"] = int(np.clip(sim_state["blood_pressure_sys"] + random.choice([-2, 0, 2]), 105, 115))
                sim_state["blood_pressure_dia"] = int(np.clip(sim_state["blood_pressure_dia"] + random.choice([-1, 0, 1]), 65, 75))
                sim_state["bac"] = max(0.0, round(sim_state["bac"] + random.uniform(-0.002, 0.002), 3))
                
            elif state == "Medical Emergency":
                sim_state["heart_rate"] = int(np.clip(sim_state["heart_rate"] + random.choice([-2, 0, 2]), 40, 48))
                sim_state["blood_pressure_sys"] = int(np.clip(sim_state["blood_pressure_sys"] + random.choice([-3, 0, 3]), 85, 95))
                sim_state["blood_pressure_dia"] = int(np.clip(sim_state["blood_pressure_dia"] + random.choice([-2, 0, 2]), 50, 60))
                sim_state["ignition_enabled"] = False
                
            if sim_state["bac"] >= 0.08:
                sim_state["ignition_enabled"] = False
                
        time.sleep(1.0)

# Start background thread
thread = threading.Thread(target=update_telemetry_loop, daemon=True)
thread.start()

# --- Real OpenCV Camera Stream Frame Generator ---
def generate_camera_frames():
    """Generates JPEG frames by reading real camera input and performing OpenCV facial detection."""
    face_cascade = None
    eye_cascade = None
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
        if os.path.exists(eye_path):
            eye_cascade = cv2.CascadeClassifier(eye_path)
    except Exception as e:
        print("OpenCV Cascade init error:", e)

    cap = None
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    except Exception as e:
        print("Camera open exception in Python:", e)

    frame_width = 640
    frame_height = 480
    frame_count = 0

    try:
        while True:
            has_real_frame = False
            frame = None

            if cap and cap.isOpened():
                ret, raw_frame = cap.read()
                if ret and raw_frame is not None:
                    frame = cv2.resize(raw_frame, (frame_width, frame_height))
                    has_real_frame = True

            if not has_real_frame:
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                for i in range(0, frame_width, 40):
                    cv2.line(frame, (i, 0), (i, frame_height), (15, 15, 20), 1)
                for j in range(0, frame_height, 40):
                    cv2.line(frame, (0, j), (frame_width, j), (15, 15, 20), 1)

            with state_lock:
                driver_state = sim_state["driver_state"]
                ignition = sim_state["ignition_enabled"]
                bac = sim_state["bac"]
                hr = sim_state["heart_rate"]

            frame_count += 1
            t = time.time()
            blink_state = (frame_count // 5) % 2 == 0

            cyan = (255, 240, 0)
            green = (0, 255, 100)
            amber = (0, 165, 255)
            red = (50, 50, 255)
            gray = (100, 100, 100)

            if driver_state == "Active":
                hud_color = green if ignition else amber
                status_text = "STATUS: ACTIVE & SAFE" if ignition else "STATUS: SAFETY LOCK ACTIVE"
            elif driver_state == "Drowsy":
                hud_color = amber
                status_text = "WARNING: DROWSINESS DETECTED"
            else:
                hud_color = red if blink_state else gray
                status_text = "CRITICAL: MEDICAL EMERGENCY"

            if has_real_frame and face_cascade is not None:
                gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

                for (fx, fy, fw, fh) in faces:
                    # Draw corner brackets on real face
                    l_len = 20
                    cv2.line(frame, (fx, fy), (fx + l_len, fy), hud_color, 2)
                    cv2.line(frame, (fx, fy), (fx, fy + l_len), hud_color, 2)
                    cv2.line(frame, (fx + fw, fy), (fx + fw - l_len, fy), hud_color, 2)
                    cv2.line(frame, (fx + fw, fy), (fx + fw, fy + l_len), hud_color, 2)
                    cv2.line(frame, (fx, fy + fh), (fx + l_len, fy + fh), hud_color, 2)
                    cv2.line(frame, (fx, fy + fh), (fx, fy + fh - l_len), hud_color, 2)
                    cv2.line(frame, (fx + fw, fy + fh), (fx + fw - l_len, fy + fh), hud_color, 2)
                    cv2.line(frame, (fx + fw, fy + fh), (fx + fw, fy + fh - l_len), hud_color, 2)

                    cv2.putText(frame, "FACE DETECTED", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1)

                    if eye_cascade is not None:
                        roi_gray = gray_img[fy:fy+fh, fx:fx+fw]
                        roi_color = frame[fy:fy+fh, fx:fx+fw]
                        eyes = eye_cascade.detectMultiScale(roi_gray)
                        for (ex, ey, ew, eh) in eyes:
                            cv2.circle(roi_color, (ex + ew//2, ey + eh//2), ew//2, cyan, 1)
                            cv2.circle(roi_color, (ex + ew//2, ey + eh//2), 2, green, -1)
            else:
                face_x, face_y = 320, 240
                if driver_state == "Active":
                    cv2.ellipse(frame, (face_x, face_y), (80, 110), 0, 0, 360, hud_color, 2)
                    cv2.circle(frame, (face_x - 30, face_y - 20), 10, hud_color, 2)
                    cv2.circle(frame, (face_x + 30, face_y - 20), 10, hud_color, 2)
                elif driver_state == "Drowsy":
                    tilt = int(10 * np.sin(t * 2))
                    cv2.ellipse(frame, (face_x, face_y + 10), (80, 110), tilt, 0, 360, hud_color, 2)
                    cv2.line(frame, (face_x - 40 + tilt, face_y - 15), (face_x - 20 + tilt, face_y - 15), hud_color, 3)
                    cv2.line(frame, (face_x + 20 + tilt, face_y - 15), (face_x + 40 + tilt, face_y - 15), hud_color, 3)
                else:
                    cv2.ellipse(frame, (face_x - 40, face_y + 30), (80, 110), -25, 0, 360, hud_color, 2)

            thickness = 2
            l_len = 20
            cv2.line(frame, (10, 10), (10 + l_len, 10), hud_color, thickness)
            cv2.line(frame, (10, 10), (10, 10 + l_len), hud_color, thickness)
            cv2.line(frame, (frame_width-10, 10), (frame_width-10 - l_len, 10), hud_color, thickness)
            cv2.line(frame, (frame_width-10, 10), (frame_width-10, 10 + l_len), hud_color, thickness)

            cv2.putText(frame, "SAFEWARE REAL OPENCV CV", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cyan, 2)
            cv2.putText(frame, status_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)

            ear_val = 0.32 if driver_state == "Active" else (0.08 if driver_state == "Drowsy" else 0.00)
            cv2.putText(frame, f"EAR: {ear_val:.2f}", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.05)
    finally:
        if cap and cap.isOpened():
            cap.release()

# --- Flask Routing ---
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.png')

@app.route('/interstellar.mp3')
def serve_audio():
    for folder in ['public', 'static', '.']:
        if os.path.exists(os.path.join(app.root_path, folder, 'interstellar.mp3')):
            return send_from_directory(os.path.join(app.root_path, folder), 'interstellar.mp3')
    return "Audio asset missing", 404

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Generates the OpenCV simulated multipart stream."""
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry_stream')
def telemetry_stream():
    """Server-Sent Events endpoint to stream telemetry to UI at 2Hz."""
    def event_stream():
        while True:
            with state_lock:
                data = json.dumps(sim_state)
            yield f"data: {data}\n\n"
            time.sleep(0.5)
            
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/api/set_state', methods=['POST'])
def set_state():
    """Enables manual simulation of states from frontend dashboard."""
    data = request.get_json() or {}
    new_state = data.get("driver_state")
    
    if new_state not in ["Active", "Drowsy", "Medical Emergency"]:
        return jsonify({"success": False, "error": "Invalid state"}), 400
        
    with state_lock:
        sim_state["driver_state"] = new_state
        if new_state == "Active":
            sim_state["ignition_enabled"] = True
            if sim_state["bac"] >= 0.08:
                sim_state["bac"] = 0.00
        elif new_state == "Drowsy":
            sim_state["ignition_enabled"] = True
        elif new_state == "Medical Emergency":
            sim_state["ignition_enabled"] = False
            sim_state["heart_rate"] = 44
            sim_state["blood_pressure_sys"] = 90
            sim_state["blood_pressure_dia"] = 55
            
        if data.get("simulate_alcohol"):
            sim_state["bac"] = 0.14
            sim_state["ignition_enabled"] = False
        elif new_state == "Active" and not data.get("simulate_alcohol"):
            sim_state["bac"] = 0.00
            
    return jsonify({"success": True, "state": sim_state})

@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory(os.path.join(app.root_path, 'public'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
