import os
import time
import json
import random
import threading
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template, send_from_directory

app = Flask(__name__, template_folder='templates')

IS_VERCEL = os.environ.get('VERCEL') == '1' or os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

# --- Simulation & Telemetry State ---
sim_state = {
    "ignition_enabled": True,
    "driver_state": "Active",  # Active, Drowsy, Medical Emergency
    "heart_rate": 72,
    "blood_pressure_sys": 120,
    "blood_pressure_dia": 80,
    "bac": 0.00,  # Blood Alcohol Concentration
    "camera_connected": False,
    "ear_val": 0.28,
    "eye_closed_alert": False
}

state_lock = threading.Lock()

# Load Haar Cascade Classifiers for OpenCV Face & Eye detection
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
eye_cascade_alt_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
if eye_cascade.empty():
    eye_cascade = cv2.CascadeClassifier(eye_cascade_alt_path)

def update_telemetry_step():
    """Updates telemetry data safely based on driver state."""
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

# Only start continuous background threads when running locally (not on Vercel)
if not IS_VERCEL:
    def update_telemetry_loop():
        while True:
            update_telemetry_step()
            time.sleep(1.0)
            
    thread = threading.Thread(target=update_telemetry_loop, daemon=True)
    thread.start()


# Helper function to generate single frame JPEG (Serverless Safe)
def draw_hud_frame(frame=None, using_webcam=False, is_black_frame=False):
    frame_width = 640
    frame_height = 480

    cyan = (255, 240, 0)
    green = (0, 255, 100)
    amber = (0, 165, 255)
    red = (50, 50, 255)
    white = (255, 255, 255)

    if frame is None or is_black_frame or not using_webcam:
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for i in range(0, frame_width, 40):
            cv2.line(frame, (i, 0), (i, frame_height), (25, 25, 35), 1)
        for j in range(0, frame_height, 40):
            cv2.line(frame, (0, j), (frame_width, j), (25, 25, 35), 1)

    is_eyes_closed = False
    calculated_ear = 0.28

    if using_webcam and frame is not None:
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.equalizeHist(gray_img)

        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            (fx, fy, fw, fh) = faces[0]

            l_len = 25
            th = 2
            cv2.line(frame, (fx, fy), (fx + l_len, fy), cyan, th)
            cv2.line(frame, (fx, fy), (fx, fy + l_len), cyan, th)
            cv2.line(frame, (fx + fw, fy), (fx + fw - l_len, fy), cyan, th)
            cv2.line(frame, (fx + fw, fy), (fx + fw, fy + l_len), cyan, th)
            cv2.line(frame, (fx, fy + fh), (fx + l_len, fy + fh), cyan, th)
            cv2.line(frame, (fx, fy + fh), (fx, fy + fh - l_len), cyan, th)
            cv2.line(frame, (fx + fw, fy + fh), (fx + fw - l_len, fy + fh), cyan, th)
            cv2.line(frame, (fx + fw, fy + fh), (fx + fw, fy + fh - l_len), cyan, th)

            eye_y = fy + int(fh * 0.38)
            left_eye_x = fx + int(fw * 0.33)
            right_eye_x = fx + int(fw * 0.67)
            nose_x, nose_y = fx + int(fw * 0.50), fy + int(fh * 0.55)
            mouth_x, mouth_y = fx + int(fw * 0.50), fy + int(fh * 0.76)

            mesh_points = [
                (left_eye_x, eye_y), (right_eye_x, eye_y),
                (nose_x, nose_y), (mouth_x, mouth_y),
                (fx + int(fw * 0.20), fy + int(fh * 0.28)),
                (fx + int(fw * 0.80), fy + int(fh * 0.28)),
                (fx + int(fw * 0.50), fy + int(fh * 0.95))
            ]
            
            for i in range(len(mesh_points)):
                for j in range(i + 1, len(mesh_points)):
                    cv2.line(frame, mesh_points[i], mesh_points[j], (100, 255, 200), 1, cv2.LINE_AA)
            for pt in mesh_points:
                cv2.circle(frame, pt, 3, cyan, -1)

            eye_roi_y1 = fy + int(fh * 0.18)
            eye_roi_y2 = fy + int(fh * 0.52)
            eye_roi_x1 = fx + int(fw * 0.08)
            eye_roi_x2 = fx + int(fw * 0.92)

            eye_roi_gray = gray_img[eye_roi_y1:eye_roi_y2, eye_roi_x1:eye_roi_x2]
            eye_roi_color = frame[eye_roi_y1:eye_roi_y2, eye_roi_x1:eye_roi_x2]

            eyes = eye_cascade.detectMultiScale(eye_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

            if len(eyes) == 0:
                is_eyes_closed = True
                calculated_ear = 0.08
            else:
                ear_list = []
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(eye_roi_color, (ex, ey), (ex + ew, ey + eh), green, 1)
                    cv2.circle(eye_roi_color, (ex + ew // 2, ey + eh // 2), 3, cyan, -1)
                    cv2.line(eye_roi_color, (ex, ey + eh // 2), (ex + ew, ey + eh // 2), cyan, 1)
                    cv2.line(eye_roi_color, (ex + ew // 2, ey), (ex + ew // 2, ey + eh), cyan, 1)
                    aspect_ratio = eh / float(ew)
                    ear_list.append(aspect_ratio)

                avg_ear = np.mean(ear_list) if len(ear_list) > 0 else 0.0
                calculated_ear = round(float(avg_ear * 0.65), 2)
                if calculated_ear < 0.18:
                    is_eyes_closed = True
                else:
                    is_eyes_closed = False

    else:
        with state_lock:
            driver_state = sim_state["driver_state"]

        face_x, face_y = 320, 240
        t_sim = time.time()
        pulse_rad = int(80 + 4 * np.sin(t_sim * 3))

        if driver_state == "Active":
            cv2.ellipse(frame, (face_x, face_y), (pulse_rad, 110), 0, 0, 360, green, 2)
            cv2.circle(frame, (face_x - 30, face_y - 20), 10, cyan, 2)
            cv2.circle(frame, (face_x - 30, face_y - 20), 3, cyan, -1)
            cv2.circle(frame, (face_x + 30, face_y - 20), 10, cyan, 2)
            cv2.circle(frame, (face_x + 30, face_y - 20), 3, cyan, -1)
            cv2.ellipse(frame, (face_x, face_y + 40), (25, 8), 0, 0, 180, green, 2)
            cv2.line(frame, (face_x - 30, face_y - 20), (face_x, face_y + 10), cyan, 1)
            cv2.line(frame, (face_x + 30, face_y - 20), (face_x, face_y + 10), cyan, 1)
            cv2.line(frame, (face_x, face_y + 10), (face_x, face_y + 40), cyan, 1)
            calculated_ear = 0.28
        elif driver_state == "Drowsy":
            cv2.ellipse(frame, (face_x, face_y + 10), (pulse_rad, 110), 5, 0, 360, amber, 2)
            cv2.line(frame, (face_x - 40, face_y - 15), (face_x - 20, face_y - 15), amber, 3)
            cv2.line(frame, (face_x + 20, face_y - 15), (face_x + 40, face_y - 15), amber, 3)
            cv2.ellipse(frame, (face_x, face_y + 45), (15, 25), 0, 0, 360, amber, 2)
            calculated_ear = 0.08
            is_eyes_closed = True
        else:
            cv2.ellipse(frame, (face_x - 30, face_y + 20), (pulse_rad, 110), -20, 0, 360, red, 2)
            cv2.line(frame, (face_x - 60, face_y + 5), (face_x - 40, face_y - 5), red, 3)
            cv2.line(frame, (face_x - 10, face_y + 25), (face_x + 10, face_y + 15), red, 3)
            calculated_ear = 0.00
            is_eyes_closed = True

    with state_lock:
        current_driver_state = sim_state["driver_state"]
        ignition = sim_state["ignition_enabled"]
        bac = sim_state["bac"]
        hr = sim_state["heart_rate"]

    if current_driver_state == "Active":
        hud_color = green if ignition else amber
        status_str = "STATUS: ACTIVE & NOMINAL"
    elif current_driver_state == "Drowsy":
        hud_color = amber
        status_str = "WARNING: DROWSINESS / EYE CLOSURE DETECTED"
    else:
        hud_color = red
        status_str = "CRITICAL: DRIVER UNRESPONSIVE / EMERGENCY"

    cam_label = "LIVE WEBCAM (PYTHON OpenCV AI)" if using_webcam else ("VERCEL CLOUD AI" if IS_VERCEL else "AI VECTOR TRACKING")
    cv2.putText(frame, f"SAFEWARE AI-VISION v2.0 // {cam_label}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, cyan, 2)
    cv2.putText(frame, status_str, (15, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)

    cv2.putText(frame, "FPS: 30.0", (480, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)
    ear_color = red if is_eyes_closed else cyan
    cv2.putText(frame, f"EAR: {calculated_ear:.2f} {'[CLOSED]' if is_eyes_closed else '[OPEN]'}", (480, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ear_color, 1)
    cv2.putText(frame, f"BAC: {bac:.3f}%", (480, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)
    cv2.putText(frame, f"HR: {hr} BPM", (480, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)

    if is_eyes_closed or current_driver_state in ["Drowsy", "Medical Emergency"]:
        if (int(time.time() * 4) % 2) == 0:
            alert_color = red if current_driver_state == "Medical Emergency" else amber
            cv2.rectangle(frame, (20, 60), (450, 110), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 60), (450, 110), alert_color, 2)
            cv2.putText(frame, "⚠️ EYE CLOSURE / FATIGUE ALERT!", (35, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, alert_color, 2)

    l_corner = 20
    th_c = 2
    cv2.line(frame, (10, 10), (10 + l_corner, 10), hud_color, th_c)
    cv2.line(frame, (10, 10), (10, 10 + l_corner), hud_color, th_c)
    cv2.line(frame, (frame_width-10, 10), (frame_width-10 - l_corner, 10), hud_color, th_c)
    cv2.line(frame, (frame_width-10, 10), (frame_width-10, 10 + l_corner), hud_color, th_c)
    cv2.line(frame, (10, frame_height-10), (10 + l_corner, frame_height-10), hud_color, th_c)
    cv2.line(frame, (10, frame_height-10), (10, frame_height-10 - l_corner), hud_color, th_c)
    cv2.line(frame, (frame_width-10, frame_height-10), (frame_width-10 - l_corner, frame_height-10), hud_color, th_c)
    cv2.line(frame, (frame_width-10, frame_height-10), (frame_width-10, frame_height-10 - l_corner), hud_color, th_c)

    ret_enc, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes() if ret_enc else None


# Local Camera Streamer (Running only when NOT on Vercel)
class LocalCameraStreamer:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.current_frame_bytes = None
        self.is_running = True
        self.closed_eye_counter = 0
        self.using_real_webcam = False
        if not IS_VERCEL:
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def _init_camera(self):
        for idx in [0, 1]:
            for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW]:
                try:
                    c = cv2.VideoCapture(idx, backend)
                    if c.isOpened():
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        for _ in range(3):
                            ret, f = c.read()
                        if ret and f is not None:
                            self.cap = c
                            return
                        c.release()
                except Exception:
                    pass
        self.cap = None

    def _capture_loop(self):
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                self._init_camera()
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.5)

            frame = None
            using_webcam = False
            is_black_frame = False

            if self.cap is not None and self.cap.isOpened():
                ret, captured = self.cap.read()
                if ret and captured is not None:
                    using_webcam = True
                    frame = cv2.resize(captured, (640, 480))
                    frame = cv2.flip(frame, 1)
                    if np.mean(frame) < 2.0:
                        is_black_frame = True

            self.using_real_webcam = using_webcam and not is_black_frame
            with state_lock:
                sim_state["camera_connected"] = self.using_real_webcam

            frame_bytes = draw_hud_frame(frame, using_webcam=self.using_real_webcam, is_black_frame=is_black_frame)
            if frame_bytes is not None:
                with self.lock:
                    self.current_frame_bytes = frame_bytes

            time.sleep(0.03)

    def get_frame_stream(self):
        if IS_VERCEL:
            # Yield dynamic HUD frame per request on Vercel
            for _ in range(10):
                update_telemetry_step()
                frame_bytes = draw_hud_frame()
                if frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
        else:
            while self.is_running:
                with self.lock:
                    frame_bytes = self.current_frame_bytes
                if frame_bytes is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.033)

camera_streamer = LocalCameraStreamer()


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
    """Streams live OpenCV Computer Vision frames."""
    return Response(camera_streamer.get_frame_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry_stream')
def telemetry_stream():
    """Server-Sent Events endpoint to stream real-time telemetry state to UI."""
    def event_stream():
        # Stream 5 iterations per request on Vercel to prevent infinite unfreezes
        iterations = 5 if IS_VERCEL else 1000000
        for _ in range(iterations):
            update_telemetry_step()
            with state_lock:
                data = json.dumps(sim_state)
            yield f"data: {data}\n\n"
            time.sleep(0.3)
            
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
