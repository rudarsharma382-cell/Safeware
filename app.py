import os
import time
import json
import random
import threading
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template, send_from_directory

app = Flask(__name__, template_folder='templates')

# --- Simulation State Engine ---
# Holds current real-time telemetry parameters
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
                # Normal healthy fluctuations
                sim_state["heart_rate"] = int(np.clip(sim_state["heart_rate"] + random.choice([-1, 0, 1]), 65, 80))
                sim_state["blood_pressure_sys"] = int(np.clip(sim_state["blood_pressure_sys"] + random.choice([-2, 0, 2]), 115, 125))
                sim_state["blood_pressure_dia"] = int(np.clip(sim_state["blood_pressure_dia"] + random.choice([-1, 0, 1]), 75, 85))
                # BAC is low
                sim_state["bac"] = max(0.0, round(sim_state["bac"] + random.uniform(-0.005, 0.005), 3))
                # Ignition remains enabled unless state says otherwise
                
            elif state == "Drowsy":
                # Fatigue state - lower heart rate and lower blood pressure
                sim_state["heart_rate"] = int(np.clip(sim_state["heart_rate"] + random.choice([-1, 0, 1]), 52, 62))
                sim_state["blood_pressure_sys"] = int(np.clip(sim_state["blood_pressure_sys"] + random.choice([-2, 0, 2]), 105, 115))
                sim_state["blood_pressure_dia"] = int(np.clip(sim_state["blood_pressure_dia"] + random.choice([-1, 0, 1]), 65, 75))
                sim_state["bac"] = max(0.0, round(sim_state["bac"] + random.uniform(-0.002, 0.002), 3))
                
            elif state == "Medical Emergency":
                # Extreme state - critical telemetry drop/spike, triggers safety systems
                sim_state["heart_rate"] = int(np.clip(sim_state["heart_rate"] + random.choice([-2, 0, 2]), 40, 48))
                sim_state["blood_pressure_sys"] = int(np.clip(sim_state["blood_pressure_sys"] + random.choice([-3, 0, 3]), 85, 95))
                sim_state["blood_pressure_dia"] = int(np.clip(sim_state["blood_pressure_dia"] + random.choice([-2, 0, 2]), 50, 60))
                sim_state["ignition_enabled"] = False  # Emergency brake disables driving!
                
            # If alcohol limit exceeded, lock ignition
            if sim_state["bac"] >= 0.08:
                sim_state["ignition_enabled"] = False
                
        time.sleep(1.0)

# Start background thread
thread = threading.Thread(target=update_telemetry_loop, daemon=True)
thread.start()


# --- Simulated Video Stream Frame Generator ---
def generate_camera_frames():
    """Generates JPEG frames representing OpenCV facial detection feedback."""
    frame_width = 640
    frame_height = 480
    frame_count = 0
    
    while True:
        # Create dark neon grid background
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Draw background tech HUD grids
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
        
        # Pulse alpha for neon effects
        pulse = int(127 + 127 * np.sin(t * 3))
        blink_state = (frame_count // 5) % 2 == 0
        
        # Color definitions (BGR)
        cyan = (255, 240, 0)
        green = (0, 255, 100)
        amber = (0, 165, 255)
        red = (50, 50, 255)
        gray = (100, 100, 100)
        
        # Select UI color theme based on driver state
        if driver_state == "Active":
            hud_color = green if ignition else amber
            status_text = "STATUS: ACTIVE & SAFE" if ignition else "STATUS: SAFETY LOCK ACTIVE"
        elif driver_state == "Drowsy":
            hud_color = amber
            status_text = "WARNING: DROWSINESS DETECTED"
        else:  # Medical Emergency
            hud_color = red if blink_state else gray
            status_text = "CRITICAL: MEDICAL EMERGENCY"
            
        # Draw Tech HUD border corner markers
        thickness = 2
        l_len = 20
        # Top-Left
        cv2.line(frame, (10, 10), (10 + l_len, 10), hud_color, thickness)
        cv2.line(frame, (10, 10), (10, 10 + l_len), hud_color, thickness)
        # Top-Right
        cv2.line(frame, (frame_width-10, 10), (frame_width-10 - l_len, 10), hud_color, thickness)
        cv2.line(frame, (frame_width-10, 10), (frame_width-10, 10 + l_len), hud_color, thickness)
        # Bottom-Left
        cv2.line(frame, (10, frame_height-10), (10 + l_len, frame_height-10), hud_color, thickness)
        cv2.line(frame, (10, frame_height-10), (10, frame_height-10 - l_len), hud_color, thickness)
        # Bottom-Right
        cv2.line(frame, (frame_width-10, frame_height-10), (frame_width-10 - l_len, frame_height-10), hud_color, thickness)
        cv2.line(frame, (frame_width-10, frame_height-10), (frame_width-10, frame_height-10 - l_len), hud_color, thickness)
        
        # --- DRAW SIMULATED DRIVER FACE AND CV MAPPING ---
        # Center of face
        face_x, face_y = 320, 240
        
        if driver_state == "Active":
            # Normal face structure, eye aspect ratio high
            cv2.ellipse(frame, (face_x, face_y), (80, 110), 0, 0, 360, hud_color, 2)
            # Eyes (Open)
            cv2.circle(frame, (face_x - 30, face_y - 20), 10, hud_color, 2)
            cv2.circle(frame, (face_x - 30, face_y - 20), 3, hud_color, -1)  # Pupils
            cv2.circle(frame, (face_x + 30, face_y - 20), 10, hud_color, 2)
            cv2.circle(frame, (face_x + 30, face_y - 20), 3, hud_color, -1)
            # Eyebrows
            cv2.line(frame, (face_x - 45, face_y - 40), (face_x - 15, face_y - 35), hud_color, 2)
            cv2.line(frame, (face_x + 15, face_y - 35), (face_x + 45, face_y - 40), hud_color, 2)
            # Mouth (Normal closed or slight smile)
            cv2.ellipse(frame, (face_x, face_y + 40), (25, 8), 0, 0, 180, hud_color, 2)
            # CV mesh keypoints (Active)
            keypoints = [(face_x, face_y - 80), (face_x, face_y + 10), (face_x - 20, face_y + 10), (face_x + 20, face_y + 10),
                         (face_x - 60, face_y), (face_x + 60, face_y), (face_x, face_y - 110), (face_x, face_y + 110)]
            for kp in keypoints:
                cv2.circle(frame, kp, 2, cyan, -1)
            
        elif driver_state == "Drowsy":
            # Head tilting slightly, eyes narrow/shut, yawning
            tilt = int(10 * np.sin(t * 2))
            cv2.ellipse(frame, (face_x, face_y + 10), (80, 110), tilt, 0, 360, hud_color, 2)
            
            # Eyes (Heavily Closed)
            # Draw line squint/shut
            cv2.line(frame, (face_x - 40 + tilt, face_y - 15), (face_x - 20 + tilt, face_y - 15), hud_color, 3)
            cv2.line(frame, (face_x + 20 + tilt, face_y - 15), (face_x + 40 + tilt, face_y - 15), hud_color, 3)
            # Droopy eyebrows
            cv2.line(frame, (face_x - 45 + tilt, face_y - 30), (face_x - 15 + tilt, face_y - 28), hud_color, 2)
            cv2.line(frame, (face_x + 15 + tilt, face_y - 28), (face_x + 45 + tilt, face_y - 30), hud_color, 2)
            
            # Mouth (Yawn - wide open oval)
            yawn_height = int(25 + 10 * np.sin(t * 5))
            cv2.ellipse(frame, (face_x + tilt, face_y + 45), (15, yawn_height), 0, 0, 360, hud_color, 2)
            
            # Blinking warning overlay
            if blink_state:
                cv2.rectangle(frame, (100, 80), (540, 400), red, 2)
                cv2.putText(frame, "EYE CLOSURE WARNING (EAR < 0.15)", (130, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
                
        else:  # Medical Emergency
            # Head slumped, eyes closed, asymmetric mouth
            cv2.ellipse(frame, (face_x - 40, face_y + 30), (80, 110), -25, 0, 360, hud_color, 2)
            # Closed eyes slanted
            cv2.line(frame, (face_x - 70, face_y + 5), (face_x - 50, face_y - 5), hud_color, 3)
            cv2.line(frame, (face_x - 20, face_y + 25), (face_x, face_y + 15), hud_color, 3)
            # Slack mouth
            cv2.ellipse(frame, (face_x - 45, face_y + 85), (20, 10), -15, 0, 360, hud_color, 2)
            
            if blink_state:
                cv2.rectangle(frame, (60, 40), (580, 440), red, 3)
                cv2.putText(frame, "CRITICAL: NO DRIVER RESPONSIVENESS", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
        
        # --- DRAW TEXT & ANALYSIS VALUES ---
        cv2.putText(frame, "SAFEWARE AI-VISION v1.2", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cyan, 2)
        cv2.putText(frame, status_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
        
        # Diagnostics sidebar
        cv2.putText(frame, f"FPS: 15.0", (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        ear_val = 0.28 if driver_state == "Active" else (0.06 if driver_state == "Drowsy" else 0.00)
        cv2.putText(frame, f"EAR: {ear_val:.2f}", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, f"BAC: {bac:.3f}%", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, f"HR: {hr} BPM", (480, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Encode as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.066)  # ~15 FPS


# --- Flask Routing ---
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.png')


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
            time.sleep(0.5)  # 2 updates per second
            
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
            # Flush alcohol simulation
            if sim_state["bac"] >= 0.08:
                sim_state["bac"] = 0.00
        elif new_state == "Drowsy":
            sim_state["ignition_enabled"] = True
        elif new_state == "Medical Emergency":
            sim_state["ignition_enabled"] = False
            sim_state["heart_rate"] = 44
            sim_state["blood_pressure_sys"] = 90
            sim_state["blood_pressure_dia"] = 55
            
        # Optional alcohol injection helper
        if data.get("simulate_alcohol"):
            sim_state["bac"] = 0.14
            sim_state["ignition_enabled"] = False
        elif new_state == "Active" and not data.get("simulate_alcohol"):
            sim_state["bac"] = 0.00
            
    return jsonify({"success": True, "state": sim_state})

from flask import send_from_directory
import os

@app.route('/interstellar.mp3')
def serve_audio():
    for folder in ['public', 'static', '.']:
        if os.path.exists(os.path.join(app.root_path, folder, 'interstellar.mp3')):
            return send_from_directory(os.path.join(app.root_path, folder), 'interstellar.mp3')
    return "Audio asset missing", 404

@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory(os.path.join(app.root_path, 'public'), filename)

if __name__ == '__main__':
    # Listen on all interfaces on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
