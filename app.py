from flask import Flask, Response, render_template, request, jsonify
import cv2
import time
import threading
import logging
import serial 

from config import cfg
from camera_manager import CameraManager
from detection_engine import DetectionEngine
from tracking_engine import TrackingEngine
from alert_manager import AlertManager

# Suppress annoying web logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# --- INDUSTRIAL ENGINES ---
camera = CameraManager(source=0) 
detector = DetectionEngine()
tracker = TrackingEngine()
alerts = AlertManager()

# --- ARDUINO SYSTEM (COM3) ---
try:
    # ⚠️ Keep this as COM3 to match your Arduino Uno!
    arduino = serial.Serial('COM3', 9600, timeout=1) 
    time.sleep(2) # Give the Arduino 2 seconds to wake up
    print("[SUCCESS] Arduino Physical Alarm Connected on COM3!")
except Exception as e:
    arduino = None
    print(f"[WARNING] Could not connect to Arduino: {e}")

# --- GLOBAL SYSTEM STATE ---
system_state = {
    "active_alert": None,      
    "gesture_enabled": True,   
    "crowd_enabled": False,    
    "people_count": 0,         
    "logs": []                 
}

def add_log(message):
    """Adds a timestamped log to the UI sidebar"""
    t = time.strftime("[%H:%M:%S]")
    system_state["logs"].insert(0, f"{t} {message}")
    if len(system_state["logs"]) > 50: 
        system_state["logs"].pop()
    print(f"{t} {message}")

add_log("SYSTEM ONLINE. AI CORES ACTIVE.")

def set_arduino_alarm(state):
    """Helper function to safely send signals to the Arduino"""
    if arduino:
        try:
            if state == "ON":
                arduino.write(b'1')
            elif state == "OFF":
                arduino.write(b'0')
        except Exception as e:
            print(f"[ERROR] Lost connection to Arduino: {e}")

def trigger_alarm(alert_type, p_id=None):
    """Triggers the physical Arduino alarm if one isn't already ringing"""
    if system_state["active_alert"] is None:
        system_state["active_alert"] = alert_type
        
        if p_id is not None:
            add_log(f"⚠️ EMERGENCY: {alert_type} triggered by Person {p_id}!")
            alerts.log_incident(alert_type, "CRITICAL", f"Person {p_id} triggered {alert_type}.", 1)
        
        # Trigger the physical Arduino alarm!
        set_arduino_alarm("ON")

def generate_stream():
    person_history = {} 
    
    while True:
        start_time = time.time() 

        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        results = detector.process_frame(frame)
        
        if results and len(results) > 0:
            frame = results[0].plot() 
            
            # --- TIERED CROWD MANAGEMENT ENGINE ---
            current_count = len(results[0].boxes) if results[0].boxes else 0
            system_state["people_count"] = current_count
            
            if system_state["crowd_enabled"]:
                if current_count > cfg.CRITICAL_CAPACITY:
                    if system_state["active_alert"] not in ["FALL_DETECTED", "SOS_GESTURE"]:
                        system_state["active_alert"] = "SEVERE_OVERCROWDING"
                        set_arduino_alarm("ON")
                        add_log(f"🚨 ALERT OVERCROWDED: {current_count} people detected!")
                        alerts.log_incident("SEVERE_OVERCROWDING", "CRITICAL", f"Severe capacity exceeded: {current_count}/{cfg.CRITICAL_CAPACITY}", current_count)
                
                elif cfg.WARNING_CAPACITY <= current_count <= cfg.CRITICAL_CAPACITY:
                    if system_state["active_alert"] == "SEVERE_OVERCROWDING":
                        system_state["active_alert"] = None
                        set_arduino_alarm("OFF")
                        add_log("⚠️ Crowd level downgraded to WARNING.")
                
                else:
                    if system_state["active_alert"] == "SEVERE_OVERCROWDING":
                        system_state["active_alert"] = None
                        set_arduino_alarm("OFF")
                        add_log("✅ CROWD CLEARED: Capacity restored to normal.")

            # --- INDIVIDUAL TRACKING ---
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                keypoints = results[0].keypoints
                
                active_ids = set(ids)
                stale_ids = [pid for pid in person_history.keys() if pid not in active_ids]
                for stale_id in stale_ids:
                    del person_history[stale_id]
                    if stale_id in tracker.states:
                        del tracker.states[stale_id]
                
                for i, p_id in enumerate(ids):
                    prev_y = person_history.get(p_id)
                    
                    raw_score, curr_y, raw_status, gesture = detector.analyze_logic(keypoints[i], prev_y)
                    person_history[p_id] = curr_y

                    smooth_status, smooth_score = tracker.update(p_id, raw_score, {"y": curr_y})

                    if system_state["gesture_enabled"] and gesture:
                        trigger_alarm("SOS_GESTURE", p_id)

                    if smooth_status == "CRITICAL" or raw_status == "FALL_DETECTED":
                        trigger_alarm("FALL_DETECTED", p_id)

        # 4. Draw Persistent Banners 
        if system_state["active_alert"] == "SOS_GESTURE":
            cv2.rectangle(frame, (0, 0), (cfg.INPUT_WIDTH, 80), (0, 165, 255), -1)
            cv2.putText(frame, "SOS GESTURE: HANDS RAISED (AWAITING ACKNOWLEDGMENT)", (50, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
        elif system_state["active_alert"] == "FALL_DETECTED":
            cv2.rectangle(frame, (0, cfg.INPUT_HEIGHT-80), (cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT), (0, 0, 255), -1)
            cv2.putText(frame, "EMERGENCY: FALL DETECTED (AWAITING ACKNOWLEDGMENT)", (50, cfg.INPUT_HEIGHT-25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                        
        elif system_state["active_alert"] == "SEVERE_OVERCROWDING":
            cv2.rectangle(frame, (0, 0), (cfg.INPUT_WIDTH, 80), (0, 0, 255), -1)
            cv2.putText(frame, f"ALERT OVERCROWDED: {system_state['people_count']} PEOPLE", (50, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        elif system_state["crowd_enabled"] and cfg.WARNING_CAPACITY <= system_state["people_count"] <= cfg.CRITICAL_CAPACITY:
            cv2.rectangle(frame, (0, 0), (cfg.INPUT_WIDTH, 80), (0, 165, 255), -1) 
            cv2.putText(frame, f"OVERCROWDING: {system_state['people_count']} PEOPLE", (50, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = 1.0 / elapsed
            print(f"Current Speed: {fps:.1f} FPS", end="\r") 

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret: continue
            
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- WEB UI ROUTES ---
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state', methods=['GET'])
def get_state():
    current_state = system_state.copy()
    current_state["audio_playing"] = (system_state["active_alert"] is not None)
    return jsonify(current_state)

@app.route('/acknowledge', methods=['POST'])
def acknowledge():
    if system_state["active_alert"]:
        system_state["active_alert"] = None
        set_arduino_alarm("OFF")
        add_log("✅ ALARM ACKNOWLEDGED & SILENCED BY OPERATOR.")
        alerts.log_incident("ALERT_ACKNOWLEDGED", "INFO", "Operator silenced the alarm.", 0)
    return jsonify({"status": "success"})

# --- 🚨 NEW ROUTE: EMERGENCY CALL DISPATCH ---
@app.route('/emergency_call', methods=['POST'])
def emergency_call():
    # 1. Stop the physical Arduino buzzer instantly
    if system_state["active_alert"]:
        system_state["active_alert"] = None
        set_arduino_alarm("OFF") 
    
    # 2. Log it to the UI and Database
    target = request.json.get("target", "Unknown") if request.json else "Unknown"
    log_msg = f"📞 EMERGENCY CALL INITIATED: Calling {target} representative"
    add_log(log_msg)
    alerts.log_incident("DISPATCH_CALL", "CRITICAL", f"Operator called {target}.", 0)
    
    return jsonify({"status": "success"})

@app.route('/toggle_gesture', methods=['POST'])
def toggle_gesture():
    system_state["gesture_enabled"] = not system_state["gesture_enabled"]
    state_str = "ON" if system_state["gesture_enabled"] else "OFF"
    print(f"\n⚙️ GESTURE CONTROL TURNED {state_str}") 
    return jsonify({"gesture_enabled": system_state["gesture_enabled"]})

@app.route('/toggle_crowd', methods=['POST'])
def toggle_crowd():
    system_state["crowd_enabled"] = not system_state["crowd_enabled"]
    state_str = "ON" if system_state["crowd_enabled"] else "OFF"
    print(f"\n👥 OVERCROWD DETECTION TURNED {state_str}")
    
    if not system_state["crowd_enabled"] and system_state["active_alert"] == "SEVERE_OVERCROWDING":
        system_state["active_alert"] = None
        set_arduino_alarm("OFF")
            
    return jsonify({"crowd_enabled": system_state["crowd_enabled"]})

@app.route('/play_demo', methods=['POST'])
def play_demo():
    if system_state["active_alert"]:
        print("\n⚠️ CANNOT TEST ALARM DURING ACTIVE EMERGENCY.")
        return jsonify({"status": "error"})

    if request.json and request.json.get("action") == "stop":
        set_arduino_alarm("OFF")
        print("\n🔇 TEST ALARM STOPPED.")
    else:
        set_arduino_alarm("ON")
        print("\n🔔 TEST ALARM TRIGGERED.")
            
    return jsonify({"status": "success"})

if __name__ == '__main__':
    print("--- FALLGUARD PRO SYSTEM ACTIVE ---")
    print("➡️ OPEN YOUR BROWSER TO: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)