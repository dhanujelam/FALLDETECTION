import numpy as np
from ultralytics import YOLO
from config import cfg

class DetectionEngine:
    def __init__(self):
        # We explicitly tell it it's a 'pose' task because ONNX needs to know
        self.model = YOLO(cfg.MODEL_PATH, task="pose") 

    def process_frame(self, frame):
        # Tracking with the ONNX model
        return self.model.track(frame, persist=True, verbose=False, conf=cfg.CONF_THRESHOLD)

    def analyze_logic(self, person_keypoints, prev_y):
        # Extract normalized coordinates (x, y) for the joints
        kpts = person_keypoints.xyn[0].cpu().numpy()
        
        # Filter out keypoints that weren't detected
        valid_kpts = kpts[kpts[:, 0] > 0]
        
        # Safety check: We need at least 5 valid points to calculate posture safely
        if len(valid_kpts) < 5: 
            return 0, None, "NORMAL", False

        # --- ROBUST SOS GESTURE ---
        # Grab valid Y-coordinates for the Head (0-4) and Shoulders (5-6)
        upper_body_y = [y for y in kpts[:7, 1] if y > 0]
        
        if len(upper_body_y) > 0:
            reference_y = min(upper_body_y)
            gesture_active = (0 < kpts[9][1] < reference_y) and (0 < kpts[10][1] < reference_y)
        else:
            gesture_active = False
            reference_y = 1.0 

        # --- EXTRACT SPINE COORDINATES ---
        shoulder_y = (kpts[5][1] + kpts[6][1]) / 2 if (kpts[5][1] > 0 and kpts[6][1] > 0) else max(kpts[5][1], kpts[6][1])
        hip_y = (kpts[11][1] + kpts[12][1]) / 2 if (kpts[11][1] > 0 and kpts[12][1] > 0) else max(kpts[11][1], kpts[12][1])
        shoulder_x = (kpts[5][0] + kpts[6][0]) / 2 if (kpts[5][0] > 0 and kpts[6][0] > 0) else max(kpts[5][0], kpts[6][0])
        hip_x = (kpts[11][0] + kpts[12][0]) / 2 if (kpts[11][0] > 0 and kpts[12][0] > 0) else max(kpts[11][0], kpts[12][0])

        # --- FALL LOGIC ---
        dy = abs(shoulder_y - hip_y) if (shoulder_y > 0 and hip_y > 0) else 1.0
        dx = abs(shoulder_x - hip_x) + 1e-6 if (shoulder_x > 0 and hip_x > 0) else 1e-6
        angle = np.degrees(np.arctan2(dy, dx))
        
        velocity = (hip_y - prev_y) if (prev_y is not None and hip_y > 0) else 0
        
        x_min, x_max = np.min(valid_kpts[:, 0]), np.max(valid_kpts[:, 0])
        y_min, y_max = np.min(valid_kpts[:, 1]), np.max(valid_kpts[:, 1])
        w = x_max - x_min
        h = y_max - y_min
        
        head_hip_dist = abs(reference_y - hip_y) if (reference_y < 1.0 and hip_y > 0) else 1.0
        
        # Calculate aspect ratio
        ratio = w / h if h > 0 else 0

        score = 0
        # 1. Torso angle (upright is ~90, flat is ~0)
        if angle < 60: score += 20
        if angle < 45: score += 30  # cumulative 50
        if angle < 30: score += 30  # cumulative 80

        # 2. Aspect Ratio (Width vs Height)
        if ratio > 0.8: score += 20
        if ratio > 1.2: score += 30  # cumulative 50

        # 3. Vertical alignment (Head vs Hip)
        if head_hip_dist < 0.3: score += 20
        if head_hip_dist < 0.15: score += 30  # cumulative 50

        # 4. Rapid drop (Velocity)
        if velocity > 0.05: score += 30
        
        # Cap max score
        score = min(score, 100)

        status = "FALL_DETECTED" if score >= 80 else "NORMAL"
        return int(score), hip_y, status, gesture_active