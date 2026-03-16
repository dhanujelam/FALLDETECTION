import cv2

def apply_industrial_visuals(frame, person_data, status, gesture):
    kpts = person_data.keypoints.xyn[0].cpu().numpy()
    h, w, _ = frame.shape
    
    # 1. Draw Skeleton
    # Connections: Shoulder-Hip, Shoulder-Shoulder, Hip-Hip, Arm segments
    links = [(5,6), (5,11), (6,12), (11,12), (5,7), (7,9), (6,8), (8,10)]
    color = (0, 0, 255) if (status == "CRITICAL_FALL" or gesture) else (0, 255, 0)
    
    for start, end in links:
        p1 = (int(kpts[start][0] * w), int(kpts[start][1] * h))
        p2 = (int(kpts[end][0] * w), int(kpts[end][1] * h))
        cv2.line(frame, p1, p2, color, 2)

    # 2. Add Dynamic UI Overlays
    if gesture:
        cv2.rectangle(frame, (0,0), (w, 60), (0, 165, 255), -1)
        cv2.putText(frame, "GESTURE SOS DETECTED", (w//4, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    
    if status == "CRITICAL_FALL":
        cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, "FALL ALERT: EMERGENCY", (w//4, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    
    return frame