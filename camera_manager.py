import cv2
import threading
import time
from config import cfg

class CameraManager:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        
        # FIX: Define the attribute immediately so it's always found
        self._thread = None 

        if self.cap.isOpened():
            self.running = True
            self._thread = threading.Thread(target=self._update, daemon=True)
            self._thread.start()
            print(f"[SUCCESS] AI 'Eyes' connected to source {self.source}")
        else:
            print(f"[CRITICAL] AI 'Eyes' failed to connect to source {self.source}")

    def _update(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                # Optimized for product-grade inference speed
                frame = cv2.resize(frame, (cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT))
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.1)
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self._thread is not None and self._thread.is_alive():
            # Don't wait forever, just 1 second to release
            self._thread.join(timeout=1.0) 
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def __del__(self):
        self.stop()