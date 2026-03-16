import os
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config(BaseSettings):
    # Pointing to the new ONNX model for AMD Radeon GPU acceleration
    MODEL_PATH: str = os.path.join(BASE_DIR, "yolov8n-pose.onnx")
    
    # Thresholds
    CONF_THRESHOLD: float = 0.50
    ANGLE_THRESHOLD: int = 45        # Spine angle for falls
    VELOCITY_THRESHOLD: float = 0.15 # Drop speed
    MIN_KEYPOINT_CONF: float = 0.55  # Skeleton safety lock
    
    # Crowd Control Thresholds
    WARNING_CAPACITY: int = 5        # 5 to 10 people (Visual Warning)
    CRITICAL_CAPACITY: int = 10      # > 10 people (Siren Alarm)
    
    # UI/System/Database
    INPUT_WIDTH: int = 1920          
    INPUT_HEIGHT: int = 1080
    DB_PATH: str = os.path.join(BASE_DIR, "security_events.db")
    
    model_config = SettingsConfigDict(extra="ignore")

cfg = Config()