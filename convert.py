from ultralytics import YOLO

# Load the PyTorch model
model = YOLO("yolov8n-pose.pt")

# Export it to ONNX format
model.export(format="onnx")