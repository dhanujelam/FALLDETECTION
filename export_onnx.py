from ultralytics import YOLO

def main():
    print("Loading YOLOv8n-pose model...")
    model = YOLO('yolov8n-pose.pt')
    
    print("Exporting model to ONNX format...")
    # Export the model
    success = model.export(format='onnx')
    print(f"Export successful: {success}")

if __name__ == '__main__':
    main()
