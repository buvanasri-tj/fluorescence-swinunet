import os
from ultralytics import YOLO

def main():
    # Path to YOLO config file
    YOLO_CFG = "configs/yolo_detector.yaml"

    if not os.path.exists(YOLO_CFG):
        raise FileNotFoundError(f"YOLO config not found: {YOLO_CFG}")

    print(f"Using YOLO data config: {YOLO_CFG}")

    # Load YOLO model (using YOLOv8s for speed, change to yolov8m/l if needed)
    model = YOLO("yolov8s.pt")

    # Train
    model.train(
        data=YOLO_CFG,
        imgsz=640,
        epochs=80,
        batch=8,
        lr0=1e-3,
        optimizer="Adam",
        device=0,      # GPU
        workers=4,
        name="fluorescence_detector",
        project="results/detection"
    )

    print("Detection training complete.")

if __name__ == "__main__":
    main()
