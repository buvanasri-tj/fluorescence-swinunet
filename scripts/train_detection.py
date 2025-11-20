# scripts/train_detection.py
# This will call Ultralytics YOLOv8 CLI if installed.
import subprocess, argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/yolo_detector.yaml")
    args=parser.parse_args()
    # Example using yolov8 CLI if ultralytics installed:
    # yolov8 expects a "data.yaml" describing train/val image folders + class names.
    subprocess.run(["yolo","detect","train","data=data.yaml","model=yolov8n.pt","epochs=50"], check=True)

if __name__ == "__main__":
    main()
