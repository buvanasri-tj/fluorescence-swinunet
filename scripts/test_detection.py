import os
from ultralytics import YOLO

def main():
    # --- Load trained model ---
    MODEL_PATH = "results/detection/fluorescence_detector/weights/best.pt"
    TEST_IMAGES = "data"  # YOLO will recursively find PNGs

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # --- Inference ---
    OUT_DIR = "results/detection/test_outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Running detection on test images...")

    results = model.predict(
        source=TEST_IMAGES,
        save=True,
        save_txt=True,
        save_conf=True,
        project=OUT_DIR,
        name="preds"
    )

    print("Detection completed.")
    print(f"Predictions saved at: {OUT_DIR}/preds/")

if __name__ == "__main__":
    main()
