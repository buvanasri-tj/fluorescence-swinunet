import os
import glob

def parse_yolo_txt(path):
    """
    Parse YOLO txt detection file.
    YOLO format: class x_center y_center width height confidence
    Returns number of detections.
    """
    if not os.path.exists(path):
        return 0

    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    return len(lines)

def main():
    # YOLO saves detection .txt files here:
    DET_DIR = "results/detection/test_outputs/preds/labels"

    if not os.path.isdir(DET_DIR):
        raise FileNotFoundError(f"YOLO labels folder not found: {DET_DIR}")

    print("Collecting YOLO predictions from:", DET_DIR)

    txt_files = sorted(glob.glob(os.path.join(DET_DIR, "*.txt")))

    if len(txt_files) == 0:
        raise RuntimeError("No YOLO prediction .txt files found. Did detection run successfully?")

    total_cells = 0

    print("\n--- CELL COUNT SUMMARY ---\n")

    for txt_file in txt_files:
        count = parse_yolo_txt(txt_file)
        total_cells += count

        img_name = os.path.basename(txt_file).replace(".txt", ".png")
        print(f"{img_name:30s} -> {count} cells")

    print("\n===============================")
    print(f"TOTAL CELLS DETECTED: {total_cells}")
    print("===============================")

if __name__ == "__main__":
    main()
