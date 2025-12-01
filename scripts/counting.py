# scripts/counting.py
import os
import glob
import csv

def count_from_pred_txt(pred_txt_path, score_thresh=0.5, area_thresh=0.0):
    cnt = 0
    if not os.path.exists(pred_txt_path):
        return 0
    with open(pred_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            x1,y1,x2,y2,score = map(float, parts[:5])
            area = (x2-x1)*(y2-y1)
            if score >= score_thresh and area >= area_thresh:
                cnt += 1
    return cnt

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True, help="predictions folder with .txt per image")
    p.add_argument("--out_csv", default="counts.csv")
    p.add_argument("--score_thresh", type=float, default=0.5)
    p.add_argument("--area_thresh", type=float, default=0.0)
    args = p.parse_args()

    rows = []
    for txt in sorted(glob.glob(os.path.join(args.pred_dir, "*.*"))):
        if txt.lower().endswith(".txt"):
            base = os.path.basename(txt).replace(".txt","")
            cnt = count_from_pred_txt(txt, score_thresh=args.score_thresh, area_thresh=args.area_thresh)
            rows.append((base, cnt))
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image","count"])
        writer.writerows(rows)
    print("[INFO] Counting done ->", args.out_csv)
