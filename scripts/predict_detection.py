# scripts/predict_detection.py
import os, sys, argparse, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datasets.detection_dataset import DetectionDataset, collate_fn
from utils.eval_detection import iou_boxes  # optional

def build_model(num_classes, ckpt_path=None, device="cuda"):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)
    return model

def visualize_and_save(img_pil, boxes, scores, out_path, score_thresh=0.5):
    draw = ImageDraw.Draw(img_pil)
    for (b, s) in zip(boxes, scores):
        if s < score_thresh:
            continue
        x1,y1,x2,y2 = b
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        draw.text((x1, y1-10), f"{s:.2f}", fill="yellow")
    img_pil.save(out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", default="predictions_detection")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--score_thresh", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # build model (num_classes must match training)
    # If you used cfg file, open it to find num_classes
    num_classes = 2
    model = build_model(num_classes, args.ckpt, device=device)
    model.to(device)
    model.eval()

    import torchvision.transforms as T
    transforms = T.Compose([T.ToTensor()])

    ds = DetectionDataset(args.images_dir, labels_dir_or_json=args.images_dir, transforms=transforms)
    # we used labels dir arg only to keep signature; here we only need images
    # but DetectionDataset expects labels param; we instead will iterate files directly
    # simpler: iterate images in directory
    from glob import glob
    img_files = sorted([p for p in glob(os.path.join(args.images_dir, "*")) if p.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))])

    for img_path in img_files:
        img = Image.open(img_path).convert("RGB")
        tensor = transforms(img).to(device)
        with torch.no_grad():
            out = model([tensor])[0]
        boxes = out.get("boxes", torch.empty((0,4))).cpu().numpy()
        scores = out.get("scores", torch.empty((0,))).cpu().numpy()
        base = os.path.basename(img_path)
        out_vis = os.path.join(args.out_dir, base)
        visualize_and_save(img, boxes, scores, out_vis, score_thresh=args.score_thresh)
        # write .txt predictions
        pred_txt = os.path.join(args.out_dir, base + ".txt")
        with open(pred_txt, "w") as f:
            for b,s in zip(boxes, scores):
                f.write(f"{b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f} {s:.4f}\n")
    print("[INFO] Predictions saved to", args.out_dir)
