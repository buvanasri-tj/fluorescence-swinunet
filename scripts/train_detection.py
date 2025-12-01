# scripts/train_detection.py
import os
import sys
import time
import argparse
import yaml
import csv
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# ensure repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.detection_dataset import DetectionDataset, collate_fn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.eval_detection import evaluate_simple    # we'll add this script below

def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def freeze_backbone(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "body"):
        for p in model.backbone.body.parameters():
            p.requires_grad = False

def unfreeze_backbone(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "body"):
        for p in model.backbone.body.parameters():
            p.requires_grad = True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--output_dir", default="checkpoints/detection")
    p.add_argument("--stage1_epochs", type=int, default=0,
                   help="If >0, run head-only training for this many epochs first")
    return p.parse_args()

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_checkpoint(state, path):
    torch.save(state, path)

def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    it = 0
    for imgs, targets in loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast():
            loss_dict = model(imgs, targets)
            losses = sum(v for v in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += losses.item()
        it += 1

    return running_loss / max(1, it)

if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Model
    num_classes = int(cfg["model"].get("num_classes", 2))
    model = build_model(num_classes).to(device)

    # Optionally load weights (strict=False)
    pre_ckpt = cfg["model"].get("pretrained_ckpt", None)
    if pre_ckpt:
        ckpt_path = os.path.join(ROOT, pre_ckpt)
        if os.path.exists(ckpt_path):
            print("[INFO] Loading pretrained checkpoint (partial):", ckpt_path)
            state = torch.load(ckpt_path, map_location=device)
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                pass

    # Dataset & dataloaders
    dcfg = cfg["dataset"]
    train_images = dcfg["train_images_dir"]
    train_labels = dcfg["train_labels"]
    val_images = dcfg.get("val_images_dir", None)
    val_labels = dcfg.get("val_labels", None)

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_ds = DetectionDataset(train_images, train_labels, transforms=transforms)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True,
                              num_workers=2, collate_fn=collate_fn)

    if val_images and val_labels:
        val_ds = DetectionDataset(val_images, val_labels, transforms=transforms)
        val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
                                num_workers=2, collate_fn=collate_fn)
    else:
        val_loader = None

    # Optimizer & scaler
    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"]["weight_decay"])
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=lr, momentum=0.9, weight_decay=wd)
    scaler = GradScaler()

    # CSV logging
    csv_path = os.path.join(args.output_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stage","epoch","train_loss","val_precision","val_recall","val_f1"])

    best_f1 = 0.0
    start_epoch = 0

    # RESUME support: last_checkpoint.pth contains model+optim+epoch+best_f1
    last_path = os.path.join(args.output_dir, "last_checkpoint.pth")
    if os.path.exists(last_path):
        print("[INFO] Resuming from last checkpoint")
        chk = torch.load(last_path, map_location=device)
        model.load_state_dict(chk["model"])
        optimizer.load_state_dict(chk["optimizer"])
        start_epoch = chk["epoch"] + 1
        best_f1 = chk.get("best_f1", 0.0)

    # Stage 1: optional head-only
    stage1_epochs = int(args.stage1_epochs)
    if stage1_epochs > 0:
        freeze_backbone(model)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                    lr=lr, momentum=0.9, weight_decay=wd)
        print(f"[INFO] Running Stage1 (heads only) for {stage1_epochs} epochs")
        for e in range(start_epoch, stage1_epochs):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
            val_metrics = (None, None, None)
            if val_loader:
                val_metrics = evaluate_simple(model, val_loader, device)
            print(f"Stage1 Epoch {e} loss {train_loss:.4f} val_metrics={val_metrics}")
            # write CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["stage1", e, train_loss, val_metrics[0], val_metrics[1], val_metrics[2]])
            # save last
            torch.save({"epoch": e, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_f1": best_f1}, last_path)
            # save every epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"stage1_epoch_{e}.pth"))
            # best
            if val_loader and val_metrics[2] is not None and val_metrics[2] > best_f1:
                best_f1 = val_metrics[2]
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        # after stage1, unfreeze and continue from epoch 0 for stage2
        start_epoch = 0
        unfreeze_backbone(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr*0.1, momentum=0.9, weight_decay=wd)

    # Stage 2: full training
    epochs = int(cfg["training"]["epochs"])
    print(f"[INFO] Starting Stage2 full training for {epochs} epochs (start_epoch={start_epoch})")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_metrics = (None, None, None)
        if val_loader:
            val_metrics = evaluate_simple(model, val_loader, device)

        print(f"[EPOCH {epoch}] train_loss={train_loss:.4f} val_metrics={val_metrics} time={(time.time()-t0):.1f}s")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stage2", epoch, train_loss, val_metrics[0], val_metrics[1], val_metrics[2]])

        # save last
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_f1": best_f1}, last_path)
        # every epoch state (model only)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"epoch_{epoch}.pth"))

        # save best
        if val_loader and val_metrics[2] is not None and val_metrics[2] > best_f1:
            best_f1 = val_metrics[2]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"[INFO] New best model saved with F1 {best_f1:.4f}")

    print("[INFO] Training finished.")
