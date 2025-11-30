import torch
import os

class SegmentationTrainer:
    def __init__(self, model, optimizer, criterion, device,
                 resume_path=None, save_path="checkpoints/swinunet_last.pth"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path

        self.start_epoch = 0

        # Resume training
        if resume_path and os.path.exists(resume_path):
            ckpt = torch.load(resume_path)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = ckpt["epoch"] + 1
            print(f"[SegmentationTrainer] Resuming from epoch {self.start_epoch}")

    def train_step(self, batch):
        self.model.train()
        img, mask = batch
        img, mask = img.to(self.device), mask.to(self.device)

        pred = self.model(img)
        loss = self.criterion(pred, mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def val_step(self, batch):
        self.model.eval()
        img, mask = batch
        img, mask = img.to(self.device), mask.to(self.device)

        with torch.no_grad():
            pred = self.model(img)
            loss = self.criterion(pred, mask)

        return loss.item()

    def save_checkpoint(self, epoch):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(ckpt, self.save_path)
        print(f"[SegmentationTrainer] Saved checkpoint at epoch {epoch}")
