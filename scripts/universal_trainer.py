import torch
import os
import csv

class UniversalTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        loss_fn,
        metric_fn=None,
        resume_path=None,
        ckpt_dir="checkpoints/universal/",
        best_model_name="best.pth",
        log_path="logs/train_log.csv",
        save_every=5,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.save_every = save_every

        self.ckpt_dir = ckpt_dir
        self.best_path = os.path.join(ckpt_dir, best_model_name)
        self.log_path = log_path
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Resume
        self.start_epoch = 0
        self.best_metric = -1e9
        if resume_path and os.path.exists(resume_path):
            ckpt = torch.load(resume_path)
            self.start_epoch = ckpt["epoch"] + 1
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_metric = ckpt.get("best_metric", -1e9)
            print(f"[UniversalTrainer] Resuming from epoch {self.start_epoch}")

        # CSV log header
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_metric"])

    def run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss = 0
        total_metric = 0
        count = 0

        with torch.set_grad_enabled(train):
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(imgs)
                loss = self.loss_fn(preds, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                if self.metric_fn:
                    total_metric += self.metric_fn(preds, labels)
                count += 1

        return total_loss / count, total_metric / max(count, 1)

    def save_checkpoint(self, epoch):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
        }
        torch.save(ckpt, os.path.join(self.ckpt_dir, f"epoch_{epoch}.pth"))

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(self.start_epoch, epochs):
            print(f"\nEpoch {epoch}")

            train_loss, _ = self.run_epoch(train_loader, train=True)
            val_loss, val_metric = self.run_epoch(val_loader, train=False)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Metric: {val_metric:.4f}")

            if val_metric > self.best_metric:
                self.best_metric = val_metric
                torch.save(self.model.state_dict(), self.best_path)

            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)
