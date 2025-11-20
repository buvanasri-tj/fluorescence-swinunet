# You can replace this with YOLOv8/YOLOv11 internals later
import torch.nn as nn

class DummyYOLO(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(32, num_classes*5, 1)

    def forward(self,x):
        return self.head(self.backbone(x))
