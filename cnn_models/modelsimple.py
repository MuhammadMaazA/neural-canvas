import torch
import torch.nn as nn
from config import Config

# ---- Very simple Conv block: Conv -> BN -> ReLU ----
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---- Very simple CNN backbone (no depthwise, no residuals, no SE) ----
class CustomCNN(nn.Module):
    """
    Simple CNN:
      - 4 Conv blocks with stride 2 for downsampling
      - Global average pooling
      - Linear projection to feat_dim
    """
    def __init__(self, feat_dim=384, dropout=0.3):
        super().__init__()

        # Input assumed 3 x H x W (e.g. 3 x 224 x 224)
        self.features = nn.Sequential(
            # 3 -> 32
            ConvBNAct(3, 32, k=3, s=2),   # 112 x 112
            # 32 -> 64
            ConvBNAct(32, 64, k=3, s=2),  # 56 x 56
            # 64 -> 128
            ConvBNAct(64, 128, k=3, s=2), # 28 x 28
            # 128 -> 256
            ConvBNAct(128, 256, k=3, s=2) # 14 x 14
        )

        # Global average pool to 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Project to feat_dim for the heads
        self.fc = nn.Linear(256, feat_dim)
        self.drop = nn.Dropout(dropout)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)         # [B, 256, H', W']
        x = self.pool(x)             # [B, 256, 1, 1]
        x = torch.flatten(x, 1)      # [B, 256]
        x = self.drop(x)
        x = self.fc(x)               # [B, feat_dim]
        return x                     # features


class MultiHeadClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes: dict[str, int], dropout=0.3):
        super().__init__()
        # Use the simple CNN backbone above
        self.backbone = CustomCNN(feat_dim=feat_dim, dropout=dropout)
        self.heads = nn.ModuleDict({
            k: nn.Linear(feat_dim, n) for k, n in num_classes.items()
        })

    def forward(self, x):
        feats = self.backbone(x)     # [B, feat_dim]
        return {k: head(feats) for k, head in self.heads.items()}


def build_model(cfg: Config, num_classes: dict[str, int]) -> nn.Module:
    # Keep the same name so you do not have to touch the config
    assert cfg.model_name == "custom_cnn"
    return MultiHeadClassifier(feat_dim=384, num_classes=num_classes, dropout=cfg.dropout)


def freeze_backbone(model: nn.Module, freeze: bool = True):
    # Not used for scratch model, keep for API compatibility
    for p in model.parameters():
        p.requires_grad = not freeze
