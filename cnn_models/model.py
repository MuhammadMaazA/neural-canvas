import torch
import torch.nn as nn
try:
    from config import Config
except ImportError:
    from cnn_models.config import Config

# ---- Small building blocks ----
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, groups=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False, groups=groups)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseSeparable(nn.Module):
    """Depthwise + pointwise conv to keep params low."""
    def __init__(self, in_c, out_c, s=1):
        super().__init__()
        self.dw = ConvBNAct(in_c, in_c, k=3, s=s, p=1, groups=in_c)
        self.pw = ConvBNAct(in_c, out_c, k=1, s=1, p=0)

    def forward(self, x):
        return self.pw(self.dw(x))

class SqueezeExcite(nn.Module):
    """Simple SE block to help channel attention."""
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(c, c, 3, 1, 1),
            ConvBNAct(c, c, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)

# ---- The actual CNN ----
class CustomCNN(nn.Module):
    def __init__(self, feat_dim=384, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(ConvBNAct(3,32,3,2,1), ConvBNAct(32,32,3,1,1))
        self.stage1 = nn.Sequential(DepthwiseSeparable(32, 64, 2), Residual(64), SqueezeExcite(64))
        self.stage2 = nn.Sequential(DepthwiseSeparable(64,128, 2), Residual(128), Residual(128), SqueezeExcite(128))
        self.stage3 = nn.Sequential(DepthwiseSeparable(128,256,2), Residual(256), Residual(256), SqueezeExcite(256))
        self.stage4 = nn.Sequential(DepthwiseSeparable(256,feat_dim,2), Residual(feat_dim), SqueezeExcite(feat_dim))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.pool(x); x = torch.flatten(x,1); x = self.drop(x)
        return x  # features

class MultiHeadClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes: dict[str,int], dropout=0.3):
        super().__init__()
        self.backbone = CustomCNN(feat_dim=feat_dim, dropout=dropout)
        self.heads = nn.ModuleDict({
            k: nn.Linear(feat_dim, n) for k, n in num_classes.items()
        })

    def forward(self, x):
        feats = self.backbone(x)
        return {k: head(feats) for k, head in self.heads.items()}

def build_model(cfg: Config, num_classes: dict[str,int]) -> nn.Module:
    assert cfg.model_name == "custom_cnn"
    return MultiHeadClassifier(feat_dim=384, num_classes=num_classes, dropout=cfg.dropout)

def freeze_backbone(model: nn.Module, freeze: bool = True):
    # Not used for scratch model, keep for API compatibility
    for p in model.parameters():
        p.requires_grad = True
