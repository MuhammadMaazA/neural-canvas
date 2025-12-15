# model_timm.py
import torch
import torch.nn as nn
import timm


class TimmMultiHead(nn.Module):
    """
    Shared timm backbone (e.g. ConvNeXt-Tiny) + one linear head per task.
    Compatible with your current train.py which expects:
        logits = model(x)  # dict: key -> [B, Ck]
    """
    def __init__(self, num_classes: dict, model_name: str = "convnext_tiny"):
        super().__init__()

        # Pretrained backbone, return pooled features instead of logits
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,       # no classifier head, get features
            global_pool="avg",   # [B, feat_dim]
        )

        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        else:
            # Fallback: infer feature dim with a dummy forward
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = self.backbone(dummy)
            feat_dim = out.shape[-1]

        # One head per target (artist / style / genre etc.)
        self.heads = nn.ModuleDict({
            k: nn.Linear(feat_dim, n_classes)
            for k, n_classes in num_classes.items()
        })

    def forward(self, x):
        # x: [B, 3, H, W]
        feats = self.backbone(x)          # [B, feat_dim]
        out = {k: head(feats) for k, head in self.heads.items()}
        return out


def build_model(cfg, num_classes: dict):
    """
    Wrapper so train.py can still call build_model(cfg, num_classes).

    You can optionally add cfg.backbone_name in your Config;
    otherwise we default to 'convnext_tiny'.
    """
    model_name = getattr(cfg, "backbone_name", "convnext_tiny")
    return TimmMultiHead(num_classes, model_name=model_name)
