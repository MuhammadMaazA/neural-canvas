import os
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import timm

from config import Config
from dataset import build_dataloaders
from utils import create_scheduler, save_checkpoint, load_checkpoint, load_model_weights


# ----------------- Model definition (timm + multi-head) ----------------- #

class TimmMultiHead(nn.Module):
    """
    Shared timm backbone (e.g. ConvNeXt-Tiny) + one linear head per task.

    num_classes: dict like {"artist": 129} or {"artist": 129, "style": 27, "genre": 11}
    forward(x) returns dict: {k: logits[B, Ck]}
    """
    def __init__(self, num_classes: dict, model_name: str = "convnext_tiny"):
        super().__init__()

        # Pretrained backbone returning pooled features
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,       # no classifier head, just features
            global_pool="avg",   # [B, feat_dim]
        )

        # Feature dimension
        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        else:
            # Fallback: infer with a dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.backbone(dummy)
            feat_dim = out.shape[-1]

        # One head per task
        self.heads = nn.ModuleDict({
            k: nn.Linear(feat_dim, n_classes)
            for k, n_classes in num_classes.items()
        })

    def forward(self, x):
        feats = self.backbone(x)  # [B, feat_dim]
        out = {k: head(feats) for k, head in self.heads.items()}
        return out


def build_timm_model(cfg, num_classes: dict):
    """
    Wrapper to build the timm-based multi-head model.
    You can optionally add `backbone_name` to Config; defaults to convnext_tiny.
    """
    model_name = getattr(cfg, "backbone_name", "convnext_tiny")
    print(f"Using timm backbone: {model_name}")
    return TimmMultiHead(num_classes, model_name=model_name)


# ----------------- Training logic ----------------- #

@torch.no_grad()
def top1_acc(logits, y):  # logits [B,C], y [B]
    return (logits.argmax(1) == y).float().sum().item(), y.numel()


def train():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Using device:", device)

    # Dataloaders + target info
    train_loader, val_loader, test_loader, num_classes, id2label, _ = build_dataloaders(cfg)

    # Build timm transfer model
    model = build_timm_model(cfg, num_classes).to(device)

    # Optimiser
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # One CE loss per head
    criterions = {k: nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
                  for k in num_classes}

    best_macro_acc = 0.0
    start_epoch = 1

    # ---- Separate save dir for timm checkpoints ----
    # You can optionally define cfg.save_dir_timm in Config.
    save_dir_timm = getattr(cfg, "save_dir_timm", None)
    if save_dir_timm is None:
        parent = getattr(cfg, "save_dir", "checkpoints")
        save_dir_timm = os.path.join(parent, "timm")

    os.makedirs(save_dir_timm, exist_ok=True)
    print(f"Saving checkpoints to: {save_dir_timm}")


    # ---- Checkpoint loading (NO auto-detect from old CNN checkpoints) ----
    checkpoint_path = cfg.resume_from if getattr(cfg, "resume_from", None) else None

    if checkpoint_path and os.path.exists(checkpoint_path):
        if getattr(cfg, "resume_training", False):
            # Resume training from saved epoch
            print(f"Loading checkpoint from {checkpoint_path}")
            start_epoch, best_macro_acc = load_checkpoint(model, optimizer, checkpoint_path, device)
            start_epoch += 1  # Resume from next epoch
            print(f"Resuming training from epoch {start_epoch} (best macro acc: {best_macro_acc:.4f})")
            # Create scheduler for remaining epochs
            remaining_epochs = cfg.epochs - start_epoch + 1
            scheduler = create_scheduler(
                optimizer,
                steps_per_epoch=len(train_loader),
                epochs=remaining_epochs,
                max_lr=cfg.lr,
            )
        else:
            # Load weights but restart from epoch 1
            print(f"Loading model weights from {checkpoint_path}")
            best_macro_acc = load_model_weights(model, checkpoint_path, device)
            start_epoch = 1
            print(f"Loaded weights (best macro acc: {best_macro_acc:.4f}). Restarting from epoch 1")
            # Create scheduler for full training
            scheduler = create_scheduler(
                optimizer,
                steps_per_epoch=len(train_loader),
                epochs=cfg.epochs,
                max_lr=cfg.lr,
            )
    else:
        if checkpoint_path and not os.path.exists(checkpoint_path):
            print(f"Warning: Specified checkpoint {checkpoint_path} not found. Starting from scratch.")
        print("Starting training from scratch (timm model)")
        start_epoch = 1
        scheduler = create_scheduler(
            optimizer,
            steps_per_epoch=len(train_loader),
            epochs=cfg.epochs,
            max_lr=cfg.lr,
        )

    # ----------------- Main training loop ----------------- #
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            x = batch["pixel_values"].to(device)
            labels = {k: batch[f"label_{k}"].to(device) for k in num_classes.keys()}

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)  # dict: k -> [B, Ck]

            # Weighted sum of losses
            loss = 0.0
            for (k, w) in zip(num_classes.keys(), cfg.loss_weights):
                loss = loss + w * criterions[k](logits[k], labels[k])

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss/len(pbar):.4f}")

        # ---- Validation ----
        model.eval()
        head_acc_sums = {k: 0.0 for k in num_classes}
        head_counts   = {k: 0   for k in num_classes}

        with torch.no_grad():
            for batch in tqdm(val_loader, leave=False, desc="Validating"):
                x = batch["pixel_values"].to(device)
                labels = {k: batch[f"label_{k}"].to(device) for k in num_classes.keys()}

                logits = model(x)

                for k in num_classes:
                    correct, total = top1_acc(logits[k], labels[k])
                    head_acc_sums[k] += correct
                    head_counts[k]   += total

        head_acc = {k: (head_acc_sums[k] / head_counts[k]) for k in num_classes}
        macro_acc = sum(head_acc.values()) / len(head_acc)

        print(
            f"Epoch {epoch}: "
            + "  ".join([f"{k}_acc={head_acc[k]:.4f}" for k in head_acc])
            + f"  macro_acc={macro_acc:.4f}"
        )

        if macro_acc > best_macro_acc:
            best_macro_acc = macro_acc
            ckpt = os.path.join(cfg.save_dir, f"best_timm_macro{best_macro_acc:.4f}.pt")
            save_checkpoint(model, optimizer, epoch, best_macro_acc, ckpt)
            print(f"Saved checkpoint to {ckpt}")

    print(f"\nTraining completed!")
    print(f"Best macro val acc: {best_macro_acc:.4f}")

    # Final test evaluation
    print(f"\nEvaluating on test set...")
    model.eval()
    test_head_acc_sums = {k: 0.0 for k in num_classes}
    test_head_counts   = {k: 0   for k in num_classes}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch["pixel_values"].to(device)
            labels = {k: batch[f"label_{k}"].to(device) for k in num_classes.keys()}

            logits = model(x)

            for k in num_classes:
                correct, total = top1_acc(logits[k], labels[k])
                test_head_acc_sums[k] += correct
                test_head_counts[k]   += total

    test_head_acc = {k: (test_head_acc_sums[k] / test_head_counts[k]) for k in num_classes}
    test_macro_acc = sum(test_head_acc.values()) / len(test_head_acc)

    print(f"Test Results:")
    print("   " + "  ".join([f"{k}_acc={test_head_acc[k]:.4f}" for k in test_head_acc]))
    print(f"   macro_acc={test_macro_acc:.4f}")


if __name__ == "__main__":
    train()
