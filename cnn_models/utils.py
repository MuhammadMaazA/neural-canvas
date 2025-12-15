import os
import math
import glob
import torch
from torch.optim.lr_scheduler import OneCycleLR

def create_scheduler(optimizer, steps_per_epoch, epochs, max_lr):
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0
    )

def save_checkpoint(model, optimizer, epoch, acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": acc
    }, path)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint and return epoch and validation accuracy."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    val_acc = checkpoint["val_acc"]
    return epoch, val_acc

def load_model_weights(model, checkpoint_path, device):
    """Load only model weights from checkpoint and return validation accuracy."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    val_acc = checkpoint.get("val_acc", 0.0)
    return val_acc

def find_best_checkpoint(save_dir):
    """Find the checkpoint with the highest macro accuracy."""
    pattern = os.path.join(save_dir, "best_multitask_macro*.pt")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Extract accuracy from filename: best_multitask_macro0.5949.pt -> 0.5949
    def get_acc(path):
        try:
            filename = os.path.basename(path)
            # Extract number after "macro"
            acc_str = filename.replace("best_multitask_macro", "").replace(".pt", "")
            return float(acc_str)
        except:
            return -1.0
    
    # Sort by accuracy (descending) and return the best one
    checkpoints.sort(key=get_acc, reverse=True)
    return checkpoints[0] if checkpoints else None
