from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    dataset_name: str = "huggan/wikiart"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 0          # keep 0 on macOS while debugging
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 25
    seed: int = 42
    model_name: str = "custom_cnn"
    use_amp: bool = False         # keep False on MPS for simplicity
    save_dir: str = "checkpoints"
    label_smoothing: float = 0.05
    dropout: float = 0.3
    # ---- NEW: multi-task ----
    multitask: bool = True
    targets: tuple = ("artist", "style", "genre")
    loss_weights: tuple = (1.0, 0.5, 0.5)   # tune if needed
    # ---- Debug subset (set to None for full dataset) ----
    debug_train_limit: Optional[int] = None
    debug_val_limit: Optional[int] = None
    debug_test_limit: Optional[int] = None
    # ---- Checkpoint resuming ----
    resume_from: Optional[str] = None  # Path to checkpoint file, or None to auto-detect best
    resume_training: bool = False      # If True, resume from saved epoch. If False, load weights but restart from epoch 1
