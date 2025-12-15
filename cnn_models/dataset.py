import random
from typing import Callable, Tuple, Sequence
from collections import defaultdict
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
import torch

def build_transforms(cfg: Config) -> Tuple[Callable, Callable]:
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    val_t = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return train_t, val_t

def _set_transform(ds, tfm: Callable, target_keys: Sequence[str]):
    def _tfm(example):
        imgs = example["image"]

        def to_out(img):
            return tfm(img.convert("RGB"))

        if isinstance(imgs, list):  # batched
            px = torch.stack([to_out(img) for img in imgs])
            out = {"pixel_values": px}
            for k in target_keys:
                out[f"label_{k}"] = torch.tensor([int(v) for v in example[k]], dtype=torch.long)
            return out
        else:  # single
            out = {"pixel_values": to_out(imgs)}
            for k in target_keys:
                out[f"label_{k}"] = int(example[k])
            return out

    ds.set_transform(_tfm)
    return ds

def build_dataloaders(cfg: Config):
    raw = load_dataset(cfg.dataset_name, split="train")
    
    total_size = len(raw)
    print(f"Total dataset size: {total_size} samples")

    # pick target columns
    target_keys = cfg.targets if cfg.multitask else ("artist",)

    # stratified split by the PRIMARY target (artist)
    # Split: 80% train, 10% validation, 10% test
    labels = raw["artist"]
    idx = list(range(len(raw)))
    random.Random(cfg.seed).shuffle(idx)

    by_class = defaultdict(list)
    for i in idx:
        by_class[labels[i]].append(i)

    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    
    train_idx, val_idx, test_idx = [], [], []
    for _, indices in by_class.items():
        n_total = len(indices)
        n_train = max(1, int(n_total * train_split))
        n_val = max(1, int(n_total * val_split))
        # Remaining goes to test
        n_test = n_total - n_train - n_val
        
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    train_ds = raw.select(train_idx)
    val_ds   = raw.select(val_idx)
    test_ds  = raw.select(test_idx)
    
    print(f"📈 Dataset splits:")
    print(f"   Training:   {len(train_ds)} samples ({100*len(train_ds)/total_size:.1f}%)")
    print(f"   Validation: {len(val_ds)} samples ({100*len(val_ds)/total_size:.1f}%)")
    print(f"   Testing:    {len(test_ds)} samples ({100*len(test_ds)/total_size:.1f}%)")

    # optional debug limits: shuffle then select to avoid bias
    if getattr(cfg, "debug_train_limit", None):
        train_ds = train_ds.shuffle(seed=cfg.seed).select(range(cfg.debug_train_limit))
    if getattr(cfg, "debug_val_limit", None):
        val_ds = val_ds.shuffle(seed=cfg.seed).select(range(cfg.debug_val_limit))
    if getattr(cfg, "debug_test_limit", None):
        test_ds = test_ds.shuffle(seed=cfg.seed).select(range(cfg.debug_test_limit))

    # transforms
    train_t, val_t = build_transforms(cfg)
    train_ds = _set_transform(train_ds, train_t, target_keys)
    val_ds   = _set_transform(val_ds,   val_t,   target_keys)
    test_ds  = _set_transform(test_ds,  val_t,   target_keys)  # test uses same transforms as val

    # loaders (default collate is fine because we return only tensors)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False
    )

    # per-target class counts and label maps
    num_classes = {k: raw.features[k].num_classes for k in target_keys}
    id2label = {k: {i: n for i, n in enumerate(raw.features[k].names)} for k in target_keys}
    label2id = {k: {n: i for i, n in id2label[k].items()} for k in target_keys}

    return train_loader, val_loader, test_loader, num_classes, id2label, label2id
