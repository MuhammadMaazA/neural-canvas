#!/usr/bin/env python3
"""
Training script for the from-scratch art expert transformer.
Uses WikiArt, ELI5, and OpenAssistant datasets for training.
"""

import os
from pathlib import Path

# Get project root (cross-platform)
SCRIPT_DIR = Path(__file__).resolve().parent
LLM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = LLM_DIR.parent

# Cross-platform HF cache - only set if server path exists
if os.path.exists('/cs/student/projects1/2023/muhamaaz/datasets'):
    os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
    os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
    os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
# else use default HuggingFace cache

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup
from llm.models.art_expert_model import ArtExpertTransformer
from llm.utils.curated_art_dataset import load_combined_art_datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TrainingConfig:
    """Training configuration"""
    # Model
    VOCAB_SIZE = 50257  # GPT-2 vocab
    DIM = 512
    N_LAYERS = 8
    N_HEADS = 8
    N_KV_HEADS = 2
    MAX_LEN = 512
    DROPOUT = 0.1
    LABEL_SMOOTHING = 0.1

    # Dataset config
    USE_WIKIART = True
    USE_ELI5 = True
    USE_OPENASSISTANT = True

    # Training
    NUM_EPOCHS = 20
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.1
    GRAD_CLIP = 1.0
    WARMUP_RATIO = 0.1

    # Early stopping
    PATIENCE = 5

    # Splits
    TRAIN_SPLIT = 0.95
    VAL_SPLIT = 0.05

    # Paths (cross-platform)
    @staticmethod
    def get_checkpoint_dir():
        if os.path.exists('/cs/student/projects1/2023/muhamaaz/checkpoints'):
            return "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_from_scratch"
        return str(PROJECT_ROOT / 'checkpoints' / 'cnn_explainer_from_scratch')
    
    @staticmethod
    def get_log_dir():
        if os.path.exists('/cs/student/projects1/2023/muhamaaz/logs'):
            return "/cs/student/projects1/2023/muhamaaz/logs"
        return str(PROJECT_ROOT / 'logs')
    
    CHECKPOINT_DIR = None  # Set dynamically
    LOG_DIR = None  # Set dynamically

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    MIXED_PRECISION = True

    # Logging
    LOG_INTERVAL = 100
    EVAL_INTERVAL = 1


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"cnn_explainer_scratch_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, config, filepath,
                   train_losses=None, val_losses=None, val_perplexities=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
        'train_losses': train_losses or [],
        'val_losses': val_losses or [],
        'val_perplexities': val_perplexities or [],
        'config': {
            'vocab_size': config.VOCAB_SIZE,
            'dim': config.DIM,
            'n_layers': config.N_LAYERS,
            'n_heads': config.N_HEADS,
            'n_kv_heads': config.N_KV_HEADS,
            'max_len': config.MAX_LEN,
            'dropout': config.DROPOUT
        }
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved: {filepath}")


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, config, logger):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
            logits, loss = model(inputs, targets)

        if loss is None:
            continue

        loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        num_batches += 1

        current_lr = scheduler.get_last_lr()[0] if scheduler else config.LEARNING_RATE
        pbar.set_postfix({
            'loss': f"{loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}",
            'lr': f"{current_lr:.6f}"
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


@torch.no_grad()
def validate_epoch(model, dataloader, device, config):
    model.eval()
    total_loss = 0
    num_batches = 0

    for inputs, targets in tqdm(dataloader, desc="Validation"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
            logits, loss = model(inputs, targets)

        if loss is None:
            continue

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')

    return avg_loss, perplexity


def plot_training_curves(train_losses, val_losses, val_perplexities, save_path):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('CNN Explainer Training - From Scratch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_perplexities, 'g-', label='Val Perplexity', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Validation Perplexity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    config = TrainingConfig()
    logger = setup_logging(config.LOG_DIR)

    print("\n" + "=" * 80)
    print("CNN EXPLAINER - FROM SCRATCH TRAINING")
    print("=" * 80)
    print("Task: Train LLM for art + AI literacy (CNN explanation)")
    print("Model: Custom Art Expert Transformer (56M params)")
    print("Datasets: 3 sources (WikiArt + ELI5 + OpenAssistant = ~210K)")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mixed Precision: {config.MIXED_PRECISION}")
    print("=" * 80 + "\n")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    logger.info("Creating model...")
    model = ArtExpertTransformer(
        vocab_size=config.VOCAB_SIZE,
        dim=config.DIM,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        n_kv_heads=config.N_KV_HEADS,
        max_len=config.MAX_LEN,
        dropout=config.DROPOUT,
        label_smoothing=config.LABEL_SMOOTHING
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params/1e6:.1f}M")

    # Load datasets
    logger.info("Loading datasets...")
    
    full_dataset = load_combined_art_datasets(
        tokenizer=tokenizer,
        max_len=config.MAX_LEN,
        use_wikiart=config.USE_WIKIART,
        use_eli5=config.USE_ELI5,
        use_openassistant=config.USE_OPENASSISTANT
    )
    
    logger.info(f"Total combined samples: {len(full_dataset):,}")

    # Split
    train_size = int(len(full_dataset) * config.TRAIN_SPLIT)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Scheduler
    num_training_steps = len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS * config.NUM_EPOCHS
    warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    logger.info(f"\nTraining steps: {num_training_steps:,}")
    logger.info(f"Warmup steps: {warmup_steps:,}")

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION and config.DEVICE == 'cuda' else None

    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_perplexities = []
    patience_counter = 0

    # Training loop
    logger.info("\nStarting training...")
    training_start = time.time()

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start = time.time()

        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch+1}/{config.NUM_EPOCHS}")
        logger.info(f"{'='*80}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, config.DEVICE, config, logger)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_perplexity = validate_epoch(model, val_loader, config.DEVICE, config)
        val_losses.append(val_loss)
        val_perplexities.append(val_perplexity)

        epoch_time = time.time() - epoch_start

        # Print results
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{config.NUM_EPOCHS} RESULTS")
        print(f"{'='*80}")
        print(f"Train Loss:       {train_loss:.4f}")
        print(f"Val Loss:         {val_loss:.4f}")
        print(f"Val Perplexity:   {val_perplexity:.2f}")
        print(f"Epoch Time:       {epoch_time/60:.1f} min")
        print(f"{'='*80}\n")

        # Check improvement
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0

            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, config, best_path,
                          train_losses, val_losses, val_perplexities)
            logger.info(f"New best model saved (improved by {improvement:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epoch(s)")

            if patience_counter >= config.PATIENCE:
                logger.info(f"\nEARLY STOPPING")
                break

        # Save latest
        latest_path = os.path.join(config.CHECKPOINT_DIR, "latest.pt")
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, config, latest_path,
                       train_losses, val_losses, val_perplexities)

        # Plot
        plot_path = os.path.join(config.CHECKPOINT_DIR, "training_curves.png")
        plot_training_curves(train_losses, val_losses, val_perplexities, plot_path)

    total_time = time.time() - training_start

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total Time:       {total_time/3600:.2f} hours")
    print(f"Best Val Loss:    {best_val_loss:.4f}")
    print(f"Best Perplexity:  {np.exp(best_val_loss):.2f}")
    print(f"Best Model:       {os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
