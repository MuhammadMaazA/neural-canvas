"""
Training Script for Art Expert Transformer (Model 1)
====================================================
Trains a 35-50M parameter transformer from scratch on curated art + AI literacy datasets

This is MODEL 1 (trained from scratch) for coursework comparison with MODEL 2 (pretrained)
"""

# Set cache directories FIRST
import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import json
import time
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.art_expert_model import create_art_expert_model
from utils.curated_art_dataset import load_curated_art_datasets, TextDataset


class TrainingConfig:
    """Optimized training configuration for 35-50M model"""

    # Model
    MODEL_SIZE = "base"  # "small" (20M), "base" (35M), or "large" (50M)
    TOKENIZER_NAME = "gpt2"  # GPT-2 tokenizer (50K vocab, no auth needed)
    MAX_SEQ_LEN = 512  # Shorter = faster training

    # Dataset sizes (EXPANDED for better learning)
    ART_KNOWLEDGE = 120000   # Increased from 80K
    AI_LITERACY = 150000     # Increased from 75K
    CONVERSATIONAL = 150000  # Increased from 80K
    # Total: ~420K samples (was 235K)

    # Training hyperparameters (OPTIMIZED for small models)
    NUM_EPOCHS = 30
    BATCH_SIZE = 16  # Larger batch for stability
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 32
    LEARNING_RATE = 3e-4  # Sweet spot for transformers
    WEIGHT_DECAY = 0.1
    GRAD_CLIP = 1.0
    WARMUP_RATIO = 0.1  # 10% warmup
    LABEL_SMOOTHING = 0.1

    # Early stopping
    PATIENCE = 5  # Stop if no improvement for 5 epochs

    # Validation split
    TRAIN_SPLIT = 0.95
    VAL_SPLIT = 0.05

    # Paths
    CHECKPOINT_DIR = "/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom"
    LOG_DIR = "/cs/student/projects1/2023/muhamaaz/logs"

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    MIXED_PRECISION = True  # Use FP16 for faster training

    # Logging
    LOG_INTERVAL = 100
    EVAL_INTERVAL = 1  # Validate every epoch
    SAVE_INTERVAL = 2  # Save checkpoint every 2 epochs


def setup_logging(log_dir: str):
    """Setup logging to file and console"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model1_training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, config, filepath: str,
                   train_losses=None, val_losses=None, val_perplexities=None):
    """Save training checkpoint"""
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
            'model_size': config.MODEL_SIZE,
            'tokenizer_name': config.TOKENIZER_NAME,
            'max_seq_len': config.MAX_SEQ_LEN,
        }
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str):
    """Load checkpoint"""
    if not os.path.exists(filepath):
        return None
    return torch.load(filepath, map_location='cpu')


def create_lr_scheduler(optimizer, num_training_steps: int, warmup_steps: int):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, config, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
            logits, loss = model(inputs, targets)

        if loss is None:
            continue

        # Scale loss for gradient accumulation
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
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
            scheduler.step()

        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        num_batches += 1

        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'loss': f"{loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}",
            'lr': f"{current_lr:.6f}"
        })

        # Periodic logging
        if batch_idx % config.LOG_INTERVAL == 0:
            logger.info(f"Step {batch_idx}/{len(dataloader)}, Loss: {loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}, LR: {current_lr:.6f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


@torch.no_grad()
def validate_epoch(model, dataloader, device, config):
    """Validate model"""
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
    """Plot and save training curves"""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Progress - Art Expert Model 1', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Perplexity plot
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
    """Main training function"""
    config = TrainingConfig()
    logger = setup_logging(config.LOG_DIR)

    # Print header
    print("\n" + "=" * 80)
    print("NEURAL CANVAS - MODEL 1 TRAINING")
    print("Art Expert Transformer (Trained from Scratch)")
    print("=" * 80)
    print("Coursework: COMP0220 Deep Learning for Robotics and AI")
    print("Student: Muhammad Maaz")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model Size: {config.MODEL_SIZE.upper()}")
    print(f"Mixed Precision: {config.MIXED_PRECISION}")
    print("=" * 80 + "\n")

    logger.info("Starting Model 1 training...")

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocabulary size: {vocab_size:,}")

    # Load datasets
    logger.info("\nLoading curated datasets...")
    start_time = time.time()

    all_texts, stats = load_curated_art_datasets(
        art_knowledge=config.ART_KNOWLEDGE,
        ai_literacy=config.AI_LITERACY,
        conversational=config.CONVERSATIONAL
    )

    logger.info(f"Loaded {len(all_texts):,} samples in {time.time() - start_time:.1f}s")

    # Create dataset
    full_dataset = TextDataset(all_texts, tokenizer, max_len=config.MAX_SEQ_LEN)

    # Split train/val
    train_size = int(len(full_dataset) * config.TRAIN_SPLIT)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")

    # Create dataloaders
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

    # Create model
    logger.info("\nInitializing model...")
    model = create_art_expert_model(vocab_size, config.MODEL_SIZE).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params/1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Learning rate scheduler
    num_training_steps = len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS * config.NUM_EPOCHS
    warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
    scheduler = create_lr_scheduler(optimizer, num_training_steps, warmup_steps)

    logger.info(f"Training steps: {num_training_steps:,}")
    logger.info(f"Warmup steps: {warmup_steps:,}")

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION and config.DEVICE == 'cuda' else None

    # Try to resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_perplexities = []
    patience_counter = 0

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest.pt")
    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint:
        logger.info(f"Resuming from checkpoint...")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_perplexities = checkpoint.get('val_perplexities', [])

        logger.info(f"Resumed from epoch {start_epoch}, best loss: {best_val_loss:.4f}")
    else:
        logger.info("Starting training from scratch")

    # Training loop
    logger.info("\nStarting training loop...")
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
        current_lr = optimizer.param_groups[0]['lr']

        # Print results
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{config.NUM_EPOCHS} RESULTS")
        print(f"{'='*80}")
        print(f"Train Loss:       {train_loss:.4f}")
        print(f"Val Loss:         {val_loss:.4f}")
        print(f"Val Perplexity:   {val_perplexity:.2f}")
        print(f"Learning Rate:    {current_lr:.6f}")
        print(f"Epoch Time:       {epoch_time/60:.1f} min")
        print(f"Train-Val Gap:    {val_loss - train_loss:.4f}")
        print(f"{'='*80}\n")

        logger.info(f"Epoch {epoch+1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, PPL: {val_perplexity:.2f}")

        # Check for improvement
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, config, best_path,
                          train_losses, val_losses, val_perplexities)

            logger.info(f"✓ NEW BEST MODEL - Improved by {improvement:.4f}")
        else:
            patience_counter += 1
            logger.info(f"⚠ No improvement for {patience_counter} epoch(s)")

            if patience_counter >= config.PATIENCE:
                logger.info(f"\nEARLY STOPPING - No improvement for {config.PATIENCE} epochs")
                break

        # Save latest checkpoint
        latest_path = os.path.join(config.CHECKPOINT_DIR, "latest.pt")
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, config, latest_path,
                       train_losses, val_losses, val_perplexities)

        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            periodic_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, config, periodic_path,
                          train_losses, val_losses, val_perplexities)

        # Update plot
        plot_path = os.path.join(config.CHECKPOINT_DIR, "training_curves.png")
        plot_training_curves(train_losses, val_losses, val_perplexities, plot_path)

    # Training complete
    total_time = time.time() - training_start

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total Time:       {total_time/3600:.2f} hours")
    print(f"Best Val Loss:    {best_val_loss:.4f}")
    print(f"Best Perplexity:  {np.exp(best_val_loss):.2f}")
    print(f"Epochs Trained:   {epoch+1}")
    print(f"Best Model:       {os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')}")
    print(f"{'='*80}\n")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
