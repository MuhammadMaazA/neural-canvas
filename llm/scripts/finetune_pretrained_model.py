"""
Fine-tuning Script for Pretrained Model (Model 2)
==================================================
Fine-tunes GPT-2 (124M params) or DistilGPT-2 (82M params) on art + AI literacy datasets

This is MODEL 2 (pretrained + fine-tuned) for coursework comparison with MODEL 1 (from scratch)

Purpose:
- Show transfer learning benefits
- Compare with custom model
- Demonstrate fine-tuning best practices
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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.curated_art_dataset import load_curated_art_datasets, TextDataset


class FineTuningConfig:
    """Optimized configuration for fine-tuning pretrained models"""

    # Model selection
    # Options: "gpt2" (124M), "distilgpt2" (82M), "gpt2-medium" (355M - if GPU allows)
    MODEL_NAME = "distilgpt2"  # Faster, lighter than GPT-2
    MAX_SEQ_LEN = 512

    # Dataset sizes (same as Model 1 for fair comparison)
    ART_KNOWLEDGE = 80000
    AI_LITERACY = 75000
    CONVERSATIONAL = 80000

    # Fine-tuning hyperparameters (conservative for pretrained models)
    NUM_EPOCHS = 10  # Fewer epochs needed (already trained!)
    BATCH_SIZE = 8  # Smaller batch for larger model
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 32
    LEARNING_RATE = 2e-5  # MUCH lower for fine-tuning
    WEIGHT_DECAY = 0.01
    GRAD_CLIP = 1.0
    WARMUP_RATIO = 0.05  # Less warmup needed
    LABEL_SMOOTHING = 0.0  # No label smoothing for fine-tuning

    # Early stopping
    PATIENCE = 3  # Converges faster

    # Validation split
    TRAIN_SPLIT = 0.95
    VAL_SPLIT = 0.05

    # Paths
    CHECKPOINT_DIR = "/cs/student/projects1/2023/muhamaaz/checkpoints/model2_pretrained"
    LOG_DIR = "/cs/student/projects1/2023/muhamaaz/logs"

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    MIXED_PRECISION = True

    # Logging
    LOG_INTERVAL = 100
    EVAL_INTERVAL = 1
    SAVE_INTERVAL = 2


def setup_logging(log_dir: str):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model2_finetuning_{timestamp}.log")

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
    """Save checkpoint"""
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
            'model_name': config.MODEL_NAME,
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


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, config, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Fine-tuning")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss

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
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        num_batches += 1

        # Update progress bar
        current_lr = scheduler.get_last_lr()[0] if scheduler else config.LEARNING_RATE
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
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss

        if loss is None:
            continue

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')

    return avg_loss, perplexity


def plot_training_curves(train_losses, val_losses, val_perplexities, save_path):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Fine-tuning Progress - Model 2 (Pretrained)', fontsize=14, fontweight='bold')
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
    """Main fine-tuning function"""
    config = FineTuningConfig()
    logger = setup_logging(config.LOG_DIR)

    # Print header
    print("\n" + "=" * 80)
    print("NEURAL CANVAS - MODEL 2 FINE-TUNING")
    print(f"Fine-tuning {config.MODEL_NAME} on Art + AI Literacy")
    print("=" * 80)
    print("Coursework: COMP0220 Deep Learning for Robotics and AI")
    print("Student: Muhammad Maaz")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Base Model: {config.MODEL_NAME}")
    print(f"Mixed Precision: {config.MIXED_PRECISION}")
    print("=" * 80 + "\n")

    logger.info("Starting Model 2 fine-tuning...")

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocabulary size: {vocab_size:,}")

    # Load pretrained model
    logger.info(f"\nLoading pretrained model: {config.MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
    model = model.to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params/1e6:.1f}M")

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

    # Optimizer (lower LR for fine-tuning!)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    num_training_steps = len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS * config.NUM_EPOCHS
    warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    logger.info(f"\nTraining steps: {num_training_steps:,}")
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
        logger.info("Starting fine-tuning from pretrained weights")

    # Fine-tuning loop
    logger.info("\nStarting fine-tuning loop...")
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

            # Also save as HuggingFace format for easy loading
            hf_path = os.path.join(config.CHECKPOINT_DIR, "best_model_hf")
            model.save_pretrained(hf_path)
            tokenizer.save_pretrained(hf_path)

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

        # Update plot
        plot_path = os.path.join(config.CHECKPOINT_DIR, "training_curves.png")
        plot_training_curves(train_losses, val_losses, val_perplexities, plot_path)

    # Training complete
    total_time = time.time() - training_start

    print(f"\n{'='*80}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Total Time:       {total_time/3600:.2f} hours")
    print(f"Best Val Loss:    {best_val_loss:.4f}")
    print(f"Best Perplexity:  {np.exp(best_val_loss):.2f}")
    print(f"Epochs Trained:   {epoch+1}")
    print(f"Best Model:       {os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')}")
    print(f"HF Format:        {os.path.join(config.CHECKPOINT_DIR, 'best_model_hf')}")
    print(f"{'='*80}\n")

    logger.info("Fine-tuning complete!")


if __name__ == "__main__":
    main()
