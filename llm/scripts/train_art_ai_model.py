"""
IMPROVED Art + AI Literacy Model Training
==========================================
Fixes overfitting issues from train_model1.py

KEY CHANGES:
- Lower LR: 2e-4 (was 6e-4 - too high!)
- Higher dropout: 0.2 (was 0.1 - too low!)
- Smaller dataset: ~310K (was 5M - too much!)
- Better early stopping: patience=3
- Art + AI literacy focused datasets
"""

# SET CACHE DIRECTORIES FIRST
import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

from models.modern_transformer import ModernTransformer
from utils.art_ai_dataset_loader import load_art_ai_literacy_datasets, TextDataset


class ArtAIConfig:
    """IMPROVED Configuration - Fixes overfitting"""

    # Model Architecture - LARGE (GPT-2 Large equivalent ~774M params)
    DIM = 1280              # Increased from 1024
    N_LAYERS = 24           # Increased from 12 (doubled!)
    N_HEADS = 20            # Increased from 16
    N_KV_HEADS = 4          # Keep GQA
    MAX_SEQ_LEN = 2048
    DROPOUT = 0.2

    # Dataset Sizes - MILLION SCALE
    ART_KNOWLEDGE_SAMPLES = 150000      # WikiArt + Wikipedia + ArtBench
    AI_LITERACY_SAMPLES = 300000        # ELI5 + SQuAD + OpenAssistant
    CONVERSATIONAL_SAMPLES = 250000     # PersonaChat + DailyDialog
    IMAGE_CAPTIONS_SAMPLES = 100000     # COCO (for CNN integration)
    C4_WEB_SAMPLES = 200000             # High quality web text
    # Total: ~1M samples

    # Tokenizer - Using EleutherAI (better, no auth needed!)
    TOKENIZER_NAME = "EleutherAI/gpt-neo-125m"  # Better than GPT-2, 50K vocab
    MAX_LEN = 1024

    # Training - IMPROVED HYPERPARAMETERS
    NUM_EPOCHS = 50
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4     # ✓ Effective batch=32 (was 64)
    LEARNING_RATE = 2e-4                # ✓ REDUCED from 6e-4
    WEIGHT_DECAY = 0.1
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 3000                 # ✓ REDUCED from 10K
    LABEL_SMOOTHING = 0.15              # ✓ INCREASED from 0.1
    PATIENCE = 3                        # ✓ AGGRESSIVE (was 20!)

    TRAIN_SPLIT = 0.95
    VAL_SPLIT = 0.05

    CHECKPOINT_DIR = "/cs/student/projects1/2023/muhamaaz/checkpoints/art_ai_model"
    LOG_DIR = "/cs/student/projects1/2023/muhamaaz/logs"
    CHECKPOINT_INTERVAL = 2
    EVAL_INTERVAL = 1

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4


def setup_logging(log_dir: str):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"art_ai_training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, tokenizer, config, filepath: str,
                   train_losses=None, val_losses=None, val_perplexities=None):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'tokenizer_name': config.TOKENIZER_NAME,
        'train_losses': train_losses or [],
        'val_losses': val_losses or [],
        'val_perplexities': val_perplexities or [],
        'config': {
            'dim': config.DIM,
            'n_layers': config.N_LAYERS,
            'n_heads': config.N_HEADS,
            'n_kv_heads': config.N_KV_HEADS,
            'max_seq_len': config.MAX_SEQ_LEN,
            'dropout': config.DROPOUT,
            'vocab_size': tokenizer.vocab_size,
        }
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved: {filepath}")


def load_latest_checkpoint(checkpoint_dir: str):
    """Find and load latest checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
    latest = checkpoints[-1]
    latest_path = os.path.join(checkpoint_dir, latest)

    logging.info(f"Found checkpoint: {latest}")
    return torch.load(latest_path, map_location='cpu')


def create_lr_scheduler(optimizer, num_training_steps: int, warmup_steps: int):
    """LR scheduler with warmup and cosine decay"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip: float, logger):
    """Train one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits, loss = model(inputs, targets)

        if loss is None:
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})

        if batch_idx % 100 == 0:
            logger.info(f"Step {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def validate_epoch(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)

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

    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot perplexity
    ax2.plot(epochs, val_perplexities, 'g-', label='Val Perplexity', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Validation Perplexity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nTraining curves saved to: {save_path}")


def main():
    """Main training function"""
    config = ArtAIConfig()
    logger = setup_logging(config.LOG_DIR)

    print("=" * 80)
    print("ART + AI LITERACY CHATBOT TRAINING")
    print("=" * 80)
    print("Coursework: COMP0220 - Deep Learning for Robotics and AI")
    print("Theme: AI Literacy for Art Education")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    print("\nIMPROVEMENTS APPLIED:")
    print("✓ Learning Rate: 2e-4 (was 6e-4 - TOO HIGH)")
    print("✓ Dropout: 0.2 (was 0.1 - TOO LOW)")
    print("✓ Dataset: ~310K samples (was 5M - TOO MUCH)")
    print("✓ Early Stopping: patience=3 (was 20 - TOO LONG)")
    print("✓ Label Smoothing: 0.15 (was 0.1)")
    print("=" * 80)

    logger.info("Starting training...")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocab size: {vocab_size:,}")

    # Load datasets
    logger.info("Loading Art + AI Literacy datasets...")
    start_time = time.time()

    all_texts = load_art_ai_literacy_datasets(
        art_knowledge_samples=config.ART_KNOWLEDGE_SAMPLES,
        ai_literacy_samples=config.AI_LITERACY_SAMPLES,
        conversational_samples=config.CONVERSATIONAL_SAMPLES,
        image_captions_samples=config.IMAGE_CAPTIONS_SAMPLES + config.C4_WEB_SAMPLES  # Include C4
    )
    logger.info(f"Loaded {len(all_texts):,} samples in {time.time() - start_time:.1f}s")

    # Split datasets
    split_idx = int(len(all_texts) * config.TRAIN_SPLIT)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]

    logger.info(f"Train: {len(train_texts):,} samples")
    logger.info(f"Val: {len(val_texts):,} samples")

    # Create dataloaders
    train_dataset = TextDataset(train_texts, tokenizer, max_len=config.MAX_LEN, augment=False)
    val_dataset = TextDataset(val_texts, tokenizer, max_len=config.MAX_LEN, augment=False)

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

    # Initialize model
    logger.info("Initializing model...")
    model = ModernTransformer(
        vocab_size=vocab_size,
        dim=config.DIM,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        n_kv_heads=config.N_KV_HEADS,
        max_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT,
        label_smoothing=config.LABEL_SMOOTHING
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params/1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # LR scheduler
    num_training_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = create_lr_scheduler(optimizer, num_training_steps, config.WARMUP_STEPS)

    # Try to resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_perplexities = []
    patience_counter = 0

    checkpoint = load_latest_checkpoint(config.CHECKPOINT_DIR)

    if checkpoint:
        logger.info(f"Resuming from epoch {checkpoint['epoch'] + 1}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_perplexities = checkpoint.get('val_perplexities', [])
        logger.info(f"Resumed from epoch {start_epoch}, best loss: {best_val_loss:.4f}")
    else:
        logger.info("Starting training from scratch")

    # Training loop
    training_start = time.time()

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start = time.time()

        logger.info(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, config.DEVICE,
            config.GRAD_CLIP, logger
        )
        train_losses.append(train_loss)

        # Validate
        val_loss, val_perplexity = validate_epoch(model, val_loader, config.DEVICE)
        val_losses.append(val_loss)
        val_perplexities.append(val_perplexity)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # Print results
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch+1}/{config.NUM_EPOCHS} COMPLETED")
        print("=" * 80)
        print(f"Train Loss:      {train_loss:.4f}")
        print(f"Val Loss:        {val_loss:.4f}")
        print(f"Val Perplexity:  {val_perplexity:.2f}")
        print(f"Learning Rate:   {current_lr:.6f}")
        print(f"Epoch Time:      {epoch_time/60:.1f} min")
        print(f"Train-Val Gap:   {val_loss - train_loss:.4f}")
        print("=" * 80 + "\n")

        logger.info(f"Epoch {epoch+1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(config.CHECKPOINT_DIR, "art_ai_best.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, tokenizer, config, best_path,
                          train_losses, val_losses, val_perplexities)
            logger.info(f"✓ NEW BEST MODEL: Val Loss {val_loss:.4f}, Perplexity {val_perplexity:.2f}")
        else:
            patience_counter += 1
            logger.info(f"⚠ No improvement for {patience_counter} epoch(s)")

            if patience_counter >= config.PATIENCE:
                logger.info(f"EARLY STOPPING after {patience_counter} epochs without improvement")
                print("\n" + "=" * 80)
                print(f"EARLY STOPPING: No improvement for {config.PATIENCE} epochs")
                print(f"Best Val Loss: {best_val_loss:.4f}")
                print(f"Best Perplexity: {np.exp(best_val_loss):.2f}")
                print("=" * 80 + "\n")
                break

        # Save periodic checkpoint
        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, tokenizer, config, checkpoint_path,
                          train_losses, val_losses, val_perplexities)

            # Save training plot
            plot_path = os.path.join(config.CHECKPOINT_DIR, "training_curves.png")
            plot_training_curves(train_losses, val_losses, val_perplexities, plot_path)

    # Final save
    final_path = os.path.join(config.CHECKPOINT_DIR, "art_ai_final.pt")
    save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, tokenizer, config, final_path,
                   train_losses, val_losses, val_perplexities)

    # Final plot
    plot_path = os.path.join(config.CHECKPOINT_DIR, "training_curves_final.png")
    plot_training_curves(train_losses, val_losses, val_perplexities, plot_path)

    total_time = time.time() - training_start
    logger.info(f"Training complete: {total_time/3600:.2f} hours")
    logger.info(f"Best perplexity: {np.exp(best_val_loss):.2f}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best model: art_ai_best.pt")
    print(f"Final perplexity: {np.exp(best_val_loss):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
