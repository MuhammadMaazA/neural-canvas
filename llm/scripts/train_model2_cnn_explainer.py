#!/usr/bin/env python3
"""
Fine-tuning script for GPT-2 Medium on art explanation task.
"""

# Set cache directories FIRST
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

import sys
import torch
from torch.utils.data import DataLoader, random_split
import time
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path (cross-platform)
sys.path.insert(0, str(LLM_DIR))

from utils.cnn_explainer_dataset import load_cnn_explainer_dataset


class FineTuningConfig:
    """Configuration for Model 2 (fine-tuning)"""
    
    # Model - using GPT-2 Medium (better quality, 355M params)
    MODEL_NAME = "gpt2-medium"  # 355M params - better than distilgpt2 (82M) or gpt2 (124M)
    MAX_SEQ_LEN = 512
    
    # Dataset size
    DATASET_SIZE = "medium"  # "small", "medium", or "large"
    
    # Fine-tuning hyperparameters (more conservative than training from scratch)
    NUM_EPOCHS = 8  # Fewer epochs needed (converges faster with better model)
    BATCH_SIZE = 4   # Smaller batch for larger model (355M needs more memory)
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 32 (4*8)
    LEARNING_RATE = 2e-5  # MUCH lower for fine-tuning!
    WEIGHT_DECAY = 0.01
    GRAD_CLIP = 1.0
    WARMUP_RATIO = 0.05
    
    # Early stopping
    PATIENCE = 3
    
    # Validation
    TRAIN_SPLIT = 0.95
    VAL_SPLIT = 0.05
    
    # Paths (cross-platform)
    @staticmethod
    def get_checkpoint_dir():
        if os.path.exists('/cs/student/projects1/2023/muhamaaz/checkpoints'):
            return "/cs/student/projects1/2023/muhamaaz/checkpoints/model2_cnn_explainer_gpt2medium"
        return str(PROJECT_ROOT / 'checkpoints' / 'model2_cnn_explainer_gpt2medium')
    
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
    SAVE_INTERVAL = 2


def setup_logging(log_dir: str):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model2_cnn_explainer_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, epoch, loss, config, filepath,
                   train_losses=None, val_losses=None, val_perplexities=None, save_hf=False):
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
    logging.info(f"Saved checkpoint: {filepath}")
    
    # Also save in HuggingFace format for easy loading
    if save_hf:
        hf_path = filepath.replace('.pt', '_hf')
        model.save_pretrained(hf_path)
        tokenizer.save_pretrained(hf_path)
        logging.info(f"Saved HuggingFace format: {hf_path}")


def load_checkpoint(filepath: str):
    """Load checkpoint"""
    if not os.path.exists(filepath):
        return None
    return torch.load(filepath, map_location='cpu', weights_only=False)


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, config, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Fine-tuning")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
        
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
        
        if batch_idx % config.LOG_INTERVAL == 0:
            logger.info(f"Step {batch_idx}/{len(dataloader)}, Loss: {loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0


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
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model 2 (Fine-tuned) - CNN Explainer', fontsize=14, fontweight='bold')
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
    """Main fine-tuning function"""
    config = FineTuningConfig()
    logger = setup_logging(config.LOG_DIR)
    
    # Print header
    print("\n" + "=" * 80)
    print("MODEL 2: CNN EXPLAINER (FINE-TUNED)")
    print("=" * 80)
    print(f"Base Model: {config.MODEL_NAME}")
    print("Purpose: Explain CNN art classification outputs")
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80 + "\n")
    
    logger.info("Starting Model 2 (Fine-tuned CNN Explainer) training...")
    
    # Create directories
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
    
    # Load dataset
    logger.info(f"\nLoading CNN Explainer dataset (size: {config.DATASET_SIZE})...")
    start_time = time.time()
    
    full_dataset = load_cnn_explainer_dataset(
        tokenizer=tokenizer,
        max_len=config.MAX_SEQ_LEN,
        size=config.DATASET_SIZE
    )
    
    logger.info(f"Dataset loaded in {time.time() - start_time:.1f}s")
    
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
    
    # Try to resume
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_perplexities = []
    patience_counter = 0
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest.pt")
    checkpoint = load_checkpoint(checkpoint_path)
    
    if checkpoint:
        logger.info("Resuming from checkpoint...")
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
    
    # Training loop
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
            save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, epoch, val_loss, config, best_path,
                          train_losses, val_losses, val_perplexities, save_hf=True)
            
            logger.info(f"NEW BEST MODEL - Improved by {improvement:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epoch(s)")
            
            if patience_counter >= config.PATIENCE:
                logger.info(f"\nEARLY STOPPING - No improvement for {config.PATIENCE} epochs")
                break
        
        # Save latest
        latest_path = os.path.join(config.CHECKPOINT_DIR, "latest.pt")
        save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, epoch, val_loss, config, latest_path,
                       train_losses, val_losses, val_perplexities)
        
        # Plot
        plot_path = os.path.join(config.CHECKPOINT_DIR, "training_curves.png")
        plot_training_curves(train_losses, val_losses, val_perplexities, plot_path)
    
    # Done
    total_time = time.time() - training_start
    
    print(f"\n{'='*80}")
    print("FINE-TUNING COMPLETE - MODEL 2")
    print(f"{'='*80}")
    print(f"Total Time:       {total_time/3600:.2f} hours")
    print(f"Best Val Loss:    {best_val_loss:.4f}")
    print(f"Best Perplexity:  {np.exp(best_val_loss):.2f}")
    print(f"Best Model:       {os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')}")
    print(f"HF Format:        {os.path.join(config.CHECKPOINT_DIR, 'best_model_hf')}")
    print(f"{'='*80}\n")
    
    logger.info("Fine-tuning complete!")


if __name__ == "__main__":
    main()

