"""
UCL COMP0220 Coursework - Complete LLM Training Script
Model 1: GPT-2 Style Transformer FROM SCRATCH

This script implements a complete, professional-grade training pipeline:
- Loads 3 datasets: Cornell Movie Dialogs (100K), ELI5 (50K), Technical Q&A (30K)
- Trains Transformer from scratch (12 layers, 768 dim, 12 heads)
- Implements regularization: Dropout, L2 weight decay, Gradient Clipping
- Comprehensive evaluation: Perplexity, BLEU, Hallucination tracking
- Checkpointing and resume capability
- Detailed logging and progress tracking

Total: 180K samples, 30 epochs, ~6-8 hours on RTX 3090 Ti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
import time
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Import our modules
from llm.transformer_model import GPT2FromScratch
from llm.dataset_utils import load_all_datasets, build_vocab, TextDataset
from llm.model_evaluator import evaluate_model, calculate_perplexity


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training hyperparameters and settings"""
    
    # Model Architecture
    D_MODEL = 768
    N_HEADS = 12
    N_LAYERS = 12
    MAX_LEN = 256
    DROPOUT = 0.1
    
    # Dataset Configuration
    CORNELL_SAMPLES = 100000
    ELI5_SAMPLES = 50000
    TECH_SAMPLES = 30000
    MAX_VOCAB = 30000
    MIN_WORD_FREQ = 5
    
    # Training Configuration
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01  # L2 regularization
    GRAD_CLIP = 1.0
    WARMUP_EPOCHS = 2
    
    # Data Split
    TRAIN_SPLIT = 0.9
    VAL_SPLIT = 0.1
    
    # Checkpointing
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
    RESUME_FROM = None  # Path to checkpoint to resume from, or None
    
    # Evaluation
    EVAL_INTERVAL = 2  # Evaluate every N epochs
    NUM_EVAL_SAMPLES = 10
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4


# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging(log_dir: str = "logs"):
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, loss, vocab, config, filepath: str):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'vocab': vocab,
        'config': {
            'd_model': config.D_MODEL,
            'n_heads': config.N_HEADS,
            'n_layers': config.N_LAYERS,
            'max_len': config.MAX_LEN,
            'dropout': config.DROPOUT,
            'vocab_size': len(vocab),
        }
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, model, optimizer, scheduler, config):
    """Load training checkpoint"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    vocab = checkpoint['vocab']
    
    logging.info(f"Checkpoint loaded: {filepath}")
    logging.info(f"Resuming from epoch {start_epoch}")
    
    return start_epoch, vocab, checkpoint['loss']


def create_learning_rate_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 2):
    """Create learning rate scheduler with warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup: linear increase
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip: float, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits, loss = model(inputs, targets)
        
        if loss is None:
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Log periodically
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    scheduler.step()
    
    return avg_loss


def validate(model, dataloader, device, vocab_size: int, logger):
    """Validate model"""
    model.eval()
    perplexity = calculate_perplexity(model, dataloader, device, vocab_size)
    return perplexity


def main():
    """Main training function"""
    config = TrainingConfig()
    logger = setup_logging()
    
    # Print header
    print("=" * 80)
    print("UCL COMP0220 - MODEL 1: Transformer FROM SCRATCH")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 80)
    
    logger.info("Starting training...")
    logger.info(f"Configuration: {vars(config)}")
    
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # ========================================================================
    # LOAD DATASETS
    # ========================================================================
    logger.info("Loading datasets...")
    start_time = time.time()
    
    try:
        all_texts = load_all_datasets(
            cornell_samples=config.CORNELL_SAMPLES,
            eli5_samples=config.ELI5_SAMPLES,
            tech_samples=config.TECH_SAMPLES
        )
        logger.info(f"Loaded {len(all_texts):,} total samples in {time.time() - start_time:.1f}s")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise
    
    # ========================================================================
    # BUILD VOCABULARY
    # ========================================================================
    logger.info("Building vocabulary...")
    start_time = time.time()
    
    vocab = build_vocab(all_texts, max_vocab=config.MAX_VOCAB, min_freq=config.MIN_WORD_FREQ)
    vocab_size = len(vocab)
    logger.info(f"Vocabulary built: {vocab_size:,} tokens in {time.time() - start_time:.1f}s")
    
    # Save vocabulary
    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.json")
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    logger.info(f"Vocabulary saved: {vocab_path}")
    
    # ========================================================================
    # SPLIT DATASETS
    # ========================================================================
    logger.info("Splitting datasets...")
    split_idx = int(len(all_texts) * config.TRAIN_SPLIT)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    logger.info(f"Train: {len(train_texts):,} samples")
    logger.info(f"Val: {len(val_texts):,} samples")
    
    # ========================================================================
    # CREATE DATALOADERS
    # ========================================================================
    logger.info("Creating dataloaders...")
    
    train_dataset = TextDataset(train_texts, vocab, max_len=config.MAX_LEN)
    val_dataset = TextDataset(val_texts, vocab, max_len=config.MAX_LEN)
    
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
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # ========================================================================
    # INITIALIZE MODEL
    # ========================================================================
    logger.info("Initializing model...")
    
    model = GPT2FromScratch(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        max_len=config.MAX_LEN,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model initialized:")
    logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Architecture: {config.N_LAYERS} layers, {config.D_MODEL} dim, {config.N_HEADS} heads")
    
    # ========================================================================
    # OPTIMIZER AND SCHEDULER
    # ========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,  # L2 regularization
        betas=(0.9, 0.95)
    )
    
    scheduler = create_learning_rate_scheduler(
        optimizer,
        num_epochs=config.NUM_EPOCHS,
        warmup_epochs=config.WARMUP_EPOCHS
    )
    
    logger.info(f"Optimizer: AdamW (lr={config.LEARNING_RATE}, weight_decay={config.WEIGHT_DECAY})")
    logger.info(f"Scheduler: Warmup + Cosine Annealing")
    
    # ========================================================================
    # RESUME FROM CHECKPOINT (if specified)
    # ========================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.RESUME_FROM and os.path.exists(config.RESUME_FROM):
        logger.info(f"Resuming from checkpoint: {config.RESUME_FROM}")
        start_epoch, vocab, best_val_loss = load_checkpoint(
            config.RESUME_FROM, model, optimizer, scheduler, config
        )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch+1}/{config.NUM_EPOCHS}")
        logger.info(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            config.DEVICE, config.GRAD_CLIP, logger
        )
        
        # Validate
        val_perplexity = validate(model, val_loader, config.DEVICE, vocab_size, logger)
        val_loss = np.log(val_perplexity)  # Convert perplexity back to loss
        
        # Calculate learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        epoch_time = time.time() - epoch_start_time
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Perplexity: {val_perplexity:.2f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        logger.info(f"  Time: {epoch_time/60:.1f} minutes")
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "MODEL1_BEST.pt")
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                vocab, config, best_checkpoint_path
            )
            logger.info(f"  New best model saved! (Perplexity: {val_perplexity:.2f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                vocab, config, checkpoint_path
            )
        
        # Evaluate (generate samples, calculate BLEU, detect hallucinations)
        if (epoch + 1) % config.EVAL_INTERVAL == 0:
            logger.info("\nRunning comprehensive evaluation...")
            try:
                eval_results = evaluate_model(
                    model, val_loader, vocab, config.DEVICE,
                    source_texts=all_texts[:1000],  # Sample for efficiency
                    num_samples=config.NUM_EVAL_SAMPLES
                )
                logger.info(f"Evaluation Results:")
                logger.info(f"  Perplexity: {eval_results['perplexity']:.2f}")
                logger.info(f"  BLEU Score: {eval_results['bleu_score']:.4f}")
                if eval_results['hallucination_rate'] is not None:
                    logger.info(f"  Hallucination Rate: {eval_results['hallucination_rate']:.2%}")
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
        
        # Print progress
        elapsed_time = time.time() - training_start_time
        avg_time_per_epoch = elapsed_time / (epoch - start_epoch + 1)
        remaining_epochs = config.NUM_EPOCHS - (epoch + 1)
        estimated_remaining = remaining_epochs * avg_time_per_epoch
        
        logger.info(f"\nProgress: {epoch+1}/{config.NUM_EPOCHS} epochs")
        logger.info(f"Elapsed: {elapsed_time/3600:.1f} hours")
        logger.info(f"Estimated remaining: {estimated_remaining/3600:.1f} hours")
    
    # ========================================================================
    # FINAL SAVE
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    
    final_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "MODEL1_FINAL.pt")
    save_checkpoint(
        model, optimizer, scheduler, config.NUM_EPOCHS - 1,
        best_val_loss, vocab, config, final_checkpoint_path
    )
    
    total_time = time.time() - training_start_time
    logger.info(f"Total training time: {total_time/3600:.2f} hours")
    logger.info(f"Best validation perplexity: {np.exp(best_val_loss):.2f}")
    logger.info(f"Final model saved: {final_checkpoint_path}")
    logger.info(f"Best model saved: {os.path.join(config.CHECKPOINT_DIR, 'MODEL1_BEST.pt')}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best Perplexity: {np.exp(best_val_loss):.2f}")
    print(f"Models saved in: {config.CHECKPOINT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise
