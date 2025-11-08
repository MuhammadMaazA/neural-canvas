"""
Model 1: Transformer from Scratch
UCL COMP0220 Coursework - AI Art Expert & Creator

Modern architecture: RoPE, RMSNorm, GQA, SwiGLU
Training on AI Literacy + Art Domain datasets
Theme: AI literacy for art education
"""

# SET CACHE DIRECTORIES FIRST (before any imports)
import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import time
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modern_transformer import ModernTransformer
from utils.dataset_loader import load_all_datasets, TextDataset
from utils.model_evaluator import evaluate_model, calculate_perplexity


class Config:
    """Training configuration - AI Art Expert & Creator"""
    # Model Architecture
    DIM = 1024
    N_LAYERS = 12
    N_HEADS = 16
    N_KV_HEADS = 4
    MAX_SEQ_LEN = 2048
    DROPOUT = 0.1
    
    # Datasets (Coursework Aligned: AI Literacy + Art Domain)
    # Theme: AI Art Expert & Creator (matches CNN/NST project)
    # NEW: Lightweight datasets for 50GB quota
    ELI5_SAMPLES = 40000              # AI literacy Q&A explanations
    CONVERSATIONAL_SAMPLES = 30000    # Podcast/dialogue style AI discussions
    ART_TEXT_SAMPLES = 20000          # Art descriptions (text-only, no images)
    AI_QA_SAMPLES = 30000             # AI/ML technical Wikipedia
    
    # Legacy (deprecated - too large for quota)
    WIKIART_SAMPLES = 0               # DISABLED - 30GB+ with images
    OPENWEBTEXT_SAMPLES = 0           # DISABLED - 100GB+
    C4_SAMPLES = 0                    # DISABLED - 100GB+
    
    # Tokenizer & Training
    TOKENIZER_NAME = "gpt2"
    MAX_LEN = 512
    
    NUM_EPOCHS = 30
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 2000
    
    TRAIN_SPLIT = 0.95
    VAL_SPLIT = 0.05
    
    CHECKPOINT_DIR = "/cs/student/projects1/2023/muhamaaz/checkpoints/model1"
    LOG_DIR = "/cs/student/projects1/2023/muhamaaz/logs"
    CHECKPOINT_INTERVAL = 2
    EVAL_INTERVAL = 1
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4


def setup_logging(log_dir: str = "logs"):
    """Setup logging"""
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


def save_checkpoint(model, optimizer, scheduler, epoch, loss, tokenizer, config, filepath: str):
    """Save checkpoint for modern transformer"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'tokenizer_name': config.TOKENIZER_NAME,
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


def create_lr_scheduler(optimizer, num_training_steps: int, warmup_steps: int = 1000):
    """Learning rate scheduler with warmup and cosine decay"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    scheduler.step()
    
    return avg_loss


def validate(model, dataloader, device, vocab_size: int):
    """Validate model"""
    model.eval()
    perplexity = calculate_perplexity(model, dataloader, device, vocab_size)
    return perplexity


def main():
    """Main training function"""
    config = Config()
    logger = setup_logging(config.LOG_DIR)
    
    print("=" * 60)
    print("Model 1: Transformer from Scratch")
    print("UCL COMP0220 Coursework - AI Art Expert & Creator")
    print("Theme: AI Literacy + Art Domain")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Datasets: SQuAD Q&A ({config.ELI5_SAMPLES:,}) + BookCorpus ({config.AI_QA_SAMPLES:,})")
    print("=" * 60)
    
    logger.info("Starting training...")
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    logger.info(f"Loading tokenizer: {config.TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocab size: {vocab_size:,}")
    
    logger.info("Loading datasets...")
    start_time = time.time()
    
    all_texts = load_all_datasets(
        eli5_samples=config.ELI5_SAMPLES,
        conversational_samples=config.CONVERSATIONAL_SAMPLES,
        art_text_samples=config.ART_TEXT_SAMPLES,
        ai_qa_samples=config.AI_QA_SAMPLES
    )
    logger.info(f"Loaded {len(all_texts):,} samples in {time.time() - start_time:.1f}s")
    
    # Split datasets
    split_idx = int(len(all_texts) * config.TRAIN_SPLIT)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    logger.info(f"Train: {len(train_texts):,} samples")
    logger.info(f"Val: {len(val_texts):,} samples")
    
    # Create dataloaders
    train_dataset = TextDataset(train_texts, tokenizer, max_len=config.MAX_LEN)
    val_dataset = TextDataset(val_texts, tokenizer, max_len=config.MAX_LEN)
    
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
    
    logger.info("Initializing model...")
    model = ModernTransformer(
        vocab_size=vocab_size,
        dim=config.DIM,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        n_kv_heads=config.N_KV_HEADS,
        max_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params/1e6:.1f}M")
    
    # Optimizer with modern settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Scheduler with step-based warmup
    num_training_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = create_lr_scheduler(optimizer, num_training_steps, config.WARMUP_STEPS)
    
    # Training loop
    best_val_loss = float('inf')
    training_start = time.time()
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        
        logger.info(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Train epoch
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            
            # Forward pass
            logits, loss = model(inputs, targets)
            
            if loss is None:
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
            
            if batch_idx % 100 == 0:
                logger.info(f"Step {global_step}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Validate
        if (epoch + 1) % config.EVAL_INTERVAL == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc="Validation"):
                    inputs = inputs.to(config.DEVICE)
                    targets = targets.to(config.DEVICE)
                    
                    logits, loss = model(inputs, targets)
                    if loss is not None:
                        val_loss += loss.item()
                        val_batches += 1
            
            val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            val_perplexity = np.exp(val_loss)
        else:
            val_loss = float('inf')
            val_perplexity = float('inf')
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        if val_perplexity < float('inf'):
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Perplexity: {val_perplexity:.2f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        logger.info(f"  Time: {epoch_time/60:.1f} min")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.CHECKPOINT_DIR, "model1_best.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, tokenizer, config, best_path)
            logger.info(f"New best model: Perplexity {val_perplexity:.2f}")
        
        # Save periodic
        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, tokenizer, config, checkpoint_path)
    
    final_path = os.path.join(config.CHECKPOINT_DIR, "model1_final.pt")
    save_checkpoint(model, optimizer, scheduler, config.NUM_EPOCHS-1, best_val_loss, tokenizer, config, final_path)
    
    total_time = time.time() - training_start
    logger.info(f"Training complete: {total_time/3600:.2f} hours")
    if best_val_loss < float('inf'):
        logger.info(f"Best perplexity: {np.exp(best_val_loss):.2f}")
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Best model: model1_best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
