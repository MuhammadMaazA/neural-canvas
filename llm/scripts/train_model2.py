"""
Model 2: Llama 2 7B Fine-Tuning
UCL COMP0220 Coursework

LoRA with 4-bit quantization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import os
import json
import time
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_loader import load_wikiart, TextDataset


class Config:
    """Configuration"""
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    USE_4BIT = True
    USE_8BIT = False
    
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    WIKIART_SAMPLES = 20000
    MAX_LEN = 512
    
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.03
    MAX_GRAD_NORM = 0.3
    
    TRAIN_SPLIT = 0.95
    VAL_SPLIT = 0.05
    
    CHECKPOINT_DIR = "/cs/student/projects1/2023/muhamaaz/checkpoints/model2"
    LOG_DIR = "/cs/student/projects1/2023/muhamaaz/logs"
    SAVE_STEPS = 500
    EVAL_STEPS = 250
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16 = True
    BF16 = False


def setup_logging(log_dir: str = "logs"):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"llama2_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_llama_model(config):
    """Load Llama 2 7B with quantization and LoRA"""
    
    logger = logging.getLogger(__name__)
    
    # Quantization config for QLoRA
    if config.USE_4BIT:
        logger.info("Loading model with 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.USE_8BIT:
        logger.info("Loading model with 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        logger.info("Loading model in full precision...")
        bnb_config = None
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.FP16 else torch.float32,
    )
    
    # Prepare for k-bit training
    if config.USE_4BIT or config.USE_8BIT:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"Total parameters: {all_params:,} ({all_params/1e6:.1f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    logger.info(f"Trainable ratio: {100 * trainable_params / all_params:.2f}%")
    
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, tokenizer, config, filepath: str):
    """Save LoRA checkpoint"""
    # Save LoRA adapters only (much smaller than full model)
    model.save_pretrained(filepath)
    
    # Save training state
    state = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, os.path.join(filepath, "training_state.pt"))
    
    # Save config
    config_dict = {
        'model_name': config.MODEL_NAME,
        'lora_r': config.LORA_R,
        'lora_alpha': config.LORA_ALPHA,
        'max_len': config.MAX_LEN,
    }
    with open(os.path.join(filepath, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logging.info(f"Checkpoint saved: {filepath}")


def main():
    """Main fine-tuning function"""
    config = Config()
    logger = setup_logging(config.LOG_DIR)
    
    print("=" * 60)
    print("Model 2: Llama 2 7B Fine-Tuning")
    print("UCL COMP0220 Coursework")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    logger.info("Starting fine-tuning...")
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    logger.info(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info("Loading WikiArt dataset...")
    start_time = time.time()
    
    art_texts = load_wikiart(max_samples=config.WIKIART_SAMPLES)
    logger.info(f"Loaded {len(art_texts):,} samples in {time.time() - start_time:.1f}s")
    
    # Split dataset
    split_idx = int(len(art_texts) * config.TRAIN_SPLIT)
    train_texts = art_texts[:split_idx]
    val_texts = art_texts[split_idx:]
    
    logger.info(f"Train: {len(train_texts):,} samples")
    logger.info(f"Val: {len(val_texts):,} samples")
    
    # Create dataloaders
    train_dataset = TextDataset(train_texts, tokenizer, max_len=config.MAX_LEN)
    val_dataset = TextDataset(val_texts, tokenizer, max_len=config.MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    # Load model with LoRA
    logger.info("Loading Llama 2 7B model...")
    model = load_llama_model(config)
    
    # Optimizer (only optimize LoRA parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Training steps: {num_training_steps:,} (warmup: {num_warmup_steps:,})")
    
    # Training loop
    best_val_loss = float('inf')
    training_start = time.time()
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        
        logger.info(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Training
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            
            # Forward pass
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss / config.GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            
            # Update weights every GRADIENT_ACCUMULATION_STEPS
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                pbar.set_postfix({
                    'loss': f"{loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                })
                
                # Save checkpoint
                if global_step % config.SAVE_STEPS == 0:
                    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_step_{global_step}")
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, 
                                  loss.item(), tokenizer, config, checkpoint_path)
                
                # Validation
                if global_step % config.EVAL_STEPS == 0:
                    model.eval()
                    val_loss = 0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for val_inputs, val_targets in val_loader:
                            val_inputs = val_inputs.to(config.DEVICE)
                            val_targets = val_targets.to(config.DEVICE)
                            
                            val_outputs = model(input_ids=val_inputs, labels=val_targets)
                            val_loss += val_outputs.loss.item()
                            val_batches += 1
                    
                    val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
                    val_perplexity = np.exp(val_loss)
                    
                    logger.info(f"Step {global_step} - Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(config.CHECKPOINT_DIR, "best_model")
                        save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                      val_loss, tokenizer, config, best_path)
                        logger.info(f"New best model saved: Perplexity {val_perplexity:.2f}")
                    
                    model.train()
        
        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Time: {epoch_time/60:.1f} min")
    
    # Final save
    final_path = os.path.join(config.CHECKPOINT_DIR, "final_model")
    save_checkpoint(model, optimizer, scheduler, config.NUM_EPOCHS-1, global_step,
                  best_val_loss, tokenizer, config, final_path)
    
    total_time = time.time() - training_start
    logger.info(f"Fine-tuning complete: {total_time/3600:.2f} hours")
    if best_val_loss < float('inf'):
        logger.info(f"Best perplexity: {np.exp(best_val_loss):.2f}")
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Best model: {os.path.join(config.CHECKPOINT_DIR, 'best_model')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
