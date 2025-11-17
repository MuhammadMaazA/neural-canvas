#!/usr/bin/env python3
"""
Fancy training progress dashboard
Shows real-time training metrics with color
"""
import time
import os
import re
from datetime import datetime

LOG_FILE = "/cs/student/projects1/2023/muhamaaz/neural-canvas/llm/scripts/training.log"
CHECKPOINT_DIR = "/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom_v2_fixed_wikiart"

# ANSI color codes
BOLD = '\033[1m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RED = '\033[91m'
RESET = '\033[0m'

def clear_screen():
    os.system('clear')

def get_completed_epochs():
    """Extract completed epoch information"""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
        
        epochs = []
        for line in lines:
            if 'Epoch' in line and 'Train:' in line:
                # Parse: Epoch 1 - Train: 2.5075, Val: 2.0588, PPL: 7.84
                match = re.search(r'Epoch (\d+) - Train: ([\d.]+), Val: ([\d.]+), PPL: ([\d.]+)', line)
                if match:
                    epochs.append({
                        'epoch': int(match.group(1)),
                        'train_loss': float(match.group(2)),
                        'val_loss': float(match.group(3)),
                        'perplexity': float(match.group(4))
                    })
        return epochs
    except:
        return []

def get_current_step():
    """Get current training step info"""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
        
        # Find last Training: line
        for line in reversed(lines):
            if 'Training:' in line and '%' in line:
                # Parse: Training:  74%|███| 8837/11894 [20:17<06:58,  7.31it/s, loss=2.0752, lr=0.000274]
                match = re.search(r'(\d+)%.*?(\d+)/(\d+).*?loss=([\d.]+)', line)
                if match:
                    return {
                        'percent': int(match.group(1)),
                        'current_step': int(match.group(2)),
                        'total_steps': int(match.group(3)),
                        'current_loss': float(match.group(4))
                    }
        return None
    except:
        return None

def get_checkpoints():
    """List available checkpoints"""
    try:
        files = os.listdir(CHECKPOINT_DIR)
        checkpoints = [f for f in files if f.endswith('.pt')]
        return sorted(checkpoints)
    except:
        return []

def display_dashboard():
    """Display fancy training dashboard"""
    clear_screen()
    
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}🎨 NEURAL CANVAS - TRAINING DASHBOARD 🎨{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{YELLOW}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print()
    
    # Completed Epochs
    epochs = get_completed_epochs()
    if epochs:
        print(f"{BOLD}{GREEN}📊 COMPLETED EPOCHS:{RESET}")
        print(f"{'-'*70}")
        print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Perplexity':<12}")
        print(f"{'-'*70}")
        for ep in epochs[-5:]:  # Last 5 epochs
            trend = ""
            if len(epochs) > 1:
                prev_idx = epochs.index(ep) - 1
                if prev_idx >= 0:
                    if ep['val_loss'] < epochs[prev_idx]['val_loss']:
                        trend = f"{GREEN}↓{RESET}"
                    else:
                        trend = f"{RED}↑{RESET}"
            
            print(f"{ep['epoch']:<8} {ep['train_loss']:<12.4f} {ep['val_loss']:<12.4f} {ep['perplexity']:<12.2f} {trend}")
        
        # Show improvement
        if len(epochs) >= 2:
            first = epochs[0]
            last = epochs[-1]
            val_improvement = ((first['val_loss'] - last['val_loss']) / first['val_loss']) * 100
            print(f"\n{BOLD}Overall Improvement:{RESET} Val Loss: {first['val_loss']:.4f} → {last['val_loss']:.4f} ({val_improvement:+.1f}%)")
    else:
        print(f"{YELLOW}No completed epochs yet...{RESET}")
    
    print()
    
    # Current Progress
    step_info = get_current_step()
    if step_info:
        print(f"{BOLD}{BLUE}🔄 CURRENT EPOCH PROGRESS:{RESET}")
        print(f"{'-'*70}")
        progress = step_info['percent']
        bar_length = 50
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"Progress: [{GREEN}{bar}{RESET}] {progress}%")
        print(f"Step: {step_info['current_step']:,} / {step_info['total_steps']:,}")
        print(f"Current Loss: {step_info['current_loss']:.4f}")
        
        # Calculate ETA
        remaining_steps = step_info['total_steps'] - step_info['current_step']
        if epochs:
            current_epoch = epochs[-1]['epoch'] + 1
            remaining_epochs = 30 - current_epoch
            print(f"Current Epoch: {current_epoch}/30")
            print(f"Remaining Epochs: {remaining_epochs}")
    else:
        print(f"{YELLOW}Waiting for training to start...{RESET}")
    
    print()
    
    # Checkpoints
    checkpoints = get_checkpoints()
    if checkpoints:
        print(f"{BOLD}{CYAN}💾 SAVED CHECKPOINTS:{RESET}")
        print(f"{'-'*70}")
        for ckpt in checkpoints:
            size = os.path.getsize(os.path.join(CHECKPOINT_DIR, ckpt)) / (1024*1024)
            print(f"  ✓ {ckpt:<30} ({size:.0f} MB)")
    
    print()
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{YELLOW}Refreshing every 10 seconds... (Press Ctrl+C to exit){RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")

def main():
    """Main monitoring loop"""
    try:
        while True:
            display_dashboard()
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n\n{GREEN}✓ Monitoring stopped. Training continues in background!{RESET}\n")

if __name__ == "__main__":
    main()
