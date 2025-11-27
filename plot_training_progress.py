#!/usr/bin/env python3
"""
Plot training and validation loss from logs
"""
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

LOG_FILE = "/cs/student/projects1/2023/muhamaaz/neural-canvas/llm/scripts/training.log"
OUTPUT_FILE = "/cs/student/projects1/2023/muhamaaz/neural-canvas/training_progress.png"

def parse_training_log():
    """Extract epoch metrics from training log"""
    epochs = []
    train_losses = []
    val_losses = []
    perplexities = []
    
    with open(LOG_FILE, 'r') as f:
        for line in f:
            # Match: Epoch 1 - Train: 2.5075, Val: 2.0588, PPL: 7.84
            match = re.search(r'Epoch (\d+) - Train: ([\d.]+), Val: ([\d.]+), PPL: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                ppl = float(match.group(4))
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                perplexities.append(ppl)
    
    return epochs, train_losses, val_losses, perplexities

def create_plot():
    """Create and save training progress plots"""
    epochs, train_losses, val_losses, perplexities = parse_training_log()
    
    if not epochs:
        print("❌ No epoch data found in log file!")
        return
    
    print(f"📊 Found data for {len(epochs)} epochs: {epochs}")
    print(f"   Train Loss: {train_losses}")
    print(f"   Val Loss: {val_losses}")
    print(f"   Perplexity: {perplexities}")
    print()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Add value labels on points
    for i, (e, tl, vl) in enumerate(zip(epochs, train_losses, val_losses)):
        if i % 1 == 0:  # Label every point
            ax1.annotate(f'{tl:.2f}', (e, tl), textcoords="offset points", 
                        xytext=(0,8), ha='center', fontsize=9, color='blue')
            ax1.annotate(f'{vl:.2f}', (e, vl), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=9, color='red')
    
    # Plot 2: Perplexity
    ax2.plot(epochs, perplexities, 'g-^', label='Perplexity', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax2.set_title('Model Perplexity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    # Add value labels
    for i, (e, p) in enumerate(zip(epochs, perplexities)):
        if i % 1 == 0:
            ax2.annotate(f'{p:.2f}', (e, p), textcoords="offset points", 
                        xytext=(0,8), ha='center', fontsize=9, color='green')
    
    # Overall title
    fig.suptitle('🎨 Neural Canvas Training Progress', fontsize=16, fontweight='bold')
    
    # Add summary stats
    if len(epochs) >= 2:
        train_improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100
        val_improvement = ((val_losses[0] - val_losses[-1]) / val_losses[0]) * 100
        ppl_improvement = ((perplexities[0] - perplexities[-1]) / perplexities[0]) * 100
        
        stats_text = f'Epochs: {epochs[-1]}/30 | Train ↓{train_improvement:.1f}% | Val ↓{val_improvement:.1f}% | PPL ↓{ppl_improvement:.1f}%'
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {OUTPUT_FILE}")
    print(f"\n📈 TRAINING SUMMARY:")
    print(f"   Epochs completed: {epochs[-1]}/30")
    print(f"   Train Loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f} (↓{train_improvement:.1f}%)")
    print(f"   Val Loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f} (↓{val_improvement:.1f}%)")
    print(f"   Perplexity: {perplexities[0]:.2f} → {perplexities[-1]:.2f} (↓{ppl_improvement:.1f}%)")

if __name__ == "__main__":
    create_plot()
