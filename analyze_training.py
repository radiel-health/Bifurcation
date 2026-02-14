"""
Analyze training results and visualize learning curves.
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from config import config

def load_training_history():
    """Load training history from JSON."""
    history_path = config.checkpoint_dir / "training_history.json"
    if not history_path.exists():
        print(f"Training history not found at {history_path}")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history

def plot_learning_curves(history):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss (log-normalized WSS)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Learning rate schedule
    ax2.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = config.checkpoint_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved learning curves to {save_path}")
    
    plt.show()

def print_training_summary(history):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    num_epochs = len(history['train_loss'])
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    best_val_loss = min(history['val_loss'])
    best_epoch = history['val_loss'].index(best_val_loss) + 1
    
    print(f"Total epochs trained: {num_epochs}")
    print(f"\nFinal losses:")
    print(f"  Train: {final_train_loss:.6f}")
    print(f"  Val:   {final_val_loss:.6f}")
    print(f"\nBest validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    
    # Check for overfitting
    gap = final_train_loss - final_val_loss
    if gap > 0.1:
        print(f"\n⚠️  Possible overfitting detected (train-val gap: {gap:.4f})")
    elif gap < -0.05:
        print(f"\n⚠️  Unusual: Val loss lower than train loss (gap: {gap:.4f})")
    else:
        print(f"\n✓ Healthy train-val gap: {gap:.4f}")
    
    # Check for convergence
    last_10_val = history['val_loss'][-10:]
    val_std = (max(last_10_val) - min(last_10_val))
    if val_std < 0.01:
        print(f"✓ Model converged (val loss stable: ±{val_std:.4f} over last 10 epochs)")
    else:
        print(f"⚠️  Model still improving (val loss varying: ±{val_std:.4f})")
    
    print("=" * 80 + "\n")

def check_checkpoints():
    """Check what checkpoints were saved."""
    print("Saved checkpoints:")
    
    best_path = config.checkpoint_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
        print(f"  ✓ best_model.pt (epoch {ckpt['epoch']}, val_loss: {ckpt['val_loss']:.6f})")
    else:
        print(f"  ❌ best_model.pt not found")
    
    # Check periodic checkpoints
    periodic = list(config.checkpoint_dir.glob("checkpoint_epoch*.pt"))
    if periodic:
        print(f"  ✓ {len(periodic)} periodic checkpoint(s)")
        for p in sorted(periodic):
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            print(f"    - {p.name} (val_loss: {ckpt['val_loss']:.6f})")
    
    print()

def main():
    """Analyze training results."""
    print("\n" + "=" * 80)
    print("TRAINING ANALYSIS")
    print("=" * 80 + "\n")
    
    # Load history
    history = load_training_history()
    if history is None:
        return
    
    # Print summary
    print_training_summary(history)
    
    # Check checkpoints
    check_checkpoints()
    
    # Plot curves
    print("Generating plots...")
    plot_learning_curves(history)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()