"""
Training script for wall shear stress prediction model.

Usage:
    python train.py

This script:
1. Loads preprocessed PyG graphs from ProcessedData/
2. Creates train/val/test splits
3. Trains WSSPredictor model with MSE loss on log-normalized WSS
4. Saves checkpoints and logs metrics
5. Evaluates on test set

Key features:
- Stratified splits by aspect ratio
- Log1p normalization for WSS (handles 7 orders of magnitude)
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- Gradient clipping for stability
"""

import sys
from pathlib import Path
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

# Import from local modules
from config import config
from dataset import WSSDataset, get_dataloaders
from model import WSSPredictor, count_parameters, get_model_summary
import torchbnn as bnn

kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.025

def create_model(device):
    """
    Create WSSPredictor model with config settings.
    
    Args:
        device: torch device
        
    Returns:
        model: WSSPredictor instance on device
    """
    model = WSSPredictor(
        node_feature_dim=config.node_feature_dim,
        flow_param_dim=config.flow_param_dim,
        hidden_dim=config.hidden_dim,
        context_dim=config.context_dim,
        output_dim=config.target_dim,
        num_geom_layers=config.num_geom_layers,
        num_task_layers=config.num_task_layers,
        task_hidden_dim=config.task_hidden_dim,
        dropout=config.dropout_rate
    )
    
    return model.to(device)

def compute_loss(y_pred, y_true, reduction='mean'):
    """
    Compute MSE loss on log-normalized WSS.
    
    The WSS values span 7 orders of magnitude (1e-14 to 1e-5), so we use
    log1p normalization before computing MSE.
    
    Args:
        y_pred: [num_nodes, 2] predicted WSS components (already normalized)
        y_true: [num_nodes, 2] true WSS components (already normalized)
        reduction: 'mean' or 'sum'
        
    Returns:
        loss: scalar tensor
    """
    # Both y_pred and y_true are already log1p normalized by dataset
    # Just compute MSE
    loss = nn.functional.mse_loss(y_pred, y_true, reduction=reduction)
    
    return loss


def train_epoch(model, loader, optimizer, device, grad_clip=None):
    """
    Train for one epoch.
    
    Args:
        model: WSSPredictor instance
        loader: DataLoader for training data
        optimizer: Optimizer instance
        device: torch device
        grad_clip: Gradient clipping value (None to disable)
        
    Returns:
        avg_loss: Average loss over epoch
    """
    model.train()
    total_loss = 0
    num_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Forward pass
        y_pred = model(batch)
        
        # Compute loss
        loss = (1-kl_weight)*compute_loss(y_pred, batch.y) + kl_weight*kl_loss(model)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs
    
    avg_loss = total_loss / num_samples
    return avg_loss

@torch.no_grad()
def validate_epoch(model, loader, device):
    """
    Validate on validation set.
    
    Args:
        model: WSSPredictor instance
        loader: DataLoader for validation data
        device: torch device
        
    Returns:
        avg_loss: Average loss over validation set
    """
    model.eval()
    total_loss = 0
    num_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Forward pass
        y_pred = model(batch)
        
        # Compute loss
        loss = compute_loss(y_pred, batch.y)
        
        # Accumulate
        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs
    
    avg_loss = total_loss / num_samples
    return avg_loss


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """
    Save model checkpoint.
    
    Args:
        model: WSSPredictor instance
        optimizer: Optimizer instance
        epoch: Current epoch number
        val_loss: Validation loss
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    """
    Load model checkpoint.
    
    Args:
        model: WSSPredictor instance
        optimizer: Optimizer instance
        path: Path to checkpoint
        device: torch device
        
    Returns:
        epoch: Epoch number
        val_loss: Validation loss
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    
    return epoch, val_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_dir,
    early_stop_patience=20,
    grad_clip=1.0,
    save_best_only=True,
):
    """
    Full training loop with validation, checkpointing, and early stopping.
    
    Args:
        model: WSSPredictor instance
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: torch device
        num_epochs: Maximum number of epochs
        checkpoint_dir: Directory to save checkpoints
        early_stop_patience: Epochs without improvement before stopping
        grad_clip: Gradient clipping value
        save_best_only: Only save when validation improves
        
    Returns:
        training_history: Dict with train/val losses per epoch
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
    }
    
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_clip)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save best model
            if save_best_only:
                best_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(model, optimizer, epoch, val_loss, best_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
        
        # Save periodic checkpoint
        if epoch % config.save_every_n_epochs == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
        
        # Early stopping
        if early_stop_patience is not None and epochs_no_improve >= early_stop_patience:
            print(f"\n⚠ Early stopping triggered (no improvement for {early_stop_patience} epochs)")
            break
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Final val loss: {val_loss:.6f}")
    print("=" * 80 + "\n")
    
    return history


def main():
    """Main training script."""
    
    print("\n" + "=" * 80)
    print("WALL SHEAR STRESS PREDICTION - TRAINING")
    print("=" * 80)
    print()
    
    # Set device
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"✓ Using CPU")
    print()
    
    # Load dataset
    print("Loading dataset...")
    
    batch_size = config.batch_size
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,
    )
    
    print(f"Dataset loaded")
    print(f"  Train: {len(train_loader.dataset)} graphs")
    print(f"  Val: {len(val_loader.dataset)} graphs")
    print(f"  Test: {len(test_loader.dataset)} graphs")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(device)
    
    # Print summary
    get_model_summary(model)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr
        )
    else:
        scheduler = None
    
    # Create checkpoint directory
    config.checkpoint_dir.mkdir(exist_ok=True)
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config.num_epochs,
        checkpoint_dir=config.checkpoint_dir,
        early_stop_patience=config.early_stop_patience if config.use_early_stopping else None,
        grad_clip=config.grad_clip_value if config.use_grad_clip else None,
        save_best_only=config.save_best_only,
    )
    
    # Save training history
    history_path = config.checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")
    
    # Load best model for final evaluation
    best_path = config.checkpoint_dir / "best_model.pt"
    if best_path.exists():
        print("\nLoading best model for final evaluation...")
        load_checkpoint(model, optimizer, best_path, device)
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_loss = validate_epoch(model, test_loader, device)
    print(f"Test Loss: {test_loss:.6f}")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
