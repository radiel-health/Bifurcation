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
import time
import json
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim

# Import from local modules
from config import config
from dataset import get_dataloaders
from model import WSSPredictor
import torchbnn as bnn

kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.025

def create_model(device):
    """
    Create WSSPredictor model with config settings.
    """
    model = WSSPredictor(
        node_feature_dim=config.node_feature_dim, # This is now 7
        # Note: We deleted flow_param_dim and context_dim from here!
        hidden_dim=config.hidden_dim,
        output_dim=config.target_dim,
        num_geom_layers=config.num_geom_layers,
        num_task_layers=config.num_task_layers,
        task_hidden_dim=config.task_hidden_dim,
        dropout=config.dropout_rate,
        output_range=config.output_range
    )
    
    return model.to(device)

def compute_loss(y_pred, y_true, reduction='mean'):
    """
    Compute loss on log-normalized WSS.
    Automatically handles BNN Quantiles (Pinball Loss) or Mean (MSE Loss).
    """
    # If TaskHead output_range=True, shape is [2, num_nodes, 3] for [0.025, 0.975] quantiles
    if y_pred.dim() == 3 and y_pred.shape[0] == 2:
        quantiles = [0.025, 0.975]
        loss = 0
        
        for i, q in enumerate(quantiles):
            errors = y_true - y_pred[i]
            # Pinball (Quantile) Loss formula
            loss_q = torch.max((q - 1) * errors, q * errors)
            
            if reduction == 'mean':
                loss += loss_q.mean()
            else:
                loss += loss_q.sum()
                
        return loss
        
    # Default to MSE if outputting mean (output_range=False)
    else:
        return nn.functional.mse_loss(y_pred, y_true, reduction=reduction)


def train_epoch(model, loader, optimizer, device, grad_clip=None, accumulation_steps=16):
    model.train()
    total_loss = 0
    num_samples = 0

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        # Bring back your heartbeat message!
        print(f"Batch {i}: {batch.num_graphs} graphs, {batch.x.shape[0]} total nodes, {batch.edge_index.shape[1]} total edges")
        
        batch = batch.to(device)

        y_pred = model(batch)
        step_loss = (1 - kl_weight) * compute_loss(y_pred, batch.y) + kl_weight * kl_loss(model)
        
        # Scale the loss
        scaled_loss = step_loss / accumulation_steps
        scaled_loss.backward()

        # Update every N steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            
            # Optional: print a little checkmark to show the update happened
            print(f"  --> Optimizer step completed (Effective Batch Size: {accumulation_steps if (i+1)%accumulation_steps==0 else (i+1)%accumulation_steps})")

        total_loss += step_loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / num_samples

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


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_loss, path):
    """
    Save model checkpoint including scheduler and best loss state.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    """
    Load model checkpoint, safely handling older formats.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    return epoch, val_loss, best_val_loss


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs,
    checkpoint_dir, early_stop_patience=20, grad_clip=1.0, save_best_only=True,
    start_epoch=1, best_val_loss=float('inf'), history=None
):
    epochs_no_improve = 0
    
    # Initialize history if starting fresh
    if history is None:
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }
    
    print("\n" + "=" * 80)
    print(f"TRAINING (Starting from Epoch {start_epoch})")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Target Epochs: {num_epochs}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    # Loop starts from the resumed epoch
    for epoch in range(start_epoch, num_epochs + 1):
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
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_loss, best_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
        
        # Save periodic checkpoint
        if epoch % config.save_every_n_epochs == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_loss, ckpt_path)
            
        # ALWAYS save the latest checkpoint for seamless resuming
        latest_path = checkpoint_dir / "latest_checkpoint.pt"
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_loss, latest_path)
        
        # Save training history to JSON immediately so it stays synced
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
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


def ensure_processed_data_exists():
    """Check if processed data exists, run preprocessing if not."""
    processed_dir = config.processed_data_dir
    
    # Check if directory exists and has .pt files
    pt_files = list(processed_dir.rglob("*.pt")) if processed_dir.exists() else []
    
    if not pt_files:
        print(f"ProcessedData/3D is empty. Running pre_process.py...")
        
        # Run pre_process.py
        result = subprocess.run(
            [sys.executable, "pre_process.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"pre_process.py failed with error:\n{result.stderr}")
            sys.exit(1)
        
        print(result.stdout)
        
        # Check again for .pt files
        pt_files = list(processed_dir.rglob("*.pt")) if processed_dir.exists() else []
    
    # Assert that processed data exists
    assert len(pt_files) > 0, (
        "ProcessedData/3D was empty and tried running pre_process.py and it was still empty"
    )
    
    print(f"✓ Found {len(pt_files)} processed data files")


def main():
    """Main training script."""
    ensure_processed_data_exists()
    
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
    
    print("Loading dataset...")
    batch_size = config.batch_size
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,
    )
    
    print(f"Dataset loaded")
    print(f"  Train: {len(train_loader.dataset)} graphs")
    print(f"  Val: {len(val_loader.dataset)} graphs")
    print(f"  Test: {len(test_loader.dataset)} graphs")
    print()
    
    print("Creating model...")
    model = create_model(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
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
    
    config.checkpoint_dir.mkdir(exist_ok=True)
    
    # ==========================================
    # NEW: Resume Logic
    # ==========================================
    latest_ckpt_path = config.checkpoint_dir / "latest_checkpoint.pt"
    history_path = config.checkpoint_dir / "training_history.json"
    
    start_epoch = 1
    best_val_loss = float('inf')
    history = None
    
    if latest_ckpt_path.exists():
        print(f"\n[INFO] Found existing checkpoint. Resuming from {latest_ckpt_path}...")
        loaded_epoch, _, loaded_best_val = load_checkpoint(model, optimizer, scheduler, latest_ckpt_path, device)
        start_epoch = loaded_epoch + 1
        best_val_loss = loaded_best_val
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
                
        print(f"[INFO] Successfully restored weights. Resuming at Epoch {start_epoch} (Best Val Loss: {best_val_loss:.6f})")
    
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
        start_epoch=start_epoch,           # Pass in the resume states
        best_val_loss=best_val_loss,       # Pass in the resume states
        history=history                    # Pass in the history
    )
    
    # Load best model for final evaluation
    best_path = config.checkpoint_dir / "best_model.pt"
    if best_path.exists():
        print("\nLoading best model for final evaluation...")
        load_checkpoint(model, optimizer, scheduler, best_path, device)
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_loss = validate_epoch(model, test_loader, device)
    print(f"Test Loss: {test_loss:.6f}")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
