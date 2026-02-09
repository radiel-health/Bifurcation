"""
Training script for bifurcation WSS prediction with ensemble.

Implements LOO-CV ensemble training following AVFlow Gen 3 approach.
Trains multiple fold models for ensemble predictions.

Run: python train.py [--folds 3] [--strategy re_interp]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import numpy as np

from config import config
from dataset import get_dataloaders, BifurcationWSSDataset
from Models.model import BifurcationWSSPredictor, get_model_summary


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    """
    Train for one epoch.
    
    Args:
        model: BifurcationWSSPredictor
        loader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: torch device
        grad_clip: Gradient clipping value
        
    Returns:
        avg_loss: Average training loss
    """
    model.train()
    total_loss = 0
    num_graphs = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(batch)
        loss = criterion(y_pred, batch.y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        num_graphs += batch.num_graphs
    
    avg_loss = total_loss / num_graphs
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Validate model.
    
    Args:
        model: BifurcationWSSPredictor
        loader: Validation DataLoader
        criterion: Loss function
        device: torch device
        
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0
    num_graphs = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_pred = model(batch)
            loss = criterion(y_pred, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
    
    avg_loss = total_loss / num_graphs
    return avg_loss


def train_fold(fold_idx, train_loader, val_loader, device, epochs=300):
    """
    Train a single fold model.
    
    Args:
        fold_idx: Fold index (for saving checkpoints)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch device
        epochs: Number of epochs
        
    Returns:
        best_model_state: State dict of best model
        history: Training history dict
    """
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold_idx}")
    print(f"{'='*80}\n")
    
    # Create model
    model = BifurcationWSSPredictor(
        original_node_feat_dim=config.node_feature_dim,
        edge_channels=config.edge_feature_dim,
        aggregated_edge_feat_dim=config.aggregated_edge_feat_dim,
        hidden_gcn_dim=config.hidden_gcn_dim,
        out_channels=config.out_channels,
        num_gcn_layers=config.num_gcn_layers,
        context_dim=config.context_dim,
        unet_hidden=config.unet_hidden,
        unet_depth=config.unet_depth,
        unet_pool_ratio=config.unet_pool_ratio
    ).to(device)
    
    # Loss function
    if config.loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif config.loss_fn == 'huber':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {config.loss_fn}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config.scheduler_mode,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config.grad_clip)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
        
        # Check for improvement
        if val_loss < best_val_loss - config.early_stop_tol:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = config.checkpoint_dir / f"best_state_fold_{fold_idx}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_val_loss,
                'history': history
            }, checkpoint_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nFold {fold_idx} complete | Best Val Loss: {best_val_loss:.6f}")
    
    return best_model_state, history


def main(args):
    """Main training function."""
    print("=" * 80)
    print("BIFURCATION WSS PREDICTION - TRAINING")
    print("=" * 80)
    print()
    
    # Device
    device = config.get_device()
    print(f"Using device: {device}")
    print()
    
    # Create directories
    config.create_directories()
    
    # Load data
    print(f"Loading data with strategy: {args.strategy}")
    train_loader, val_loader, test_loader = get_dataloaders(
        strategy=args.strategy,
        batch_size=config.batch_size
    )
    print()
    
    # Print model architecture once
    sample_model = BifurcationWSSPredictor(
        original_node_feat_dim=config.node_feature_dim,
        edge_channels=config.edge_feature_dim,
        aggregated_edge_feat_dim=config.aggregated_edge_feat_dim,
        hidden_gcn_dim=config.hidden_gcn_dim,
        out_channels=config.out_channels,
        num_gcn_layers=config.num_gcn_layers,
        context_dim=config.context_dim
    )
    get_model_summary(sample_model)
    del sample_model
    print()
    
    # Train ensemble (multiple folds or just one model)
    fold_histories = []
    
    for fold_idx in range(args.folds):
        best_state, history = train_fold(
            fold_idx,
            train_loader,
            val_loader,
            device,
            epochs=config.num_epochs
        )
        fold_histories.append(history)
    
    # Save combined training history
    history_path = config.checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(fold_histories, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nCheckpoints saved to: {config.checkpoint_dir}")
    print(f"Training history saved to: {history_path}")
    print()
    print("Next steps:")
    print("  1. Evaluate: python evaluate.py")
    print("  2. Calibrate: python calibrate.py")
    print("  3. Infer: python infer.py --angle 30 --mesh 750 --re 150")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bifurcation WSS prediction model")
    parser.add_argument('--folds', type=int, default=3, help="Number of ensemble folds")
    parser.add_argument('--strategy', type=str, default='re_interp',
                        choices=['re_interp', 'angle_transfer', 'random'],
                        help="Data split strategy")
    
    args = parser.parse_args()
    main(args)
