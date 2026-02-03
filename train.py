"""
Training script for bifurcation WSS prediction model.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

from config import Config
from model import create_model, count_parameters
from dataset import create_dataloaders


class Trainer:
    """Handles model training and validation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: str = 'auto',
    ):
        """
        Args:
            model: The neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: Device to train on ('cuda', 'cpu', or 'auto')
        """
        self.config = config
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        if config.loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif config.loss_fn == 'mae':
            self.criterion = nn.L1Loss()
        elif config.loss_fn == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")
        
        # Optimizer
        if config.optimizer_name == 'adam':
            self.optimizer = Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer_name == 'adamw':
            self.optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer_name}")
        
        # Learning rate scheduler
        if config.use_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
            )
        else:
            self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        # Create checkpoint directory
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch.x, batch.edge_index, batch.batch)
            
            # Compute loss
            loss = self.criterion(predictions, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        for batch in self.val_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            predictions = self.model(batch.x, batch.edge_index, batch.batch)
            
            # Compute loss
            loss = self.criterion(predictions, batch.y)
            total_loss += loss.item()
            num_batches += 1
            
            # Store for metrics
            all_predictions.append(predictions.cpu())
            all_targets.append(batch.y.cpu())
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute additional metrics
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        mse = torch.mean((all_predictions - all_targets) ** 2).item()
        
        # R² score
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot).item()
        
        metrics = {
            'val_loss': avg_loss,
            'val_mae': mae,
            'val_mse': mse,
            'val_r2': r2,
        }
        
        return metrics
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            print(f"Checkpoint {checkpoint_path} not found")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return True
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Full training loop with early stopping.
        
        Args:
            num_epochs: Number of epochs to train (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Patience: {self.config.patience}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Batch size: {self.config.batch_size}")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['val_loss']
            self.val_losses.append(val_loss)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val MAE:    {val_metrics['val_mae']:.6f}")
            print(f"  Val R²:     {val_metrics['val_r2']:.4f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"  [BEST] New best model! (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.config.patience})")
            
            # Save regular checkpoint every 10 epochs
            if self.current_epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pt')
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {self.current_epoch} epochs")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pt')


def main():
    """Main training function."""
    print("="*60)
    print("Bifurcation WSS Prediction - Model Training")
    print("="*60)
    
    # Load config
    config = Config()
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, norm_stats = create_dataloaders(
        config,
        batch_size=config.batch_size,
        num_workers=0,  # Set to 0 for Windows compatibility
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    # Train model
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Best model saved to: {config.checkpoint_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
