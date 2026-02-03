"""
Quick test to verify training setup without running full training.
"""

from config import Config
from model import create_model
from dataset import create_dataloaders
from train import Trainer

print("Testing training setup...")
print("="*60)

# Load config
config = Config()

# Override for quick test
config.batch_size = 2
config.num_epochs = 2

# Create dataloaders
print("\n1. Loading datasets...")
train_loader, val_loader, test_loader, norm_stats = create_dataloaders(
    config,
    batch_size=config.batch_size,
    num_workers=0,
)

# Create model
print("\n2. Creating model...")
model = create_model(config)

# Create trainer
print("\n3. Creating trainer...")
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
)

# Test one training step
print("\n4. Testing one training batch...")
batch = next(iter(train_loader))
print(f"   Batch nodes: {batch.num_nodes}")
print(f"   Batch edges: {batch.num_edges}")

# Test validation
print("\n5. Testing validation...")
val_metrics = trainer.validate()
print(f"   Val loss: {val_metrics['val_loss']:.6f}")
print(f"   Val MAE: {val_metrics['val_mae']:.6f}")
print(f"   Val R²: {val_metrics['val_r2']:.4f}")

print("\n[OK] Training setup test complete!")
print("\nTo start full training, run: python train.py")
