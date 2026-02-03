"""
Graph Neural Network model for bifurcation WSS prediction.

Adapted from LidDrivenCavity model with GraphSAGE architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

from config import Config


class WSSPredictor(nn.Module):
    """
    GraphSAGE-based model for predicting wall shear stress on bifurcation surfaces.
    
    Architecture:
    - Input: Node features [N, 14] + Graph structure [2, E]
    - GraphSAGE layers with batch normalization and dropout
    - Output: WSS magnitude [N, 1]
    """
    
    def __init__(
        self,
        in_channels: int = 14,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = 'relu',
    ):
        """
        Args:
            in_channels: Number of input features per node (14 for bifurcation)
            hidden_channels: Hidden dimension size
            out_channels: Number of output values per node (1 for WSS magnitude)
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            activation: Activation function ('relu', 'elu', 'gelu')
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Final prediction layer
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the network.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            batch: Batch assignment vector [N] (optional, for batched graphs)
            
        Returns:
            predictions: WSS magnitude predictions [N, 1]
        """
        # Apply GraphSAGE layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            
            # Apply dropout (except last layer)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final prediction
        x = self.output_layer(x)
        
        return x
    
    def predict(self, x, edge_index):
        """
        Make predictions without gradients (inference mode).
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            
        Returns:
            predictions: WSS magnitude predictions [N, 1]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index)


class WSSPredictorWithUncertainty(WSSPredictor):
    """
    Extended model that predicts both WSS magnitude and uncertainty.
    
    Useful for identifying regions where the model is less confident.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional head for uncertainty estimation
        self.uncertainty_layer = nn.Linear(self.hidden_channels, 1)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass predicting both WSS and uncertainty.
        
        Returns:
            (predictions, uncertainty): Both [N, 1]
        """
        # Apply GraphSAGE layers (reuse parent implementation)
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Prediction heads
        wss_pred = self.output_layer(x)
        uncertainty = F.softplus(self.uncertainty_layer(x))  # Ensure positive
        
        return wss_pred, uncertainty


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(config: Config, with_uncertainty: bool = False):
    """
    Factory function to create model from config.
    
    Args:
        config: Configuration object
        with_uncertainty: Whether to use uncertainty estimation
        
    Returns:
        model: Instantiated model
    """
    ModelClass = WSSPredictorWithUncertainty if with_uncertainty else WSSPredictor
    
    model = ModelClass(
        in_channels=config.num_features,
        hidden_channels=config.hidden_dim,
        out_channels=config.num_targets,
        num_layers=config.num_layers,
        dropout=config.dropout,
        activation=config.activation,
    )
    
    num_params = count_parameters(model)
    print(f"Created {ModelClass.__name__} with {num_params:,} parameters")
    
    return model


if __name__ == '__main__':
    """Test model creation and forward pass."""
    from config import Config
    import torch
    from torch_geometric.data import Data
    
    print("Testing WSSPredictor model...")
    
    # Create config
    config = Config()
    
    # Create model
    model = create_model(config)
    print(f"\nModel architecture:")
    print(model)
    
    # Create dummy data
    num_nodes = 1000
    num_edges = 8000
    
    x = torch.randn(num_nodes, config.num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    data = Data(x=x, edge_index=edge_index)
    
    # Forward pass
    print(f"\nTesting forward pass...")
    print(f"  Input shape: {data.x.shape}")
    print(f"  Edge shape: {data.edge_index.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")
    
    # Test uncertainty model
    print("\n" + "="*60)
    print("Testing WSSPredictorWithUncertainty...")
    
    model_uncertain = create_model(config, with_uncertainty=True)
    
    model_uncertain.eval()
    with torch.no_grad():
        wss_pred, uncertainty = model_uncertain(data.x, data.edge_index)
    
    print(f"  WSS prediction shape: {wss_pred.shape}")
    print(f"  Uncertainty shape: {uncertainty.shape}")
    print(f"  Uncertainty range: [{uncertainty.min():.6f}, {uncertainty.max():.6f}]")
    
    # Test with batch
    print("\n" + "="*60)
    print("Testing with batched graphs...")
    
    batch = torch.cat([
        torch.zeros(500, dtype=torch.long),
        torch.ones(500, dtype=torch.long)
    ])
    
    with torch.no_grad():
        output_batched = model(data.x, data.edge_index, batch)
    
    print(f"  Batch vector shape: {batch.shape}")
    print(f"  Output shape: {output_batched.shape}")
    
    print("\nModel test complete!")
