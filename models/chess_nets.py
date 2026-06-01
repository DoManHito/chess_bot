"""
Chess Neural Network Architecture for Move Classification.

This module implements a deep neural network architecture designed for chess move
classification and evaluation. The network consists of three main components:

1. **ChessCoreNet**: A convolutional backbone that processes 25-channel board
   representations (12 channels for white pieces, 12 for black pieces, 1 for turn indicator).
   Uses residual blocks to maintain gradient flow through deep networks.

2. **MoveClassifierNet**: A multi-task network that takes the core network's output
   and produces two outputs simultaneously:
   - Classification logits: 6-class output for move quality (Best, Excellent, Good,
     Inaccuracy, Mistake, Blunder)
   - Value prediction: Scalar evaluation of the position after the move

Input Tensor Shape: (batch_size, 25, 8, 8)
- Layers 0-5: White pieces (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
- Layers 6-11: Black pieces (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
- Layers 12-17: White pieces after move
- Layers 18-23: Black pieces after move
- Layer 24: Turn indicator (1 = White to move, 0 = Black to move)

Output Shapes:
- class_logits: (batch_size, 6) - logits for 6 move quality classes
- value: (batch_size,) - normalized evaluation in range [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessResidualBlock(nn.Module):
    """
    Residual block for chess neural network.

    Implements a residual connection that helps train deeper networks by allowing
    gradients to flow directly through the network. Each block consists of:
    - Conv2d + BatchNorm + ReLU
    - Conv2d + BatchNorm
    - Residual addition with final ReLU

    Args:
        channels: Number of input/output channels (default: 128)
    """
    def __init__(self, channels=128):
        super().__init__()
        # First convolutional layer with batch normalization and ReLU activation
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x: Input tensor of shape (batch_size, channels, 8, 8)

        Returns:
            Output tensor after residual connection and ReLU activation
        """
        residual = x  # Save input for residual connection
        # First conv block: conv -> batch norm -> relu
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv block: conv -> batch norm (no relu yet)
        out = self.bn2(self.conv2(out))
        # Add residual connection
        out += residual
        # Final ReLU activation
        return F.relu(out)


class ChessCoreNet(nn.Module):
    """
    Core convolutional network for chess board representation processing.

    This network serves as the feature extractor for chess positions. It takes
    a 25-channel board representation and processes it through:
    1. Initial convolutional layer to project 25 channels to hidden channels
    2. Multiple residual blocks for deep feature extraction
    3. Maintains spatial dimensions (8x8) throughout for position-aware features

    Args:
        in_channels: Number of input channels (default: 25 - standard chess board)
        num_blocks: Number of residual blocks (default: 4)
        hidden_channels: Hidden channel dimension (default: 128)

    Input Shape: (batch_size, 25, 8, 8)
    Output Shape: (batch_size, hidden_channels, 8, 8)
    """
    def __init__(self, in_channels=25, num_blocks=4, hidden_channels=128):
        super().__init__()
        # Initial convolution: projects input channels to hidden channels
        # Uses kernel_size=3 with padding=1 to maintain spatial dimensions
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)
        # Stack of residual blocks for deep feature extraction
        self.blocks = nn.ModuleList([ChessResidualBlock(hidden_channels) for _ in range(num_blocks)])

    def forward(self, x):
        """
        Forward pass through the core network.

        Args:
            x: Input tensor of shape (batch_size, 25, 8, 8)

        Returns:
            Feature tensor of shape (batch_size, hidden_channels, 8, 8)
        """
        # Initial convolution with batch norm and ReLU
        x = F.relu(self.bn_init(self.conv_init(x)))
        # Pass through all residual blocks
        for block in self.blocks:
            x = block(x)
        return x


class MoveClassifierNet(nn.Module):
    """
    Multi-task neural network for move classification and value prediction.

    This network takes the output from ChessCoreNet and branches into two parallel
    prediction heads:
    1. **Classification Head**: Predicts move quality class (Best, Excellent, Good,
       Inaccuracy, Mistake, Blunder) using a fully connected network with dropout.
    2. **Value Head**: Predicts the evaluation of the position after the move,
       normalized to [-1, 1] range using tanh activation.

    The network shares features from the core network between both heads,
    allowing the model to learn correlated patterns between move quality and
    position evaluation.

    Args:
        core_net: Instance of ChessCoreNet for feature extraction
        num_classes: Number of classification categories (default: 6)
        hidden_channels: Hidden dimension for core network (default: 128)

    Input Shape: (batch_size, 25, 8, 8)
    Output Shapes:
        - class_logits: (batch_size, num_classes) - raw logits for classification
        - value: (batch_size,) - normalized evaluation score
    """
    def __init__(self, core_net, num_classes=6, hidden_channels=128):
        super().__init__()
        self.core = core_net  # Feature extractor from ChessCoreNet
        # Reduce spatial dimensions from 8x8 to 1x1 while compressing channels
        self.conv_reduce = nn.Conv2d(hidden_channels, 16, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(16)

        # Classification head: 16*8*8=1024 -> 256 -> num_classes
        # Uses dropout (p=0.4) to prevent overfitting
        self.fc_class1 = nn.Linear(16 * 8 * 8, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.fc_class2 = nn.Linear(256, num_classes)

        # Value head: 16*8*8=1024 -> 32 -> 1
        # Uses tanh to normalize output to [-1, 1] range
        self.fc_value1 = nn.Linear(16 * 8 * 8, 32)
        self.fc_value2 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Forward pass producing classification logits and value prediction.

        Args:
            x: Input tensor of shape (batch_size, 25, 8, 8)

        Returns:
            Tuple of (class_logits, value) where:
            - class_logits: (batch_size, num_classes) raw classification logits
            - value: (batch_size,) normalized evaluation in [-1, 1]
        """
        # Extract features using the core network
        features = self.core(x)  # (batch_size, hidden_channels, 8, 8)
        # Reduce spatial dimensions and apply ReLU
        x_shared = F.relu(self.bn_reduce(self.conv_reduce(features)))  # (batch_size, 16, 8, 8)
        # Flatten spatial dimensions to (batch_size, 1024)
        flattened = x_shared.view(x_shared.size(0), -1)

        # Classification branch
        xc = F.relu(self.fc_class1(flattened))  # (batch_size, 256)
        xc = self.dropout(xc)  # Apply dropout for regularization
        class_logits = self.fc_class2(xc)  # (batch_size, num_classes)

        # Value prediction branch
        xv = F.relu(self.fc_value1(flattened))  # (batch_size, 32)
        value = torch.tanh(self.fc_value2(xv)).squeeze(-1)  # (batch_size,) in [-1, 1]

        return class_logits, value