import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessCoreNet(nn.Module):
    def __init__(self, in_channels=25, num_blocks=4, hidden_channels=128):
        super().__init__()
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)
        self.blocks = nn.ModuleList([ChessResidualBlock(hidden_channels) for _ in range(num_blocks)])
        
    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.blocks:
            x = block(x)
        return x

class MoveClassifierNet(nn.Module):
    def __init__(self, core_net, num_classes=6, hidden_channels=128):
        super().__init__()
        self.core = core_net
        self.conv_reduce = nn.Conv2d(hidden_channels, 16, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(16)
        
        self.fc_class1 = nn.Linear(16 * 8 * 8, 256)
        self.dropout = nn.Dropout(p=0.4) 
        self.fc_class2 = nn.Linear(256, num_classes)
        
        self.fc_value1 = nn.Linear(16 * 8 * 8, 32)
        self.fc_value2 = nn.Linear(32, 1)

    def forward(self, x):
        features = self.core(x)
        x_shared = F.relu(self.bn_reduce(self.conv_reduce(features)))
        flattened = x_shared.view(x_shared.size(0), -1)
        
        xc = F.relu(self.fc_class1(flattened))
        xc = self.dropout(xc)              
        class_logits = self.fc_class2(xc)
        
        xv = F.relu(self.fc_value1(flattened))
        value = torch.tanh(self.fc_value2(xv)).squeeze(-1)
        
        return class_logits, value