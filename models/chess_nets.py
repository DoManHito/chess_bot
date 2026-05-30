import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessResidualBlock(nn.Module):
    """Базовый остаточный блок (как в AlphaZero) для обработки шахматной доски."""
    def __init__(self, channels=128):
        super().__init__()
        # ИСПРАВЛЕНО: Блок работает строго со скрытыми каналами (hidden_channels)
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
    """Общая свёрточная база для извлечения признаков позиции."""
    # ИСПРАВЛЕНО: По умолчанию теперь 25 входных каналов вместо 13
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

class AIPlayerNet(nn.Module):
    """Модель самого бота (Policy + Value)."""
    def __init__(self, core_net, hidden_channels=128):
        super().__init__()
        self.core = core_net
        
        # Голова Policy (выдает вероятности для всех возможных ~4096 ходов)
        self.policy_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096) 
        
        # Голова Value (оценка позиции от -1 до 1)
        self.value_conv = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        self.value_fc1 = nn.Linear(3 * 8 * 8, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        features = self.core(x)
        
        # Policy
        p = F.relu(self.policy_conv(features))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p) # Пропускать через Softmax будем в MCTS
        
        # Value
        v = F.relu(self.value_conv(features))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value

class MoveClassifierNet(nn.Module):
    """Модель классификатора с сохранением точной геометрии доски 8х8."""
    def __init__(self, core_net, num_classes=6, hidden_channels=128):
        super().__init__()
        self.core = core_net
        
        # ИСПРАВЛЕНО: Вместо пулинга сжимаем КАНАЛЫ (128 -> 16), сохраняя сетку 8x8
        self.conv_reduce = nn.Conv2d(hidden_channels, 16, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(16)
        
        # Финальный размер плоского тензора: 16 каналов * 8 * 8 клеток = 1024
        self.fc1 = nn.Linear(16 * 8 * 8, 256)
        self.dropout = nn.Dropout(p=0.4) 
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.core(x)
        
        # Уменьшаем глубину признаков, сохраняя тактическую точность клеток
        x = F.relu(self.bn_reduce(self.conv_reduce(features)))
        flattened = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(flattened))
        x = self.dropout(x)              
        
        class_logits = self.fc2(x)
        return class_logits