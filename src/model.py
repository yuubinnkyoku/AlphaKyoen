import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallResNet(nn.Module):
    def __init__(self, num_blocks=4, filters=64):
        super().__init__()
        
        # 入力: 3チャネル (自分, 相手, 手番)
        self.conv_in = nn.Conv2d(3, filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(filters)
        
        # ResBlockの積み重ね
        self.blocks = nn.ModuleList([
            ResBlock(filters) for _ in range(num_blocks)
        ])
        
        # Policy Head (どこに打つか: 81通り)
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 9 * 9, 81)
        
        # Value Head (勝率: -1 ~ 1)
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.bn_in(self.conv_in(x)))
        
        for block in self.blocks:
            h = block(h)
            
        # Policy
        p = F.relu(self.policy_conv(h))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1) # Log probability
        
        # Value
        v = F.relu(self.value_conv(h))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out