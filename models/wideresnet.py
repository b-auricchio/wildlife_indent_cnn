import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#basic residual block
class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_inout = (in_features == out_features)
        self.shortcut = (not self.equal_inout) and nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        
    def forward(self, x):
        if not self.equal_inout:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_inout else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equal_inout else self.shortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, n_layers, in_features, out_features, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_features, out_features, n_layers, stride, drop_rate)
    def _make_layer(self, block, in_features, out_features, n_layers, stride, drop_rate):
        layers = []
        for i in range(int(n_layers)):
            layers.append(block(i == 0 and in_features or out_features, out_features, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, in_channels, depth_factor, widen_factor, num_classes, drop_rate=0.0):
        super(WideResNet, self).__init__()
        k = widen_factor
        n = depth_factor

        features = [16, 16*k, 32*k, 64*k]
        
        block = BasicBlock
        self.conv1 = nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, features[0], features[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, features[1], features[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, features[2], features[3], block, 2, drop_rate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(features[-1])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(features[-1], num_classes)
        self.features = features[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.gap(out)
        out = self.flatten(out)
        return self.fc(out)
