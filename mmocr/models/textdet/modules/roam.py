import torch
import torch.nn as nn


# Residual Orthogonal Attention Module (ROAM)
class ROAM(nn.Module):
    def __init__(self, in_channels):
        super(ROAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 1
        self.horizontal = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1)),
            nn.Sigmoid()
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        h = self.horizontal(y)
        v = self.vertical(y)
        ort = h + v
        x = x * ort + x
        return h, v, x
