from __future__ import annotations

import torch
import torch.nn as nn


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        tiny: bool = False,
    ):
        super().__init__()
        if tiny:
            hidden_dim = min(hidden_dim, 64)
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualTCNBlock(hidden_dim, dilation=1, dropout=dropout),
            ResidualTCNBlock(hidden_dim, dilation=2, dropout=dropout),
            ResidualTCNBlock(hidden_dim, dilation=4, dropout=dropout),
            ResidualTCNBlock(hidden_dim, dilation=8, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
