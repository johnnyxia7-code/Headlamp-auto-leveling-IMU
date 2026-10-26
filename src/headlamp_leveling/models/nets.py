"""Neural pitch estimators.

All models map a normalised window (B, C, W) to (B, 2):
    [:, 0]  static load pitch (deg)    -> drives the headlamp leveling
    [:, 1]  road grade (deg)           -> auxiliary target, regularises training

They are deliberately small (tens of k parameters) so they fit headlamp /
body-controller class MCUs after int8 quantisation.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, dilation: int, stride: int, dropout: float):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.skip = (nn.Identity() if c_in == c_out and stride == 1
                     else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.drop(self.bn2(self.conv2(y)))
        return self.act(y + self.skip(x))


class TCNRegressor(nn.Module):
    """Strided dilated residual CNN + mean/max pooling over time."""

    def __init__(self, in_ch: int, channels: Sequence[int] = (32, 48, 64, 64), kernel_size: int = 5,
                 dropout: float = 0.1, n_out: int = 2):
        super().__init__()
        layers, c = [], in_ch
        for i, c_out in enumerate(channels):
            layers.append(ResBlock(c, c_out, kernel_size, dilation=2 ** min(i, 3), stride=2, dropout=dropout))
            c = c_out
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(h)


class GRURegressor(nn.Module):
    """Recurrent estimator; can run sample-by-sample on target with a persistent state."""

    def __init__(self, in_ch: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1, n_out: int = 2):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv1d(in_ch, hidden, 5, stride=2, padding=2), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x).transpose(1, 2)
        out, _ = self.gru(h)
        return self.head(out[:, -1])


class CovarianceFeatures(nn.Module):
    """Differentiable second-order statistics of learned signal projections.

    The pitch is encoded in the *ratio* of covariances (e.g. cov(f_z, a) /
    cov(f_x, a)), which plain ReLU CNNs approximate poorly. This layer
    high-passes a set of learned 1x1 projections and returns their
    normalised cross-covariance matrix (upper triangle) for the head.
    """

    def __init__(self, in_ch: int, n_proj: int = 8, hp_kernel: int = 101):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, n_proj, 1)
        self.hp_kernel = hp_kernel
        iu = torch.triu_indices(n_proj, n_proj)
        self.register_buffer("iu0", iu[0], persistent=False)
        self.register_buffer("iu1", iu[1], persistent=False)
        self.out_dim = iu.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.proj(x)
        k = min(self.hp_kernel, p.shape[-1] | 1)
        trend = nn.functional.avg_pool1d(p, k, stride=1, padding=k // 2, count_include_pad=False)
        p = p - trend
        cov = p @ p.transpose(1, 2) / p.shape[-1]                       # (B, P, P)
        return cov[:, self.iu0, self.iu1]


class HybridRegressor(nn.Module):
    """Physics-structured model: covariance features + TCN context, fused in the head."""

    def __init__(self, in_ch: int, channels: Sequence[int] = (32, 48, 64, 64), kernel_size: int = 5,
                 dropout: float = 0.1, n_out: int = 2, n_proj: int = 8):
        super().__init__()
        self.cov = CovarianceFeatures(in_ch, n_proj)
        self.tcn = TCNRegressor(in_ch, channels, kernel_size, dropout, n_out=32)
        self.head = nn.Sequential(nn.Linear(self.cov.out_dim + 32, 96), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, n_out))
        self.cov_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cov(x) * self.cov_scale
        c = torch.sign(c) * torch.log1p(c.abs())                      # compress dynamic range
        return self.head(torch.cat([c, torch.relu(self.tcn(x))], dim=1))


def build_model(arch: str, in_ch: int, cfg) -> nn.Module:
    if arch == "tcn":
        return TCNRegressor(in_ch, cfg.channels, cfg.kernel_size, cfg.dropout)
    if arch == "gru":
        return GRURegressor(in_ch, cfg.gru_hidden, cfg.gru_layers, cfg.dropout)
    if arch == "hybrid":
        return HybridRegressor(in_ch, cfg.channels, cfg.kernel_size, cfg.dropout)
    raise ValueError(f"unknown arch {arch!r} (tcn | gru | hybrid)")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["TCNRegressor", "GRURegressor", "HybridRegressor", "build_model", "count_params"]
