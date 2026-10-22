"""Sequence models (LSTM, PatchTST) trained on lookback windows of the per-row features.

The window for row r is rows r-L+1 .. r of the same ticker (all dates <= t). Feature scaling and
target standardisation use the fold's train rows only. See PLAN.md, "Sequence tensors".
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from stockbench.config import NetConfig
from stockbench.targets import TASKS

log = logging.getLogger(__name__)


@dataclass
class Panel:
    F: np.ndarray  # (N, d) float32 raw features, rows in (ticker, date) order
    pos: np.ndarray  # (N,) position of each row inside its ticker block
    lookback: int

    def has_window(self) -> np.ndarray:
        return self.pos >= self.lookback - 1

    def windows(self, X: np.ndarray) -> np.ndarray:
        """(N-L+1, L, d) zero-copy view of X; the window for row r is windows[r - L + 1]."""
        return np.lib.stride_tricks.sliding_window_view(X, (self.lookback, X.shape[1]))[:, 0]


def make_panel(
    df: pd.DataFrame, feature_cols: list[str], lookback: int
) -> tuple[pd.DataFrame, Panel]:
    """Sort a (date, ticker)-indexed frame ticker-major and build the Panel over it."""
    df = df.sort_index(level=["ticker", "date"])
    pos = df.groupby(level="ticker", sort=False).cumcount().to_numpy()
    F = np.ascontiguousarray(df[feature_cols].to_numpy(dtype=np.float32))
    return df, Panel(F, pos, lookback)


class LSTMNet(nn.Module):
    def __init__(self, d: int, cfg: NetConfig):
        super().__init__()
        drop = cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(d, cfg.lstm_hidden, cfg.lstm_layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Dropout(cfg.lstm_dropout), nn.Linear(cfg.lstm_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d) -> (B,)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class PatchTSTNet(nn.Module):
    """Channel-independent PatchTST encoder.

    Every feature channel is patched and encoded on its own with shared weights; a shared flatten
    head reads one number per channel and a linear mixer combines the channels. No RevIN: the
    level of the inputs (for example the current realised vol) is signal here, not nuisance.
    """

    def __init__(self, d: int, cfg: NetConfig):
        super().__init__()
        self.patch_len, self.stride = cfg.patch_len, cfg.patch_stride
        n_patches = (cfg.lookback - cfg.patch_len) // cfg.patch_stride + 1
        self.embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, dim_feedforward=2 * cfg.d_model, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )  # fmt: skip
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.channel_head = nn.Linear(n_patches * cfg.d_model, 1)
        self.mix = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d) -> (B,)
        B, _, d = x.shape
        p = x.transpose(1, 2).unfold(-1, self.patch_len, self.stride)  # (B, d, P, patch_len)
        z = self.embed(p.reshape(B * d, p.shape[2], self.patch_len)) + self.pos
        z = self.norm(self.encoder(z))  # (B*d, P, d_model)
        c = self.channel_head(self.drop(z.flatten(1))).view(B, d)
        return self.mix(self.drop(c)).squeeze(-1)


ARCH = {"lstm": LSTMNet, "patchtst": PatchTSTNet}


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SeqModel:
    """fit(panel, y_all, train_mask, val_mask) / predict(panel, mask) over lookback windows."""

    needs_history = True

    def __init__(self, arch: str, task: str, seed: int, cfg: NetConfig):
        self.arch, self.task, self.seed, self.cfg = arch, task, seed, cfg
        self.classify = TASKS[task] == "classification"
        self.device = _device(cfg.device)

    def _scaled_windows(self, panel: Panel) -> np.ndarray:
        X = np.clip((panel.F - self.x_mean) / self.x_std, -self.cfg.clip, self.cfg.clip)
        return panel.windows(np.ascontiguousarray(X, dtype=np.float32))

    def _batch(self, W: np.ndarray, rows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(W[rows - self.cfg.lookback + 1]).to(self.device)

    @torch.no_grad()
    def _forward(self, net: nn.Module, W: np.ndarray, rows: np.ndarray) -> np.ndarray:
        net.eval()
        chunks = np.array_split(rows, max(1, len(rows) // self.cfg.batch_size))
        return np.concatenate([net(self._batch(W, idx)).float().cpu().numpy() for idx in chunks])

    def fit(self, panel: Panel, y, train_mask, val_mask):
        cfg = self.cfg
        _seed_everything(self.seed)
        y = np.asarray(y, dtype=float)
        self.x_mean = panel.F[train_mask].mean(0)
        self.x_std = panel.F[train_mask].std(0) + 1e-8
        if self.classify:
            self.y_mean, self.y_std = 0.0, 1.0
        else:
            self.y_mean, self.y_std = float(y[train_mask].mean()), float(y[train_mask].std())
        yt = ((y - self.y_mean) / self.y_std).astype(np.float32)

        ok = panel.has_window()
        if (val_mask & ~ok).any():
            raise ValueError("validation rows without a full lookback window")
        rows_tr, rows_va = np.flatnonzero(train_mask & ok), np.flatnonzero(val_mask & ok)
        W = self._scaled_windows(panel)
        net = ARCH[self.arch](panel.F.shape[1], cfg).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        crit = nn.BCEWithLogitsLoss() if self.classify else nn.MSELoss()
        rng = np.random.default_rng(self.seed)
        best, best_state, best_epoch, bad = np.inf, None, 0, 0
        for epoch in range(1, cfg.max_epochs + 1):
            net.train()
            n_batches = max(1, len(rows_tr) // cfg.batch_size)
            for idx in np.array_split(rng.permutation(rows_tr), n_batches):
                xb, yb = self._batch(W, idx), torch.from_numpy(yt[idx]).to(self.device)
                opt.zero_grad()
                loss = crit(net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            out = torch.from_numpy(self._forward(net, W, rows_va))
            val = float(crit(out, torch.from_numpy(yt[rows_va])))
            if val < best - 1e-6:
                best, best_state, best_epoch, bad = val, copy.deepcopy(net.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        net.load_state_dict(best_state)
        self.net = net.eval()
        self.info = {"epochs": epoch, "best_epoch": best_epoch, "val_loss": round(best, 5)}
        return self

    def predict(self, panel: Panel, mask) -> np.ndarray:
        rows = np.flatnonzero(mask)
        if not panel.has_window()[rows].all():
            raise ValueError("rows to predict without a full lookback window")
        out = self._forward(self.net, self._scaled_windows(panel), rows).astype(float)
        return 1 / (1 + np.exp(-out)) if self.classify else out * self.y_std + self.y_mean
