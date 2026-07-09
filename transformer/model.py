import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.train_utils import get_criterion, get_device, get_optimizer, train_epoch


class _TransformerNet(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_classes: int,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
    ):
        super().__init__()
        self.embed = nn.Linear(1, d_model)  # project each scalar feature to d_model
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1) -> embed -> (batch, seq_len, d_model)
        x = self.embed(x) + self.pos_encoder[:, : x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # mean pooling over all feature tokens
        return self.fc(x)


class TabularTransformerClassifier:
    """sklearn-compatible mini Transformer classifier for tabular data.

    Treats each feature as a token (input_size=1) so the sequence length
    equals the number of features. Compatible with sklearn Pipeline.
    """

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        lr: float = 1e-3,
        num_epochs: int = 50,
        batch_size: int = 256,
        device=None,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device(device) if device else get_device()

    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        n_classes = len(self.classes_)

        self.model_ = _TransformerNet(
            seq_len=self.n_features_,
            num_classes=n_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
        ).to(self.device)

        X_t = torch.tensor(X).unsqueeze(-1)  # (n, n_features, 1)
        y_t = torch.tensor(y)
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )

        criterion = get_criterion()
        optimizer = get_optimizer(self.model_, lr=self.lr)

        for epoch in range(self.num_epochs):
            loss, acc = train_epoch(
                self.model_, loader, optimizer, criterion, self.device
            )
            if (epoch + 1) % 10 == 0 or epoch == self.num_epochs - 1:
                print(f"[Transformer epoch {epoch+1}/{self.num_epochs}] loss: {loss:.4f}, acc: {acc:.2f}%")

        return self

    def _forward_batched(self, X):
        X = np.array(X, dtype=np.float32)
        self.model_.eval()
        X_t = torch.tensor(X).unsqueeze(-1)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        all_logits = []
        with torch.no_grad():
            for (batch,) in loader:
                all_logits.append(self.model_(batch.to(self.device)).cpu())
        return torch.cat(all_logits, dim=0)

    def predict(self, X):
        logits = self._forward_batched(X)
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, X):
        logits = self._forward_batched(X)
        return torch.softmax(logits, dim=1).numpy()
