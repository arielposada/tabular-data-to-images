import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.train_utils import get_criterion, get_device, get_optimizer, train_epoch


class _LSTMNet(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=1,        # each feature is a timestep of dimension 1
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, n_features, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])  # last timestep hidden state


class TabularLSTMClassifier:
    """sklearn-compatible LSTM classifier for tabular data.

    Treats each feature as a timestep (input_size=1) so the sequence length
    equals the number of features. Compatible with sklearn Pipeline.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        lr: float = 1e-3,
        num_epochs: int = 50,
        batch_size: int = 256,
        device=None,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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

        self.model_ = _LSTMNet(
            num_classes=n_classes,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
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
                print(f"[LSTM epoch {epoch+1}/{self.num_epochs}] loss: {loss:.4f}, acc: {acc:.2f}%")

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
