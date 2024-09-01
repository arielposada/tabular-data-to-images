import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        seq_len=900,
        num_classes=8,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(input_size, d_model)  # Project input to d_model size
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
        x = self.embed(x)  # Shape: [batch_size, seq_len, d_model]
        x += self.pos_encoder[:, : x.size(1), :]  # Add positional encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Aggregate representations
        x = self.fc(x)
        return x


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        self.num_layers = 2
        self.hidden_size = 32
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # Output shape: (batch_size, sequence_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x):
        x = self.forward(x)
        return torch.argmax(x, dim=-1)


def create_model(model_name, input_size, seq_len, num_classes, device="cuda"):
    if model_name == "transformer":
        return SimpleTransformer(
            input_size=input_size,
            seq_len=seq_len,
            num_classes=num_classes,
        ).to(device)
    elif model_name == "lstm":
        return SimpleLSTM(
            input_size=input_size,
            num_classes=num_classes,
        ).to(device)
    else:
        raise ValueError(f"Model {model_name} not recognized")
