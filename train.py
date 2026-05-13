from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


DATA_PATH = Path("preprocessed_data.json")
CHECKPOINT_PATH = Path("train_checkpoint.pt")
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.9
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class MotionPredictor(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_embedding = nn.Embedding(sequence_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]
        positions = torch.arange(sequence_length, device=x.device)
        positional_embeddings = self.positional_embedding(positions).unsqueeze(0)
        hidden = self.input_projection(x) + positional_embeddings
        encoded = self.transformer(hidden)
        last_token = encoded[:, -1]
        return self.prediction_head(last_token)


def load_data(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    with path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    x = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["Y"], dtype=torch.float32)
    return x, y, data


def build_dataloaders(x: torch.Tensor, y: torch.Tensor) -> tuple[DataLoader, DataLoader]:
    dataset = SequenceDataset(x, y)
    train_size = max(1, int(len(dataset) * TRAIN_SPLIT))
    val_size = max(1, len(dataset) - train_size)
    if train_size + val_size > len(dataset):
        train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    x, y, data = load_data(DATA_PATH)
    if x.ndim != 3:
        raise ValueError(f"Expected X to have shape [samples, sequence_length, features], got {tuple(x.shape)}")
    if y.ndim != 2:
        raise ValueError(f"Expected Y to have shape [samples, output_dim], got {tuple(y.shape)}")

    train_loader, val_loader = build_dataloaders(x, y)

    model = MotionPredictor(input_dim=x.shape[-1], sequence_length=x.shape[1]).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print(f"Training on {DEVICE} with {len(train_loader.dataset)} train samples and {len(val_loader.dataset)} val samples")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss = run_epoch(model, val_loader, loss_fn, None, DEVICE)

        print(f"Epoch {epoch:03d}/{EPOCHS} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": x.shape[-1],
                    "sequence_length": x.shape[1],
                    "x_mean": data.get("X_mean"),
                    "x_std": data.get("X_std"),
                    "y_mean": data.get("Y_mean"),
                    "y_std": data.get("Y_std"),
                    "best_val_loss": best_val_loss,
                },
                CHECKPOINT_PATH,
            )

    print(f"Saved best checkpoint to {CHECKPOINT_PATH} with val loss {best_val_loss:.6f}")

    torch.save(
        {
            "embeddings": model.input_projection.state_dict(),
            "positional_embedding": model.positional_embedding.state_dict(),
            "transformer": model.transformer.state_dict(),
            "prediction_head": model.prediction_head.state_dict(),
        },
        "robot_transformer.pt",
    )
    print("Saved component state dicts to robot_transformer.pt")


if __name__ == "__main__":
    main()