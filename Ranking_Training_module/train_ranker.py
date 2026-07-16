"""Train and export the production MMoE heavy ranker."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.feature_spec import (  # noqa: E402
    EMBEDDING_DIM,
    FEATURE_COUNT,
    FEATURE_ORDER,
    FEATURE_SPEC_VERSION,
    INPUT_DIM,
    RANKER_MODEL_VERSION,
)
from inference.ranker_service import MMoEHeavyRanker  # noqa: E402
from inference.value_function import VALUE_WEIGHTS  # noqa: E402
from config import REPOSITORY_EMBEDDING_VERSION  # noqa: E402


DATA_KEYS = (
    "user_embs",
    "repo_embs",
    "dense_features",
    "y_ctr",
    "y_save",
    "y_gh",
    "y_dwell",
    "y_follow",
)


def load_training_data(data_path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the arrays required by the ranker trainer."""
    with np.load(data_path) as data:
        missing_keys = [key for key in DATA_KEYS if key not in data]
        if missing_keys:
            raise ValueError(
                f"Training data is missing required arrays: {missing_keys}"
            )
        arrays = {key: np.asarray(data[key]) for key in DATA_KEYS}

    dense_features = arrays["dense_features"]
    if dense_features.ndim != 2 or dense_features.shape[1] != FEATURE_COUNT:
        actual_count = dense_features.shape[1] if dense_features.ndim == 2 else None
        raise ValueError(
            f"dense_features has {actual_count} columns; expected {FEATURE_COUNT} "
            f"in FEATURE_ORDER {FEATURE_ORDER}"
        )

    row_count = len(dense_features)
    mismatched = {
        key: len(value)
        for key, value in arrays.items()
        if len(value) != row_count
    }
    if mismatched:
        raise ValueError(
            f"All training arrays must contain {row_count} rows; got {mismatched}"
        )
    if row_count < 3:
        raise ValueError("Training data must contain at least 3 rows")

    for embedding_key in ("user_embs", "repo_embs"):
        embeddings = arrays[embedding_key]
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"{embedding_key} must have shape (rows, {EMBEDDING_DIM})"
            )

    return arrays


def standardize_dense_features(
    dense_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply StandardScaler-compatible population mean and scale."""
    dense_float = np.asarray(dense_features, dtype=np.float64)
    mean = dense_float.mean(axis=0)
    scale = dense_float.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    scaled = ((dense_float - mean) / scale).astype(np.float32)
    return scaled, mean, scale


def build_dataloaders(
    X: np.ndarray,
    labels: tuple[np.ndarray, ...],
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Create an 80/20 chronological train/validation split."""
    split_index = int(len(X) * 0.8)
    if split_index < 2 or split_index >= len(X):
        raise ValueError("The chronological split requires at least 2 training rows")

    def make_dataset(start: int, end: int) -> TensorDataset:
        tensors = [torch.from_numpy(X[start:end])]
        tensors.extend(
            torch.from_numpy(label[start:end])
            for label in labels
        )
        return TensorDataset(*tensors)

    train_dataset = make_dataset(0, split_index)
    val_dataset = make_dataset(split_index, len(X))
    effective_batch_size = min(batch_size, len(train_dataset))
    drop_last = len(train_dataset) % effective_batch_size == 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def compute_batch_loss(
    model: MMoEHeavyRanker,
    batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
    device: torch.device,
    bce_loss: nn.BCELoss,
    mse_loss: nn.MSELoss,
) -> tuple[torch.Tensor, int]:
    """Compute the summed five-head training objective for one batch."""
    x, y_ctr, y_save, y_gh, y_dwell, y_follow = (
        tensor.to(device) for tensor in batch
    )
    predictions = [prediction.reshape(-1) for prediction in model(x)]
    p_ctr, p_save, p_gh, p_dwell, p_follow = predictions
    loss = (
        bce_loss(p_ctr, y_ctr)
        + bce_loss(p_save, y_save)
        + bce_loss(p_gh, y_gh)
        + mse_loss(p_dwell, y_dwell)
        + bce_loss(p_follow, y_follow)
    )
    return loss, x.shape[0]


def evaluate(
    model: MMoEHeavyRanker,
    data_loader: DataLoader,
    device: torch.device,
    bce_loss: nn.BCELoss,
    mse_loss: nn.MSELoss,
) -> float:
    """Return the example-weighted validation loss."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in data_loader:
            loss, batch_count = compute_batch_loss(
                model,
                batch,
                device,
                bce_loss,
                mse_loss,
            )
            total_loss += loss.item() * batch_count
            total_examples += batch_count

    if total_examples == 0:
        raise ValueError("Validation split is empty")
    return total_loss / total_examples


def train(
    *,
    data_path: str | Path,
    epochs: int,
    lr: float,
    batch_size: int,
    output_dir: str | Path,
    model_version: str = RANKER_MODEL_VERSION,
) -> dict[str, object]:
    """Train the heavy ranker and write its production artifacts."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for BatchNorm training")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version must be a non-empty string")
    model_version = model_version.strip()

    arrays = load_training_data(data_path)
    scaled_dense, scaler_mean, scaler_scale = standardize_dense_features(
        arrays["dense_features"]
    )
    X = np.concatenate(
        [
            arrays["user_embs"].astype(np.float32, copy=False),
            arrays["repo_embs"].astype(np.float32, copy=False),
            scaled_dense,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if X.shape[1] != INPUT_DIM:
        raise ValueError(f"Training input has {X.shape[1]} columns; expected {INPUT_DIM}")

    labels = tuple(
        arrays[key].astype(np.float32, copy=False).reshape(-1)
        for key in ("y_ctr", "y_save", "y_gh", "y_dwell", "y_follow")
    )
    train_loader, val_loader = build_dataloaders(X, labels, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MMoEHeavyRanker(INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    final_train_loss = 0.0
    final_val_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, batch_count = compute_batch_loss(
                model,
                batch,
                device,
                bce_loss,
                mse_loss,
            )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_count
            total_train_examples += batch_count

        if total_train_examples == 0:
            raise ValueError("Training split did not produce any batches")
        final_train_loss = total_train_loss / total_train_examples
        final_val_loss = evaluate(
            model,
            val_loader,
            device,
            bce_loss,
            mse_loss,
        )
        print(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss={final_train_loss:.6f} - "
            f"val_loss={final_val_loss:.6f}"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "heavy_ranker.pt"
    scaler_path = destination / "feature_scaler.json"
    manifest_path = destination / "model_manifest.json"

    torch.save(model.state_dict(), model_path)
    scaler_path.write_text(
        json.dumps(
            {
                "mean": scaler_mean.tolist(),
                "scale": scaler_scale.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "model_file": model_path.name,
        "scaler_file": scaler_path.name,
        "model_version": model_version,
        "embedding_version": REPOSITORY_EMBEDDING_VERSION,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "input_dim": INPUT_DIM,
        "embedding_dim": EMBEDDING_DIM,
        "feature_count": FEATURE_COUNT,
        "value_weights": VALUE_WEIGHTS,
        "training_date": date.today().isoformat(),
        "epochs": epochs,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved manifest to {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="training_data.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", default="inference/")
    parser.add_argument("--model-version", default=RANKER_MODEL_VERSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_path=args.data_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        model_version=args.model_version,
    )


if __name__ == "__main__":
    main()
