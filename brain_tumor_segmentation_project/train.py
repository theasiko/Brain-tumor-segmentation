import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import DEFAULT_DATASET_NAME, DEFAULT_SEED, get_device, set_seed
from src.data import create_splits
from src.model import UNet
from src.train_utils import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for brain tumor segmentation.")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_ds, val_ds, _ = create_splits(args.dataset_name, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_path=best_path,
    )

    print(f"Training finished. Best model saved to: {best_path}")
    print("Final history:", history)


if __name__ == "__main__":
    main()
