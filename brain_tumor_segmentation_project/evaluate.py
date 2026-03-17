import argparse

import torch
from torch.utils.data import DataLoader

from src.config import DEFAULT_DATASET_NAME, get_device
from src.data import create_splits
from src.model import UNet
from src.train_utils import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained brain tumor segmentation model.")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    _, _, test_ds = create_splits(args.dataset_name, img_size=args.img_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    dice, iou, acc = evaluate_model(model, test_loader, device)

    print(f"TEST Dice: {dice:.4f}")
    print(f"TEST IoU: {iou:.4f}")
    print(f"TEST Pixel Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
