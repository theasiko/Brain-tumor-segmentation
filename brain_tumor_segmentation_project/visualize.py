import argparse

import torch

from src.config import DEFAULT_DATASET_NAME, get_device
from src.data import create_splits
from src.model import UNet
from src.viz import visualize_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize predictions of trained segmentation model.")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    train_ds, val_ds, test_ds = create_splits(args.dataset_name, img_size=args.img_size)
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    dataset = split_map[args.split]

    model = UNet().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    for idx in args.indices:
        visualize_segmentation(model, dataset, idx=idx, device=device, thr=args.threshold)


if __name__ == "__main__":
    main()
