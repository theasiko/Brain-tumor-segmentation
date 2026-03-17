from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def apply_clahe(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def preprocess_image(img, img_size: int) -> np.ndarray:
    img = np.array(img.convert("L"))
    img = apply_clahe(img)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    return img.astype(np.float32)


def segmentation_to_mask(segmentation: Iterable, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly in segmentation:
        pts = np.array(poly).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)

    return mask


class BrainTumorDataset(Dataset):
    def __init__(self, hf_dataset, img_size: int = 128, augment: bool = False):
        self.ds = hf_dataset
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]

        image = preprocess_image(sample["image"], self.img_size)

        mask = segmentation_to_mask(
            sample["segmentation"],
            sample["height"],
            sample["width"],
        )
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            if np.random.rand() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        image = torch.tensor(image).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask


def create_splits(dataset_name: str, img_size: int = 128) -> Tuple[BrainTumorDataset, BrainTumorDataset, BrainTumorDataset]:
    dataset = load_dataset(dataset_name)

    train_ds = BrainTumorDataset(dataset["train"], img_size=img_size, augment=True)
    val_ds = BrainTumorDataset(dataset["valid"], img_size=img_size, augment=False)
    test_ds = BrainTumorDataset(dataset["test"], img_size=img_size, augment=False)

    return train_ds, val_ds, test_ds
