from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.losses import combined_loss
from src.metrics import dice_coeff, iou_score, pixel_accuracy


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = combined_loss(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate_model(model, loader, device) -> Tuple[float, float, float]:
    model.eval()

    dice_scores = []
    iou_scores = []
    acc_scores = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.sigmoid(model(x))

            for i in range(x.size(0)):
                dice_scores.append(dice_coeff(preds[i], y[i]).item())
                iou_scores.append(iou_score(preds[i], y[i]).item())
                acc_scores.append(pixel_accuracy(preds[i], y[i]).item())

    return float(np.mean(dice_scores)), float(np.mean(iou_scores)), float(np.mean(acc_scores))


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int,
    checkpoint_path,
) -> Dict[str, List[float]]:
    checkpoint_path = Path(checkpoint_path)
    history = {"train_loss": [], "val_dice": [], "val_iou": [], "val_acc": []}
    best_dice = -1.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_dice, val_iou, val_acc = evaluate_model(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_iou={val_iou:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")

    return history
