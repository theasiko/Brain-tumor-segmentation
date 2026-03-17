import torch


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    pred = (pred > thr).float()
    target = target.float()

    inter = (pred * target).sum()
    union = pred.sum() + target.sum()

    return (2 * inter + 1e-8) / (union + 1e-8)


def iou_score(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    pred = (pred > thr).float()
    target = target.float()

    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter

    return (inter + 1e-8) / (union + 1e-8)


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    pred = (pred > thr).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return correct / total
