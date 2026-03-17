import matplotlib.pyplot as plt
import torch


def visualize_segmentation(model, dataset, idx: int, device, thr: float = 0.5) -> None:
    model.eval()

    image, mask = dataset[idx]
    image = image.to(device)

    with torch.no_grad():
        pred = model(image.unsqueeze(0))
        pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

    image_np = image[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))

    axs[0].imshow(image_np, cmap="gray")
    axs[0].set_title("Input MRI")
    axs[0].axis("off")

    axs[1].imshow(mask_np, cmap="gray")
    axs[1].set_title("Ground Truth Segmentation")
    axs[1].axis("off")

    axs[2].imshow(pred, cmap="hot")
    axs[2].set_title("Predicted Probability Map")
    axs[2].axis("off")

    axs[3].imshow(image_np, cmap="gray")
    axs[3].imshow(pred > thr, alpha=0.5, cmap="Reds")
    axs[3].set_title("Overlay (Prediction)")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()
