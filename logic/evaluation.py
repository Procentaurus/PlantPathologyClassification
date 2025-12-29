from sklearn.metrics import f1_score
import numpy as np
import torch


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()

    val_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()

        all_targets.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    micro_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    val_loss /= len(dataloader.dataset)

    return val_loss, micro_f1, macro_f1
