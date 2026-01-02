import os
import torch
import pandas as pd

from pathlib import Path
from PIL import Image
from torchvision import transforms
from timm import create_model
from torch.utils.data import Dataset, DataLoader

from .params import (
    BATCH_SIZE,
    THRESHOLD,
    CLASSES,
    IMG_HEIGHT,
    IMG_WIDTH,
    MEAN_VECTOR,
    STD_VECTOR
)

# ===========================
# CONFIG
# ===========================
CHECKPOINT = "models/2.1/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# DATASET WITH TTA
# ===========================
class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = sorted(os.listdir(img_dir))

        resize = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
        normalize = transforms.Normalize(mean=MEAN_VECTOR, std=STD_VECTOR)

        self.tta_transforms = [
            transforms.Compose([resize, transforms.ToTensor(), normalize]),
            transforms.Compose([resize, transforms.RandomHorizontalFlip(p=1.0),
                                transforms.ToTensor(), normalize]),
            transforms.Compose([resize, transforms.RandomVerticalFlip(p=1.0),
                                transforms.ToTensor(), normalize]),
            transforms.Compose([resize, transforms.RandomHorizontalFlip(p=1.0),
                                transforms.RandomVerticalFlip(p=1.0),
                                transforms.ToTensor(), normalize]),
            transforms.Compose([
                transforms.Resize((IMG_HEIGHT + 40, IMG_WIDTH + 40)),
                transforms.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
                transforms.ToTensor(), normalize
            ]),
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        imgs = torch.stack([t(img) for t in self.tta_transforms])
        return imgs, fname


# ===========================
# LOAD ENSEMBLE MODELS
# ===========================
model = create_model(
    "efficientnetv2_rw_s",
    pretrained=False,
    num_classes=len(CLASSES),
)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ===========================
# INFERENCE WITH TTA + ENSEMBLE
# ===========================
base_dir = Path(__file__).resolve().parent.parent
test_dir = base_dir / "data" / "resized_test_images"
dataset = TestDataset(test_dir)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

results = []

with torch.no_grad():
    for imgs, fnames in loader:
        # imgs: (B, TTA, C, H, W)
        B, T, C, H, W = imgs.shape
        imgs = imgs.to(DEVICE).view(B * T, C, H, W)
        logits = model(imgs)

        # reshape back to (B, T, num_classes) and average over TTA
        logits = logits.view(B, T, -1).mean(dim=1)

        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).cpu().numpy()

        for fname, pred in zip(fnames, preds):
            labels = [CLASSES[i] for i, v in enumerate(pred) if v == 1]
            if not labels:
                labels = ["healthy"]  # fallback
            results.append({
                "image": fname,
                "labels": " ".join(labels)
            })

# ===========================
# SAVE SUBMISSION
# ===========================
submission = pd.DataFrame(results)
submission.to_csv("submission.csv", index=False)
print("submission.csv created ✔")
