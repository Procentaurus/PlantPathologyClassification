import os
import torch
import pandas as pd

from pathlib import Path
from PIL import Image
from torchvision import transforms
from timm import create_model
from torch.utils.data import Dataset, DataLoader

from .params import (BATCH_SIZE,
                     THRESHOLD,
                     CLASSES,
                     IMG_HEIGHT,
                     IMG_WIDTH)

# ===========================
# CONFIG
# ===========================
CHECKPOINTS = [
    "models/2.1/best_fold_1.pth",
    "models/2.1/best_fold_2.pth",
    "models/2.1/best_fold_3.pth",
    "models/2.1/best_fold_4.pth",
    "models/2.1/best_fold_5.pth",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# DATASET
# ===========================
class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        img = self.transform(img)
        return img, fname


# ===========================
# LOAD MODELS (ENSEMBLE)
# ===========================
models = []
for ckpt in CHECKPOINTS:
    model = create_model(
        "efficientnetv2_rw_s",
        pretrained=False,
        num_classes=len(CLASSES),
    )
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    models.append(model)

# ===========================
# INFERENCE
# ===========================
base_dir = Path(__file__).resolve().parent.parent
test_dir = base_dir / "data" / "resized_test_images"
dataset = TestDataset(test_dir)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

results = []

with torch.no_grad():
    for imgs, fnames in loader:
        imgs = imgs.to(DEVICE)

        # ensemble mean
        logits = torch.zeros((imgs.size(0), len(CLASSES)), device=DEVICE)
        for model in models:
            logits += model(imgs)
        logits /= len(models)

        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).cpu().numpy()

        for fname, pred in zip(fnames, preds):
            labels = [CLASSES[i] for i, v in enumerate(pred) if v == 1]
            if len(labels) == 0:
                labels = ["healthy"]  # fallback (important)
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
