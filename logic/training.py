import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image
from timm import create_model
from torch.utils.data import Dataset, DataLoader, Subset

from .evaluation import evaluate
from .params import (
    EPOCHS,
    BATCH_SIZE,
    PATIENCE_EPOCHS,
    WARMUP_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CLASSES
)


# ===========================
# Dataset (SAFE FOR K-FOLD)
# ===========================
class LeafDataset(Dataset):
    def __init__(self, csv_df, aug_images_dir):
        self.img_dir = aug_images_dir

        # originals only (KFold-safe)
        self.original_images = csv_df["image"].tolist()

        self.all_images = os.listdir(aug_images_dir)
        self.label_map = dict(zip(csv_df["image"], csv_df["labels"]))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.original_images)

    def encode_labels(self, label_str):
        vec = torch.zeros(len(CLASSES))
        for lbl in label_str.split():
            vec[CLASSES.index(lbl)] = 1.0
        return vec

    def __getitem__(self, idx):
        orig = self.original_images[idx]
        stem = os.path.splitext(orig)[0]

        # randomly pick ONE augmented version
        candidates = [f for f in self.all_images if f.startswith(stem)]
        fname = np.random.choice(candidates)

        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        img = self.transform(img)

        labels = self.encode_labels(self.label_map[orig])
        return img, labels


if __name__ == "__main__":
    csv_labels = pd.read_csv("data/train.csv")
    aug_images_dir = "data/aug_train_images/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(csv_labels), 1):
        print(f"\nTraining fold {fold}...")

        base_dataset = LeafDataset(csv_labels, aug_images_dir)

        train_dataset = Subset(base_dataset, train_idx)
        val_dataset = Subset(base_dataset, val_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=4, pin_memory=True
        )

        model = create_model(
            "efficientnetv2_rw_s",
            pretrained=True,
            num_classes=len(CLASSES)
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )

        best_val = float("inf")
        wait = 0
        for epoch in range(EPOCHS):
            if epoch < WARMUP_EPOCHS:
                scale = 0.1 + 0.9 * (epoch / WARMUP_EPOCHS)
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE * scale

            model.train()
            train_loss = 0.0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)

            train_loss /= len(train_loader.dataset)

            val_loss, micro_f1, macro_f1 = evaluate(
                model, val_loader, criterion, device
            )

            print(
                f"Fold {fold} | Epoch {epoch+1:02d} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | "
                f"Micro {micro_f1:.4f} | " # global F1
                f"Macro {macro_f1:.4f}"    # avg F1 from all classes
            )

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                torch.save(model.state_dict(), f"best_fold_{fold}.pth")
            else:
                wait += 1
                if wait >= PATIENCE_EPOCHS:
                    print("Early stopping")
                    break

            scheduler.step()

        del model, optimizer, scheduler
        torch.cuda.empty_cache()
