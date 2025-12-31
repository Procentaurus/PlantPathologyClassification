import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image
from timm import create_model
from torch.utils.data import (Dataset,
                              DataLoader,
                              Subset)

from .evaluation import evaluate
from .params import (EPOCHS,
                     BATCH_SIZE,
                     PATIENCE_EPOCHS,
                     WARMUP_EPOCHS,
                     LEARNING_RATE,
                     WEIGHT_DECAY)


CLASSES = [
    'healthy',
    'scab',
    'frog_eye_leaf_spot',
    'rust',
    'powdery_mildew',
    'complex'
]

class LeafDataset(Dataset):

    def __init__(self, csv_df, aug_images_dir):
        self.img_dir = aug_images_dir

        # All augmented + original images
        self.images = [
            f for f in os.listdir(aug_images_dir)
            if f.lower().endswith(('.jpg'))
        ]

        # Map: original_image.jpg -> label_string
        self.label_map = {
            row['image']: row['labels'] for _, row in csv_df.iterrows()
        }

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def encode_labels(self, label_str):
        label_vec = torch.zeros(len(CLASSES), dtype=torch.float32)
        for lbl in label_str.split():
            if lbl in CLASSES:
                label_vec[CLASSES.index(lbl)] = 1.0
        return label_vec

    def _get_original_name(self, filename):
        """
        800113bb65efe69e_crop.jpg -> 800113bb65efe69e.jpg
        """
        base, ext = os.path.splitext(filename)
        original = base.split('_')[0] + ext
        return original

    def __getitem__(self, idx):
        fname = self.images[idx]
        img_path = os.path.join(self.img_dir, fname)

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        original_name = self._get_original_name(fname)
        label_str = self.label_map[original_name]
        labels = self.encode_labels(label_str)
        return image, labels


if __name__ == "__main__":
    labels = pd.read_csv("data/train.csv")
    aug_images_dir = "data/aug_train_images/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===========================
    # 5-Fold Training
    # ===========================
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 0
    for train_idx, val_idx in kf.split(labels):
        fold += 1
        print(f"Training fold {fold}...")

        train_dataset = Subset(LeafDataset(labels, aug_images_dir=aug_images_dir), train_idx)
        val_dataset = Subset(LeafDataset(labels, aug_images_dir=aug_images_dir), val_idx)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # ===========================
        # Model
        # ===========================
        model = create_model('efficientnetv2_rw_s', pretrained=True, num_classes=6)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=EPOCHS)
        history = {
            "train_loss": [],
            "val_loss": [],
            "micro_f1": [],
            "macro_f1": []
        }
        best_val_loss = float('inf')
        patience = PATIENCE_EPOCHS
        wait = 0
        for epoch in range(EPOCHS):
            # Warmup
            if epoch < WARMUP_EPOCHS:
                lr_scale = 0.1 + 0.9 * (epoch / WARMUP_EPOCHS)
                for pg in optimizer.param_groups:
                    pg['lr'] = LEARNING_RATE * lr_scale

            # Training
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            # ---- Validation ----
            val_loss, micro_f1, macro_f1 = evaluate(
                model, val_loader, criterion, device
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save(model.state_dict(), f"best_fold_{fold}.pth")  # save best
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            scheduler.step()

            # ---- Store ----
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["micro_f1"].append(micro_f1)
            history["macro_f1"].append(macro_f1)

            print(
                f"Fold {fold} | Epoch {epoch+1:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Micro F1: {micro_f1:.4f} | "
                f"Macro F1: {macro_f1:.4f}"
            )

        # Save fold
        torch.save(model.state_dict(), f"efficientnetv2_fold_{fold}.pth")

        best_epoch = int(np.argmax(history["micro_f1"]))

        print("\nBest epoch summary:")
        print(f"Epoch: {best_epoch + 1}")
        print(f"Train Loss: {history['train_loss'][best_epoch]:.4f}")
        print(f"Val Loss: {history['val_loss'][best_epoch]:.4f}")
        print(f"Micro F1: {history['micro_f1'][best_epoch]:.4f}")
        print(f"Macro F1: {history['macro_f1'][best_epoch]:.4f}")
