import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image
import pandas as pd
from timm import create_model  # EfficientNetV2
import numpy as np


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
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Map: original_image.jpg -> label_string
        self.label_map = dict(
            zip(csv_df['image'], csv_df['labels'])
        )

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
    epochs = 20
    batch_size = 8
    warmup_epochs = 2

    fold = 0
    for train_idx, val_idx in kf.split(labels):
        fold += 1
        print(f"Training fold {fold}...")

        train_dataset = Subset(LeafDataset(labels, aug_images_dir=aug_images_dir), train_idx)
        val_dataset = Subset(LeafDataset(labels, aug_images_dir=aug_images_dir), val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # ===========================
        # Model
        # ===========================
        model = create_model('efficientnetv2_rw_s', pretrained=True, num_classes=6)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=1e-4,
                                      weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=epochs)

        for epoch in range(epochs):
            # Warmup
            if epoch < warmup_epochs:
                lr_scale = 0.1 + 0.9 * (epoch / warmup_epochs)
                for pg in optimizer.param_groups:
                    pg['lr'] = 1e-4 * lr_scale

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
            print(f"Fold {fold}, Epoch {epoch+1}, Loss: {running_loss/len(train_loader.dataset):.4f}")
            scheduler.step()

        # Save fold
        torch.save(model.state_dict(), f"efficientnetv2_fold{fold}.pth")
