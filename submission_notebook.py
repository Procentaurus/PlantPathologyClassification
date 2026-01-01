# %% [markdown]
# # Plant Pathology 2021 - FGVC8 Submission Notebook
# This notebook loads the best model (fold 2), performs inference on the test images, and creates `submission.csv`.

# %%
import os
import torch
import pandas as pd

from pathlib import Path
from PIL import Image
from torchvision import transforms
from timm import create_model
from torch.utils.data import Dataset, DataLoader


# %% [markdown]
# ## Configuration
BATCH_SIZE = 16
THRESHOLD = 0.5
CLASSES = [
    'healthy', 'scab', 'frog_eye_leaf_spot', 'rust', 'powdery_mildew', 'complex'
]
IMG_HEIGHT = 320
IMG_WIDTH = 480

CHECKPOINT = "/kaggle/input/plant-pathology-2-1/pytorch/default/1/best_fold_2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Dataset Class
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

# %% [markdown]
# ## Load Model
model = create_model(
    "efficientnetv2_rw_s",
    pretrained=False,
    num_classes=len(CLASSES),
)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# %% [markdown]
# ## Prepare Test Data
test_dir = Path("/kaggle/input/plant-pathology-2021-fgvc8/test_images")
dataset = TestDataset(test_dir)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# %% [markdown]
# ## Inference and Submission
results = []

with torch.no_grad():
    for imgs, fnames in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).cpu().numpy()

        for fname, pred in zip(fnames, preds):
            labels = [CLASSES[i] for i, v in enumerate(pred) if v == 1]
            if len(labels) == 0:
                labels = ["healthy"]  # fallback
            results.append({
                "image": fname,
                "labels": " ".join(labels)
            })

submission = pd.DataFrame(results)
submission.to_csv("submission.csv", index=False)
print("submission.csv created ✔")
