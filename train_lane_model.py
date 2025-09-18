import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from lane_cnn.model import LaneCNN  # Make sure this model is enhanced

# -------- Dataset Definition -------- #
class LaneDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.resize(image, (120, 120))  # Resize first
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        if self.transform:
            image = self.transform(image)

        label = np.clip(self.labels.iloc[idx, 1], -1.0, 1.0)  # Steering control
        return image, torch.tensor([label], dtype=torch.float32)

# -------- Transform Pipeline -------- #
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to [C, H, W]
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize grayscale image
])

# -------- Dataset and DataLoader -------- #
dataset = LaneDataset('dataset/images', 'dataset/labels.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# -------- Training -------- #
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LaneCNN().to(device)

    # Dummy forward pass to init dynamic layers
    with torch.no_grad():
        model(torch.zeros(1, 1, 120, 120).to(device))

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    print(" Training started...")
    for epoch in range(20):
        model.train()
        total_loss = 0.0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/20 - Loss: {total_loss:.4f}")

    os.makedirs("lane_cnn", exist_ok=True)
    torch.save(model.state_dict(), "lane_cnn/lane_model.pth")
    print(" Model saved to lane_cnn/lane_model.pth")

# -------- Entry Point -------- #
if __name__ == '__main__':
    train_model()