import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def normalize(x, eps=1e-6):
    return (x - x.min()) / (x.max() - x.min() + eps)

def hotspot_metrics(pred, gt, percentile=0.99):
    """
    Binary hotspot evaluation
    """
    gt_thresh = torch.quantile(gt[gt > 0], percentile)
    pred_thresh = torch.quantile(pred, percentile)

    gt_bin = (gt >= gt_thresh).float()
    pred_bin = (pred >= pred_thresh).float()

    tp = (gt_bin * pred_bin).sum()
    fp = ((1 - gt_bin) * pred_bin).sum()
    fn = (gt_bin * (1 - pred_bin)).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision.item(), recall.item(), f1.item()



class CongestionDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.files = sorted(os.listdir(feature_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        x = np.load(os.path.join(self.feature_dir, name))   # H,W,C
        y = np.load(os.path.join(self.label_dir, name))     # H,W,1 or H,W

        x = normalize(x)
        y = normalize(y)

        x = torch.tensor(x, dtype=torch.float).permute(2, 0, 1)  # C,H,W
        y = torch.tensor(y, dtype=torch.float).squeeze()         # H,W

        return x, y


class CongestionCNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)).squeeze(1)

def weighted_mae(pred, gt, p=0.95, w=10.0):
    B = pred.shape[0]
    loss = 0.0

    for i in range(B):
        y = gt[i]
        p_hat = pred[i]

        thresh = torch.quantile(y, p)
        weights = torch.ones_like(y)
        weights[y >= thresh] = w

        loss += torch.mean(weights * torch.abs(p_hat - y))

    return loss / B


def topk_overlap(pred, gt, k=0.01):
    overlaps = []

    for i in range(pred.shape[0]):
        p = pred[i].flatten()
        y = gt[i].flatten()

        n = max(1, int(len(p) * k))
        p_top = torch.topk(p, n).indices
        y_top = torch.topk(y, n).indices

        overlap = len(set(p_top.tolist()) & set(y_top.tolist())) / n
        overlaps.append(overlap)

    return np.mean(overlaps)


def train(model, loader, epochs=10):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            pred = model(x)
            loss = weighted_mae(pred, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | Weighted MAE {total_loss/len(loader):.6f}")


def evaluate(model, loader):
    model.eval()
    o1, o5 = [], []
    p_list, r_list, f1_list = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            o1.append(topk_overlap(pred, y, 0.01))
            o5.append(topk_overlap(pred, y, 0.05))

            p, r, f1 = hotspot_metrics(pred, y)
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)

    print("\n=== Evaluation Metrics ===")
    print(f"Top-1% overlap : {np.mean(o1):.3f}")
    print(f"Top-5% overlap : {np.mean(o5):.3f}")
    print(f"Hotspot Precision : {np.mean(p_list):.3f}")
    print(f"Hotspot Recall    : {np.mean(r_list):.3f}")
    print(f"Hotspot F1-score  : {np.mean(f1_list):.3f}")


def visualize(model, dataset, idx=0):
    model.eval()
    x, y = dataset[idx]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).cpu().squeeze()

    y = normalize(y)
    pred = normalize(pred)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Ground Truth")
    plt.imshow(y, cmap='hot')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Prediction")
    plt.imshow(pred, cmap='hot')
    plt.colorbar()

    plt.show()


if __name__ == "__main__":

    FEATURE_DIR = r"D:\CircuitNet_processed\congestion\feature"
    LABEL_DIR   = r"D:\CircuitNet_processed\congestion\label"

    full_dataset = CongestionDataset(FEATURE_DIR, LABEL_DIR)

    # 80 / 20 split (CRITICAL)
    n_train = int(0.8 * len(full_dataset))
    n_val = len(full_dataset) - n_train
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=4, shuffle=False)

    sample_x, _ = full_dataset[0]
    model = CongestionCNN(sample_x.shape[0]).to(device)

    train(model, train_loader)
    evaluate(model, val_loader)
    visualize(model, val_set)
